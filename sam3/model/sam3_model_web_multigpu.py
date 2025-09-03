import datetime
import os
import queue
import sys

# get the list of all GPUs available for this model based on "CUDA_VISIBLE_DEVICES"
# the API layer (e.g. TorchServe) should set "CUDA_VISIBLE_DEVICES" to specify which GPUs are available to this model
AVAILABLE_GPUS = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
AVAILABLE_GPUS = [int(gpu) for gpu in AVAILABLE_GPUS]
IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
if IS_MAIN_PROCESS:
    # override "CUDA_VISIBLE_DEVICES" to give only GPU 0 to the main process -- need to set it BEFORE importing torch
    # (the worker processes will also override "CUDA_VISIBLE_DEVICES" again to run on other GPUs)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{AVAILABLE_GPUS[0]}"

import logging
import multiprocessing as mp

import socket
import uuid
from contextlib import closing

import psutil
import torch

from sam3.model.sam3_model_web import Sam3Model

logger = logging.getLogger(__name__)

if IS_MAIN_PROCESS:
    logger.info(f"setting up MultiGPU inference with {AVAILABLE_GPUS=}")


def find_free_port() -> int:
    """
    Find a free port (a random free port from 1024 to 65535 will be selected)
    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number)
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class Sam3ModelMultiGPU(Sam3Model):
    def __init__(self, *model_args, **model_kwargs):
        if IS_MAIN_PROCESS:
            self.available_gpus = AVAILABLE_GPUS
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = f"{find_free_port()}"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = f"{len(self.available_gpus)}"
        self.rank = int(os.getenv("RANK"))
        self.world_size = int(os.getenv("WORLD_SIZE"))
        self.rank_str = f"rank={self.rank} with world_size={self.world_size}"

        logger.info(f"loading model on {self.rank_str} -- this could take a while ...")
        super().__init__(*model_args, **model_kwargs)
        logger.info(f"loading model on {self.rank_str} -- DONE locally")

        if self.world_size > 1:
            # start the worker processes *after* the model is loaded in the main process
            # so that the main process can run torch.compile and fill the cache first
            if IS_MAIN_PROCESS:
                self.start_worker_processes(*model_args, **model_kwargs)
            self.start_nccl_process_group()
        logger.info(f"loading model on {self.rank_str} -- DONE on all ranks")

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        # when starting a session, we need to create a session id before dispatching
        # the request to the workers
        if request["type"] == "start_session" and request.get("session_id") is None:
            request["session_id"] = str(uuid.uuid4())
        # dispatch the request to all worker processes
        if self.world_size > 1 and self.rank == 0:
            for rank in range(1, self.world_size):
                self.command_queues[rank].put((request, False))

        response = super().handle_request(request)

        if self.world_size > 1:
            torch.distributed.barrier()  # wait for all ranks to finish
        return response

    @torch.inference_mode()
    def handle_stream_request(self, request):
        """Dispatch a stream request based on its type."""
        # dispatch the request to all worker processes
        if self.world_size > 1 and self.rank == 0:
            for rank in range(1, self.world_size):
                self.command_queues[rank].put((request, True))

        yield from super().handle_stream_request(request)

        if self.world_size > 1:
            torch.distributed.barrier()  # wait for all ranks to finish

    def start_worker_processes(self, *model_args, **model_kwargs):
        """Start worker processes for handling model inference."""
        world_size = self.world_size
        logger.info(f"spawning {world_size} worker processes")
        # Use "spawn" (instead of "fork") for different PyTorch or CUDA context
        mp_ctx = mp.get_context("spawn")
        self.command_queues = {rank: mp_ctx.Queue() for rank in range(1, world_size)}
        self.result_queues = {rank: mp_ctx.Queue() for rank in range(1, world_size)}
        parent_pid = os.getpid()
        for rank in range(1, world_size):
            # set the environment variables for each worker process
            os.environ["IS_MAIN_PROCESS"] = "0"  # mark this as a worker process
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.available_gpus[rank]}"
            os.environ["RANK"] = f"{rank}"
            logger.info(
                f"preparing to spawn worker process {rank=} with {world_size=} with env vars: "
                f"{os.environ['RANK']=}, {os.environ['WORLD_SIZE']=}, {os.environ['CUDA_VISIBLE_DEVICES']=}",
            )
            worker_process = mp_ctx.Process(
                target=_worker_process_command_loop,
                args=(
                    rank,
                    world_size,
                    self.command_queues[rank],
                    self.result_queues[rank],
                    model_args,
                    model_kwargs,
                    self.available_gpus,
                    parent_pid,
                ),
                daemon=True,
            )
            worker_process.start()
        # revert the environment variables for the main process
        os.environ["IS_MAIN_PROCESS"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["RANK"] = f"{AVAILABLE_GPUS[0]}"
        logger.info(f"spawned {world_size} worker processes")

    def start_nccl_process_group(self):
        rank = int(os.getenv("RANK"))
        world_size = int(os.getenv("WORLD_SIZE"))
        if world_size == 1:
            return

        logger.info(f"starting NCCL process group on {rank=} with {world_size=}")
        assert not torch.distributed.is_initialized()
        # use the "env://" init method with environment variables set in start_worker_processes
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            # always "cuda:0" because we set CUDA_VISIBLE_DEVICES to a single GPU in each process
            device_id=torch.device("cuda:0"),
            # a large timeout to cover potentially long model loading time due to compilation
            timeout=datetime.timedelta(seconds=14400),
        )
        logger.info(f"started NCCL process group on {rank=} with {world_size=}")


def _worker_process_command_loop(
    rank,
    world_size,
    command_queue,
    result_queue,
    model_args,
    model_kwargs,
    available_gpus,
    parent_pid,
):
    """
    The command loop for each worker process. It listens to commands from the main process
    and executes them using the model.
    """
    # Load the model in this worker process
    logger.info(
        f"starting worker process {rank=} with {world_size=} with env vars: "
        f"{os.environ['RANK']=} {os.environ['WORLD_SIZE']=} {os.environ['CUDA_VISIBLE_DEVICES']=}",
    )
    # verify that the environment variables are set correctly
    assert not IS_MAIN_PROCESS
    assert int(os.environ["RANK"]) == rank
    assert int(os.environ["WORLD_SIZE"]) == world_size
    assert int(os.environ["CUDA_VISIBLE_DEVICES"]) == available_gpus[rank]
    model_wrapper = Sam3ModelMultiGPU(*model_args, **model_kwargs)
    logger.info(f"started worker {rank=} with {world_size=}")

    # keep listening to commands from the main process
    while True:
        try:
            request, is_stream_request = command_queue.get(timeout=5.0)
            logger.info(f"worker {rank=} received request {request['type']=}")
            if is_stream_request:
                for _ in model_wrapper.handle_stream_request(request):
                    pass  # handle stream requests in a generator fashion
            else:
                model_wrapper.handle_request(request)
        except queue.Empty:
            # Usually Python's multiprocessing module will shutdown all the daemon worker
            # processes when the main process exits gracefully. However, TorchServe kills
            # the main process using SIGKILL and thereby leaving no chance for the main process
            # to clean up its daemon child processes. So here we manually check whether the
            # parent process still exists (every 5 sec as in `command_queue.get` timeout).
            if not psutil.pid_exists(parent_pid):
                logger.info(f"stopping worker {rank=} as its parent process has exited")
                sys.exit(1)
        except Exception as e:
            logger.error(f"worker {rank=} exception: {e}", exc_info=True)
