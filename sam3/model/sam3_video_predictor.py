# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import datetime
import gc
import multiprocessing as mp
import os
import queue
import socket
import sys
import time
import uuid
from contextlib import closing
from typing import List, Optional

import psutil
import torch
from sam3.logger import get_logger
from sam3.model.sam3_base_predictor import Sam3BasePredictor

logger = get_logger(__name__)


class Sam3VideoPredictor(Sam3BasePredictor):
    def __init__(
        self,
        checkpoint_path=None,
        bpe_path=None,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=True,
        async_loading_frames=False,
        video_loader_type="cv2",
        apply_temporal_disambiguation: bool = True,
        compile: bool = False,
    ):
        super().__init__()
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        from sam3.model_builder import build_sam3_video_model

        self.model = (
            build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
                compile=compile,
            )
            .cuda()
            .eval()
        )

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ):
        """Remove an object from tracking (SAM3 uses a simpler remove_object API)."""
        session = self._get_session(session_id)
        inference_state = session["state"]

        self.model.remove_object(
            inference_state=inference_state,
            obj_id=obj_id,
            is_user_action=is_user_action,
        )
        return {"is_success": True}

    def _get_session_stats(self):
        """Get a statistics string for live sessions and their GPU usage."""
        live_session_strs = []
        for sid, s in self._all_inference_states.items():
            nf = s["state"]["num_frames"]
            live_session_strs.append(f"'{sid}' ({nf} frames)")
        joined = ", ".join(live_session_strs)
        mem_alloc = torch.cuda.memory_allocated() // 1024**2
        mem_res = torch.cuda.memory_reserved() // 1024**2
        max_alloc = torch.cuda.max_memory_allocated() // 1024**2
        max_res = torch.cuda.max_memory_reserved() // 1024**2
        return (
            f"live sessions: [{joined}], GPU memory: "
            f"{mem_alloc} MiB used and {mem_res} MiB reserved"
            f" (max over time: {max_alloc} MiB used and {max_res} MiB reserved)"
        )

    def _get_torch_and_gpu_properties(self):
        """Get a string for PyTorch and GPU properties."""
        return (
            f"torch: {torch.__version__} with CUDA arch {torch.cuda.get_arch_list()}, "
            f"GPU device: {torch.cuda.get_device_properties(torch.cuda.current_device())}"
        )


class Sam3VideoPredictorMultiGPU(Sam3VideoPredictor):
    def __init__(self, *model_args, gpus_to_use=None, **model_kwargs):
        if gpus_to_use is None:
            # if not specified, use only the current GPU by default
            gpus_to_use = [torch.cuda.current_device()]

        IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
        if IS_MAIN_PROCESS:
            gpus_to_use = sorted(set(gpus_to_use))
            logger.info(f"using the following GPU IDs: {gpus_to_use}")
            assert len(gpus_to_use) > 0 and all(isinstance(i, int) for i in gpus_to_use)
            assert all(0 <= i < torch.cuda.device_count() for i in gpus_to_use)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = f"{self._find_free_port()}"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = f"{len(gpus_to_use)}"

        self.gpus_to_use = gpus_to_use
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank_str = f"rank={self.rank} with world_size={self.world_size}"
        self.device = torch.device(f"cuda:{self.gpus_to_use[self.rank]}")
        torch.cuda.set_device(self.device)
        self.has_shutdown = False
        if self.rank == 0:
            logger.info("\n\n\n\t*** START loading model on all ranks ***\n\n")

        logger.info(f"loading model on {self.rank_str} -- this could take a while ...")
        super().__init__(*model_args, **model_kwargs)
        logger.info(f"loading model on {self.rank_str} -- DONE locally")

        if self.world_size > 1 and self.rank == 0:
            # start the worker processes *after* the model is loaded in the main process
            # so that the main process can run torch.compile and fill the cache first
            self._start_worker_processes(*model_args, **model_kwargs)
            for rank in range(1, self.world_size):
                self.command_queues[rank].put(("start_nccl_process_group", None))
            self._start_nccl_process_group()

        if self.rank == 0:
            logger.info("\n\n\n\t*** DONE loading model on all ranks ***\n\n")

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        if self.has_shutdown:
            raise RuntimeError(
                "cannot handle request after the predictor has shutdown; please create a new predictor"
            )

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
        if self.has_shutdown:
            raise RuntimeError(
                "cannot handle request after the predictor has shutdown; please create a new predictor"
            )

        # dispatch the request to all worker processes
        if self.world_size > 1 and self.rank == 0:
            for rank in range(1, self.world_size):
                self.command_queues[rank].put((request, True))

        yield from super().handle_stream_request(request)

        if self.world_size > 1:
            torch.distributed.barrier()  # wait for all ranks to finish

    def _start_worker_processes(self, *model_args, **model_kwargs):
        """Start worker processes for handling model inference."""
        world_size = self.world_size
        logger.info(f"spawning {world_size - 1} worker processes")
        # Use "spawn" (instead of "fork") for different PyTorch or CUDA context
        mp_ctx = mp.get_context("spawn")
        self.command_queues = {rank: mp_ctx.Queue() for rank in range(1, world_size)}
        self.result_queues = {rank: mp_ctx.Queue() for rank in range(1, world_size)}
        parent_pid = os.getpid()
        for rank in range(1, world_size):
            # set the environment variables for each worker process
            os.environ["IS_MAIN_PROCESS"] = "0"  # mark this as a worker process
            os.environ["RANK"] = f"{rank}"
            worker_process = mp_ctx.Process(
                target=Sam3VideoPredictorMultiGPU._worker_process_command_loop,
                args=(
                    rank,
                    world_size,
                    self.command_queues[rank],
                    self.result_queues[rank],
                    model_args,
                    model_kwargs,
                    self.gpus_to_use,
                    parent_pid,
                ),
                daemon=True,
            )
            worker_process.start()
        # revert the environment variables for the main process
        os.environ["IS_MAIN_PROCESS"] = "1"
        os.environ["RANK"] = "0"
        # wait for all the worker processes to load the model and collect their PIDs
        self.worker_pids = {}
        for rank in range(1, self.world_size):
            # a large timeout to cover potentially long model loading time due to compilation
            _, worker_pid = self.result_queues[rank].get(timeout=7200)
            self.worker_pids[rank] = worker_pid
        logger.info(f"spawned {world_size - 1} worker processes")

    def _start_nccl_process_group(self):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size == 1:
            return

        logger.debug(f"starting NCCL process group on {rank=} with {world_size=}")
        assert not torch.distributed.is_initialized()
        # use the "env://" init method with environment variables set in start_worker_processes
        # a short 3-min timeout to quickly detect any synchronization failures
        timeout_sec = int(os.getenv("SAM3_COLLECTIVE_OP_TIMEOUT_SEC", "180"))
        timeout = datetime.timedelta(seconds=timeout_sec)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout,
            device_id=self.device,
        )
        # warm-up the NCCL process group by running a dummy all-reduce
        tensor = torch.ones(1024, 1024).cuda()
        torch.distributed.all_reduce(tensor)
        logger.debug(f"started NCCL process group on {rank=} with {world_size=}")

    def _find_free_port(self) -> int:
        """
        Find a free port (a random free port from 1024 to 65535 will be selected)
        https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number)
        """
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @staticmethod
    def _worker_process_command_loop(
        rank,
        world_size,
        command_queue,
        result_queue,
        model_args,
        model_kwargs,
        gpus_to_use,
        parent_pid,
    ):
        """
        The command loop for each worker process. It listens to commands from the main process
        and executes them using the model.
        """
        logger.info(f"starting worker process {rank=} with {world_size=}")
        # verify that the environment variables are set correctly
        assert int(os.environ["IS_MAIN_PROCESS"]) == 0
        assert int(os.environ["RANK"]) == rank
        assert int(os.environ["WORLD_SIZE"]) == world_size
        # load the model in this worker process
        predictor = Sam3VideoPredictorMultiGPU(
            *model_args, gpus_to_use=gpus_to_use, **model_kwargs
        )
        logger.info(f"started worker {rank=} with {world_size=}")
        # return the worker process id to the main process for bookkeeping
        worker_pid = os.getpid()
        result_queue.put(("load_model", worker_pid))

        # wait for the command to start the NCCL process group
        request_type, _ = command_queue.get(timeout=7200)
        assert request_type == "start_nccl_process_group"
        predictor._start_nccl_process_group()

        # keep listening to commands from the main process
        while True:
            try:
                request, is_stream_request = command_queue.get(timeout=5.0)
                if request == "shutdown":
                    logger.info(f"worker {rank=} shutting down")
                    torch.distributed.destroy_process_group()
                    result_queue.put(("shutdown", True))  # acknowledge the shutdown
                    sys.exit(0)

                logger.debug(f"worker {rank=} received request {request['type']=}")
                if is_stream_request:
                    for _ in predictor.handle_stream_request(request):
                        pass  # handle stream requests in a generator fashion
                else:
                    predictor.handle_request(request)
            except queue.Empty:
                # Usually Python's multiprocessing module will shutdown all the daemon worker
                # processes when the main process exits gracefully. However, the user may kill
                # the main process using SIGKILL and thereby leaving no chance for the main process
                # to clean up its daemon child processes. So here we manually check whether the
                # parent process still exists (every 5 sec as in `command_queue.get` timeout).
                if not psutil.pid_exists(parent_pid):
                    logger.info(
                        f"stopping worker {rank=} as its parent process has exited"
                    )
                    sys.exit(1)
            except Exception as e:
                logger.error(f"worker {rank=} exception: {e}", exc_info=True)

    def shutdown(self):
        """Shutdown all worker processes."""
        if self.rank == 0 and self.world_size > 1:
            logger.info(f"shutting down {self.world_size - 1} worker processes")
            for rank in range(1, self.world_size):
                self.command_queues[rank].put(("shutdown", False))
            torch.distributed.destroy_process_group()
            for rank in range(1, self.world_size):
                self.result_queues[rank].get()  # wait for the worker to acknowledge
            logger.info(f"shut down {self.world_size - 1} worker processes")
        self.has_shutdown = True

        super().shutdown()
