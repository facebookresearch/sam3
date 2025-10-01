# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import gc

import logging
import threading
import time
import uuid
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

from sam3.sam3_dense_tracking_builder import build_sam3_dense_tracking_model


class Sam3Model:
    def __init__(
        self,
        bpe_path,
        checkpoint_path,
        has_presence_token=False,
        geo_encoder_use_img_cross_attn=False,
        session_expiration_sec=1200,  # the time (sec) for a session to expire after no activities
        strict_state_dict_loading=True,
        default_output_prob_thresh=0.5,
        async_loading_frames=True,
    ):

        self.session_expiration_sec = session_expiration_sec
        self.default_output_prob_thresh = default_output_prob_thresh
        self.async_loading_frames = async_loading_frames

        self.model = (
            build_sam3_dense_tracking_model(
                bpe_path=bpe_path,
                checkpoint_path=checkpoint_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
            )
            .cuda()
            .eval()
        )

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        request_type = request["type"]
        if request_type == "start_session":
            return self.start_session(
                resource_path=request["resource_path"],
                session_id=request.get("session_id", None),
            )
        elif request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text", None),
                points=request.get("points", None),
                point_labels=request.get("point_labels", None),
                clear_old_points=request.get("clear_old_points", True),
                bounding_boxes=request.get("bounding_boxes", None),
                bounding_box_labels=request.get("bounding_box_labels", None),
                clear_old_boxes=request.get("clear_old_boxes", True),
                output_prob_thresh=request.get(
                    "output_prob_thresh", self.default_output_prob_thresh
                ),
            )
        elif request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        elif request_type == "renew_session":
            return self.renew_session(session_id=request["session_id"])
        elif request_type == "close_session":
            return self.close_session(session_id=request["session_id"])
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    @torch.inference_mode()
    def handle_stream_request(self, request):
        """Dispatch a stream request based on its type."""
        request_type = request["type"]
        if request_type == "propagate_in_video":
            yield from self.propagate_in_video(
                session_id=request["session_id"],
                propagation_direction=request.get("propagation_direction", "both"),
                start_frame_idx=request.get("start_frame_index", None),
                max_frame_num_to_track=request.get("max_frame_num_to_track", None),
                output_prob_thresh=request.get(
                    "output_prob_thresh", self.default_output_prob_thresh
                ),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    def start_session(self, resource_path, session_id=None):
        """
        Start a new inference session on an image or a video. Here `resource_path`
        can be either a path to an image file (for image inference) or an MP4 file
        or directory with JPEG video frames (for video inference).

        If `session_id` is defined, it will be used as identifier for the
        session. If it is not defined, the start_session function will create
        a session id and return it.
        """
        # get an initial inference_state from the model
        inference_state = self.model.init_state(
            resource_path=resource_path, async_loading_frames=self.async_loading_frames
        )
        if not session_id:
            session_id = str(uuid.uuid4())
        _ALL_INFERENCE_STATES[session_id] = {
            "state": inference_state,
            "last_use_time": time.time(),
            "expiration_sec": self.session_expiration_sec,
            "session_id": session_id,
            "start_time": time.time(),
            # "config_file": self.config_file,
            # "checkpoint_file": self.checkpoint_file,
        }
        logger.info(
            f"started new session {session_id}; {_get_session_stats()}; "
            f"{_get_torch_and_gpu_properties()}"
        )
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        clear_old_points: bool = True,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
    ):
        """Add text, box and/or point prompt on a specific video frame."""
        logger.info(
            f"add prompt on frame {frame_idx} in session {session_id}: "
            f"{text=}, {points=}, {point_labels=}, {clear_old_points=}, "
            f"{bounding_boxes=}, {bounding_box_labels=}, {clear_old_boxes=}"
        )
        session = _get_session(session_id)
        inference_state = session["state"]
        _extend_expiration_time(session)

        frame_idx, outputs = self.model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            points=points,
            point_labels=point_labels,
            clear_old_points=clear_old_points,
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
            clear_old_boxes=clear_old_boxes,
            output_prob_thresh=output_prob_thresh,
        )
        logger.info(
            f"got {len(outputs['out_probs'])} objects on frame {frame_idx} in session {session_id}"
        )
        return {"frame_index": frame_idx, "outputs": outputs}

    def propagate_in_video(
        self,
        session_id,
        propagation_direction,
        start_frame_idx,
        max_frame_num_to_track,
        output_prob_thresh,
    ):
        """Propagate the added prompts to get grounding results on all video frames."""
        logger.info(
            f"propagate in video in session {session_id}: "
            f"{propagation_direction=}, {start_frame_idx=}, {max_frame_num_to_track=}"
        )
        try:
            session = _get_session(session_id)
            inference_state = session["state"]
            _extend_expiration_time(session)
            if propagation_direction not in ["both", "forward", "backward"]:
                raise ValueError(
                    f"invalid propagation direction: {propagation_direction}"
                )

            # First doing the forward propagation
            if propagation_direction in ["both", "forward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                    output_prob_thresh=output_prob_thresh,
                    reverse=False,
                ):
                    yield {"frame_index": frame_idx, "outputs": outputs}
            # Then doing the backward propagation (reverse in time)
            if propagation_direction in ["both", "backward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                    output_prob_thresh=output_prob_thresh,
                    reverse=True,
                ):
                    yield {"frame_index": frame_idx, "outputs": outputs}
        finally:
            # Log upon completion (so that e.g. we can see if two propagations happen in parallel).
            # Using `finally` here to log even when the tracking is aborted with GeneratorExit.
            logger.info(
                f"propagation ended in session {session_id}; {_get_session_stats()}"
            )

    def reset_session(self, session_id):
        """Reset the session to its initial state (as when it's initial opened)."""
        logger.info(f"clear all inputs across the video in session {session_id}")
        session = _get_session(session_id)
        inference_state = session["state"]
        _extend_expiration_time(session)
        self.model.reset_state(inference_state)
        return {"is_success": True}

    def renew_session(self, session_id):
        """Renew a session (to update its last usage time and reset its expiration timer)."""
        logger.info(f"renew session {session_id}")
        session = _get_session(session_id)
        _extend_expiration_time(session)
        return {"is_success": True}

    def close_session(self, session_id):
        """
        Close a session. This method is idempotent and can be called multiple
        times on the same "session_id".
        """
        session = _ALL_INFERENCE_STATES.pop(session_id, None)
        if session is None:
            logger.warning(
                f"cannot close session {session_id} as it does not exist (it might have expired); "
                f"{_get_session_stats()}"
            )
        else:
            del session
            gc.collect()
            logger.info(f"removed session {session_id}; {_get_session_stats()}")
        return {"is_success": True}


def _get_session(session_id):
    session = _ALL_INFERENCE_STATES.get(session_id, None)
    if session is None:
        raise RuntimeError(f"Cannot find session {session_id}; it might have expired")
    return session


def _extend_expiration_time(session):
    """Extend the expiration time of a session."""
    session["last_use_time"] = time.time()


def _cleanup_expired_sessions():
    """Clean up expired sessions that haven't been used in a while."""
    while True:
        try:
            current_time = time.time()
            for session_id, session in list(_ALL_INFERENCE_STATES.items()):
                if current_time - session["last_use_time"] > session["expiration_sec"]:
                    # close the expired session
                    _ALL_INFERENCE_STATES.pop(session_id, None)
                    gc.collect()
                    logger.info(
                        f"removed expired session {session_id}; {_get_session_stats()}"
                    )
        except Exception:
            pass  # catch and ignore any errors (just to be prudent)
        time.sleep(30)  # clean up every 30 sec


def _get_session_stats():
    """Get a statistics string for live sessions and their GPU usage."""
    # print both the session ids and their video frame numbers
    live_session_strs = [
        f"'{session_id}' ({session['state']['num_frames']} frames)"
        for session_id, session in _ALL_INFERENCE_STATES.items()
    ]
    session_stats_str = (
        f"live sessions: [{', '.join(live_session_strs)}], GPU memory: "
        f"{torch.cuda.memory_allocated() // 1024**2} MiB used and "
        f"{torch.cuda.memory_reserved() // 1024**2} MiB reserved"
        f" (max over time: {torch.cuda.max_memory_allocated() // 1024**2} MiB used "
        f"and {torch.cuda.max_memory_reserved() // 1024**2} MiB reserved)"
    )
    return session_stats_str


def _get_torch_and_gpu_properties():
    """Get a string for PyTorch and GPU properties (for logging and debugging)."""
    torch_and_gpu_str = (
        f"torch: {torch.__version__} with CUDA arch {torch.cuda.get_arch_list()}, "
        f"GPU device: {torch.cuda.get_device_properties(torch.cuda.current_device())}"
    )
    return torch_and_gpu_str


# a global dictionary that holds all inference states for this model (key is session_id)
_ALL_INFERENCE_STATES = {}
# a daemon thread to clean up expired session
threading.Thread(target=_cleanup_expired_sessions, daemon=True).start()
