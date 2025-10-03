# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""Postprocessors class to transform MDETR output according to the downstream task"""

import dataclasses
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from sam3.model import box_ops

from sam3.model.data_misc import BatchedInferenceMetadata, clean_pointers, interpolate

from sam3.train.masks_ops import (
    compute_boundary,
    dilation,
    rle_encode,
    robust_rle_encode,
)

from sam3.train.transforms.basic_for_api import get_size_with_aspect_ratio


class PostProcessNullOp(nn.Module):
    def __init__(self, **kwargs):
        super(PostProcessNullOp).__init__()
        pass

    def forward(self, input):
        pass

    def process_results(self, **kwargs):
        return kwargs["find_stages"]


class PostProcessFlickr(nn.Module):
    """This module converts the model's output for Flickr30k entities evaluation.

    This processor is intended for recall@k evaluation with respect to each phrase in the sentence.
    It requires a description of each phrase (as a binary mask), and returns a sorted list of boxes for each phrase.
    """

    def __init__(self, focal_loss):
        super().__init__()
        self.focal_loss = focal_loss

    @torch.no_grad()
    def forward(self, outputs, target_sizes, positive_map, items_per_batch_element):
        """Perform the computation.
        Args:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            positive_map: tensor [total_nbr_phrases x max_seq_len] for each phrase in the batch, contains a binary
                          mask of the tokens that correspond to that sentence. Note that this is a "collapsed" batch,
                          meaning that all the phrases of all the batch elements are stored sequentially.
            items_per_batch_element: list[int] number of phrases corresponding to each batch element.
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        batch_size = target_sizes.shape[0]

        if self.focal_loss:
            prob = F.sigmoid(out_logits)
        else:
            prob = F.softmax(out_logits, -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox).cpu()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cpu()
        # and from relative [0, 1] to absolute [0, height] coordinates
        boxes = boxes * scale_fct[:, None, :]

        cum_sum = np.cumsum(items_per_batch_element)

        curr_batch_index = 0
        # binarize the map if not already binary
        pos = positive_map > 1e-6

        predicted_boxes = [[] for _ in range(batch_size)]

        # The collapsed batch dimension must match the number of items
        assert len(pos) == cum_sum[-1]

        if len(pos) == 0:
            return predicted_boxes

        # if the first batch elements don't contain elements, skip them.
        while cum_sum[curr_batch_index] == 0:
            curr_batch_index += 1

        for i in range(len(pos)):
            # scores are computed by taking the max over the scores assigned to the positive tokens

            cur_pos = pos[i].unsqueeze(0)
            scores, _ = torch.max(
                cur_pos * prob[curr_batch_index, :, : cur_pos.shape[1]], dim=-1
            )

            _, indices = torch.sort(scores, descending=True)

            assert items_per_batch_element[curr_batch_index] > 0
            predicted_boxes[curr_batch_index].append(
                boxes[curr_batch_index][indices.cpu()].tolist()
            )
            if i == len(pos) - 1:
                break

            # check if we need to move to the next batch element
            while i >= cum_sum[curr_batch_index] - 1:
                curr_batch_index += 1
                assert curr_batch_index < len(cum_sum)

        return predicted_boxes

    def process_results(
        self,
        outputs,
        targets,
        batched_targets,
        detection_results,
        orig_target_sizes,
        model,
    ):
        """Retrieve the results from the postprocessor

        Args:
            postprocessor: Post processor class to use
            outputs: output dictionary from the model
            targets: targets from the dataset
            detection_results: post-processed detection (and possibly segmentation) results
            orig_target_sizes: tensor containing the original dimension of the images
        """
        image_ids = [t["original_img_id"] for t in targets]
        sentence_ids = [t["sentence_id"] for t in targets]
        items_per_batch_element = [t["nb_eval"] for t in targets]
        positive_map_eval = batched_targets["positive_map_eval"]
        flickr_results = self(
            outputs, orig_target_sizes, positive_map_eval, items_per_batch_element
        )
        assert len(flickr_results) == len(image_ids) == len(sentence_ids)
        flickr_res = []
        for im_id, sent_id, output in zip(image_ids, sentence_ids, flickr_results):
            flickr_res.append(
                {"image_id": im_id, "sentence_id": sent_id, "boxes": output}
            )
        return flickr_res


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(
        self,
        max_dets_per_img: int,
        focal_loss: bool = False,
        soft_token: bool = False,
        iou_type="bbox",
        to_cpu: bool = True,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        compute_boundaries: bool = False,
        use_presence: bool = False,
        use_presence_semgseg: bool = False,
        detection_threshold: float = -1.0,
    ) -> None:
        super().__init__()
        self.focal_loss = focal_loss
        self.soft_token = soft_token
        self.max_dets_per_img = max_dets_per_img
        self.iou_type = iou_type
        self.to_cpu = to_cpu
        self.convert_mask_to_rle = convert_mask_to_rle
        self.always_interpolate_masks_on_gpu = always_interpolate_masks_on_gpu
        self.compute_boundaries = compute_boundaries

        self.use_presence = use_presence
        self.use_presence_semgseg = use_presence_semgseg
        if self.use_presence_semgseg:
            assert self.use_presence
        if self.use_presence:
            assert self.soft_token
        self.detection_threshold = detection_threshold

    @torch.no_grad()
    def forward(
        self,
        outputs,
        target_sizes,
        forced_labels=None,
        consistent=False,
        ret_tensordict: bool = False,  # This is experimental
    ):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            forced_labels: tensor of dimension [batch_size] containing the label to force for each image of the batch
                           This is useful when evaluating the model using standard metrics (eg on COCO, LVIS). In that case,
                           we query the model with every possible class label, so we when we pass the predictions to the evaluator,
                           we want to make sure that the predicted "class" matches the one that was queried.
            consistent: whether all target sizes are equal
            ret_tensordict: Experimental argument. If true, return a tensordict.TensorDict instead of a list of dictionaries for easier manipulation.
        """
        if ret_tensordict:
            assert (
                consistent is True
            ), "We don't support returning TensorDict if the outputs have different shapes"  # NOTE: It's possible but we don't support it.
            assert self.detection_threshold <= 0.0, "TODO: implement?"
            try:
                from tensordict import TensorDict
            except ImportError:
                logging.info(
                    "tensordict is not installed. Install by running `pip install tensordict --no-deps`. Falling back by setting `ret_tensordict=False`"
                )
                ret_tensordict = False

        out_bbox = outputs["pred_boxes"] if "pred_boxes" in outputs else None
        out_logits = outputs["pred_logits"] if "pred_logits" in outputs else None
        pred_masks = outputs["pred_masks"] if self.iou_type == "segm" else None

        assert target_sizes.shape[1] == 2
        batch_size = target_sizes.shape[0]

        boxes, scores, labels, keep = self._process_boxes_and_labels(
            target_sizes, forced_labels, out_bbox, out_logits
        )
        assert boxes is None or len(boxes) == batch_size
        out_masks, boundaries, dilated_boundaries = self._process_masks(
            target_sizes, pred_masks, consistent=consistent, keep=keep
        )
        del pred_masks

        if self.use_presence:
            if self.use_presence_semgseg:
                presence_score = outputs["presence_logit"].sigmoid()
            else:
                presence_score = outputs["presence_logit_dec"].sigmoid()
                # print("presence_score", presence_score.shape)
            # presence_score = outputs["presence_logit"].sigmoid()  # [B]
            if presence_score.ndim == 1:
                presence_score = presence_score.unsqueeze(1)  # [B, 1]
            if isinstance(scores, list):
                assert len(presence_score) == len(scores)
                scores = [s * p for s, p in zip(scores, presence_score)]
            else:
                scores = scores * presence_score  # [B, N]

        if boxes is None:
            assert out_masks is not None
            assert (
                not ret_tensordict
            ), "We don't support returning TensorDict if the output does not contain boxes"
            B = len(out_masks)
            boxes = [None] * B
            scores = [None] * B
            labels = [None] * B

        results = {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        }
        if out_masks is not None:
            if self.convert_mask_to_rle:
                results.update(masks_rle=out_masks)
            else:
                results.update(masks=out_masks)
        if boundaries is not None:
            assert dilated_boundaries is not None
            results.update(boundaries=boundaries, dilated_boundaries=dilated_boundaries)

        if ret_tensordict:
            results = TensorDict(results).auto_batch_size_()
            if self.to_cpu:
                results = results.cpu()
        else:
            # Convert a dictonary of lists/tensors to list of dictionaries
            results = [
                dict(zip(results.keys(), res_tuple))
                for res_tuple in zip(*results.values())
            ]

        return results

    def _process_masks(self, target_sizes, pred_masks, consistent=True, keep=None):
        boundaries, dilated_boundaries = None, None
        if pred_masks is None:
            return None, boundaries, dilated_boundaries
        if self.always_interpolate_masks_on_gpu:
            gpu_device = target_sizes.device
            assert gpu_device.type == "cuda"
            pred_masks = pred_masks.to(device=gpu_device)
        if consistent:
            assert keep is None, "TODO: implement?"
            # All masks should have the same shape, expected when processing a batch of size 1
            target_size = target_sizes.unique(dim=0)
            assert target_size.size(0) == 1, "Expecting all target sizes to be equal"
            out_masks = (
                interpolate(
                    pred_masks, target_size.squeeze().tolist(), mode="bilinear"
                ).sigmoid()
                > 0.5
            )
            if self.convert_mask_to_rle:
                raise RuntimeError("TODO: implement?")
            if self.compute_boundaries:
                raise RuntimeError("TODO: implement?")
            if self.to_cpu:
                out_masks = out_masks.cpu()
        else:
            out_masks = [[]] * len(pred_masks)
            if self.compute_boundaries:
                boundaries = [[]] * len(pred_masks)
                dilated_boundaries = [[]] * len(pred_masks)

            assert keep is None or len(keep) == len(pred_masks)
            for i, mask in enumerate(pred_masks):
                h, w = target_sizes[i]
                if keep is not None:
                    mask = mask[keep[i]]
                # Uses the gpu version fist, moves masks to cpu if it fails"""
                try:
                    interpolated = (
                        interpolate(
                            mask.unsqueeze(1), (h, w), mode="bilinear"
                        ).sigmoid()
                        > 0.5
                    )
                except Exception as e:
                    logging.info("Issue found, reverting to CPU mode!")
                    mask_device = mask.device
                    mask = mask.cpu()
                    interpolated = (
                        interpolate(
                            mask.unsqueeze(1), (h, w), mode="bilinear"
                        ).sigmoid()
                        > 0.5
                    )
                    interpolated = interpolated.to(mask_device)
                if self.compute_boundaries:
                    boundary = compute_boundary(interpolated.squeeze(1) > 0.5) > 0

                    # This parameter is hardcoded in TrackEval
                    bound_th = 0.008
                    # Further reducing by 15% because we're doing a square kernel, as opposed to the round kernel in trackeval
                    bound_pix = np.ceil(
                        bound_th * np.linalg.norm(interpolated.shape[-2:]) * 0.85
                    )
                    kernel_size = bound_pix * 2 + 1
                    dilated_boundary = dilation(boundary, kernel_size) > 0
                    boundaries[i] = robust_rle_encode(boundary)
                    dilated_boundaries[i] = robust_rle_encode(dilated_boundary)

                if self.convert_mask_to_rle:
                    out_masks[i] = robust_rle_encode(interpolated.squeeze(1))
                else:
                    out_masks[i] = interpolated
                    if self.to_cpu:
                        out_masks[i] = out_masks[i].cpu()

        return out_masks, boundaries, dilated_boundaries

    def _process_boxes_and_labels(
        self, target_sizes, forced_labels, out_bbox, out_logits
    ):
        if out_bbox is None:
            return None, None, None, None
        assert len(out_logits) == len(target_sizes)
        if self.soft_token:
            if self.focal_loss:
                prob = out_logits.sigmoid()
                scores, labels = prob.max(-1)
            else:
                prob = F.softmax(out_logits, -1)
                if self.to_cpu:
                    prob = prob.cpu()
                scores, labels = prob[..., :-1].max(-1)
                scores = 1 - prob[:, :, -1]

            if forced_labels is None:
                labels = torch.ones_like(labels)
            else:
                labels = forced_labels[:, None].expand_as(labels)

                # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            if self.to_cpu:
                boxes = boxes.cpu()

        else:
            if self.focal_loss:
                prob = out_logits.sigmoid()[...]
            else:
                prob = F.softmax(out_logits, -1)[
                    ..., :-1
                ]  # Remove "no_object" category

            if self.max_dets_per_img > 0:
                topk_values, topk_indexes = torch.topk(
                    prob.reshape(out_logits.shape[0], -1),
                    self.max_dets_per_img,
                    dim=1,
                )
            else:
                # "infinite" detections per image
                topk_values = prob.reshape(out_logits.shape[0], -1)
                topk_indexes = torch.arange(
                    topk_values.shape[1], device=prob.device
                ).repeat(topk_values.shape[0], 1)

            scores = topk_values
            if self.to_cpu:
                scores = scores.cpu()
            topk_boxes = topk_indexes // prob.shape[-1]
            labels = (topk_indexes % prob.shape[-1]).cpu()
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
            if self.to_cpu:
                boxes = boxes.cpu()

            # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        if self.to_cpu:
            scale_fct = scale_fct.cpu()
        boxes = boxes * scale_fct[:, None, :]

        keep = None
        if self.detection_threshold > 0:
            # Filter out the boxes with scores below the detection threshold
            keep = scores > self.detection_threshold
            assert len(keep) == len(boxes) == len(scores) == len(labels)

            boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
            scores = [s[k.to(s.device)] for s, k in zip(scores, keep)]
            labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]

        return boxes, scores, labels, keep

    def process_results(
        self,
        outputs,
        targets,
        batched_targets,
        detection_results,
        orig_target_sizes,
        model,
    ):
        """Retrieve the results from the postprocessor

        Args:
            postprocessor: Post processor class to use
            outputs: output dictionary from the model
            targets: targets from the dataset
            detection_results: post-processed detection (and possibly segmentation) results
            orig_target_sizes: tensor containing the original dimension of the images
        """
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        detection_results = self(outputs, orig_target_sizes)
        assert len(targets) == len(detection_results)

        return {
            target["image_id"].item(): output
            for target, output in zip(targets, detection_results)
        }


class PostProcessAPI(PostProcess):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(
        self,
        max_dets_per_img: int,
        focal_loss: bool = False,
        soft_token: bool = False,
        iou_type="bbox",
        use_original_ids: bool = False,
        use_original_sizes: bool = False,
        to_cpu: bool = True,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        compute_boundaries: bool = False,
        use_presence: bool = False,
        use_presence_semgseg: bool = False,
        detection_threshold: float = -1.0,
    ) -> None:
        """
        Args:
            max_dets_per_img: maximum number of detections per image
            focal_loss: whether to use focal loss or softmax
            soft_token: whether to use soft token or not (used in v0 models)
            iou_type: type of output expected. "bbox" for bounding boxes, "segm" for segmentation masks
            use_original_ids: whether to use the original image ids or the coco ids (used for coco evalutation)
            use_original_sizes: whether to use the original image sizes or the resized ones (used for segmentation evalutation)
            to_cpu: If true, the predictions are transferred to the cpu
        """
        super().__init__(
            max_dets_per_img,
            focal_loss,
            soft_token,
            iou_type,
            to_cpu=to_cpu,
            convert_mask_to_rle=convert_mask_to_rle,
            always_interpolate_masks_on_gpu=always_interpolate_masks_on_gpu,
            compute_boundaries=compute_boundaries,
            use_presence=use_presence,
            use_presence_semgseg=use_presence_semgseg,
            detection_threshold=detection_threshold,
        )
        self.use_original_ids = use_original_ids
        self.use_original_sizes = use_original_sizes

    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):
        if find_stages.loss_stages is not None:
            find_metadatas = [find_metadatas[i] for i in find_stages.loss_stages]
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            detection_results = self(
                outputs,
                (
                    meta.original_size
                    if self.use_original_sizes
                    else torch.ones_like(meta.original_size)
                ),
                forced_labels=(
                    meta.original_category_id if self.use_original_ids else None
                ),
            )
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            assert len(detection_results) == len(ids)
            for img_id, result in zip(ids, detection_results):
                if img_id.item() not in results:
                    results[img_id.item()] = result
                else:
                    assert set(results[img_id.item()].keys()) == set(result.keys())
                    for k in result.keys():
                        if isinstance(result[k], torch.Tensor):
                            results[img_id.item()][k] = torch.cat(
                                [results[img_id.item()][k], result[k]], dim=0
                            )
                        elif isinstance(result[k], list):
                            results[img_id.item()][k] += result[k]
                        else:
                            raise NotImplementedError(
                                f"Unexpected type {type(result[k])} in result."
                            )
        # Prune the results to the max number of detections per image.
        for img_id, result in results.items():
            if (
                self.max_dets_per_img > 0
                and len(result["scores"]) > self.max_dets_per_img
            ):
                _, topk_indexes = torch.topk(
                    result["scores"], self.max_dets_per_img, dim=0
                )
                if self.to_cpu:
                    topk_indexes = topk_indexes.cpu()
                for k in result.keys():
                    if isinstance(results[img_id][k], list):
                        results[img_id][k] = [
                            results[img_id][k][i] for i in topk_indexes.tolist()
                        ]
                    else:
                        results[img_id][k] = results[img_id][k].to(topk_indexes.device)[
                            topk_indexes
                        ]

        return results


class PostProcessAPIVideo(PostProcessAPI):
    """This module converts the video model's output into the format expected by the YT-VIS api"""

    def __init__(
        self,
        *args,
        to_cpu: bool = True,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        prob_thresh: float = 0.5,
        use_presence: bool = False,
        use_presence_semgseg: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            # Here we always set `convert_mask_to_rle=False` in the base `PostProcessAPI` class
            # (so that its `_process_masks` won't return a list of RLEs). If we want to return
            # RLEs for video masklets, we handle it in this `PostProcessAPIVideo` class instead.
            convert_mask_to_rle=False,
            # Here we always set `to_cpu=False` in the base `PostProcessAPI` class (so that
            # the interpolated masks won't be automatically moved back to CPU). We will handle
            # it in this `PostProcessAPIVideo` class instead.
            to_cpu=False,
            always_interpolate_masks_on_gpu=always_interpolate_masks_on_gpu,
            use_presence=use_presence,
            use_presence_semgseg=use_presence_semgseg,
            **kwargs,
        )
        # Expected keys in the output dict to postprocess
        self.EXPECTED_KEYS = [
            "pred_logits",
            "pred_boxes",
            "pred_masks",
        ]
        # Whether to post-process video masklets (under packed representation) into RLE format
        self.convert_mask_to_rle_for_video = convert_mask_to_rle
        self.to_cpu_for_video = to_cpu
        self.prob_thresh = prob_thresh

    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):
        """
        Tracking Postprocessor for SAM 3 video model.
        This function takes in the output of the SAM 3 video model and processes it to extract all the tracklet predictions.
        Args:
            find_stages: A list of tensors representing the output of the SAM 3 video model.
            find_metadatas: A list of BatchedInferenceMetadata objects containing metadata about each frame.
            **kwargs: Additional keyword arguments.
        Returns:
            A dictionary of predcitions with video_id as key.
        """

        # Import tensordict here to avoid global dependency.
        try:
            from tensordict import TensorDict
        except ImportError as e:
            logging.error(
                "tensordict is not installed, please install by running `pip install tensordict --no-deps`"
            )
            raise e
        # Notes and assumptions:
        # 1- This postprocessor assumes results only for a single video.
        # 2- There are N stage outputs corresponding to N video frames
        # 3- Each stage outputs contains PxQ preds, where P is number of prompts and Q is number of object queries. The output should also contain the tracking object ids corresponding to each object query.
        # 4- The tracking object id has a default value of -1, indicating that the object query is not tracking any object in the frame, and hence its predictions can be ingored for a given frame.
        # 5- Some objects may be tracked in a subset of frames only. So, we first extract the predictions in a packed representation (for efficient postprocessing -- specially memory)
        # and then we convert the packed representation into a padded one, where we zero pad boxes/masks for objects that are not tracked in some frames.
        # 6- We refer to objects by an object id, which is a tuple (prompt_idx, obj_id)

        assert len(find_stages) > 0, "There is nothing to postprocess?"
        PROMPT_AXIS, OBJ_QUERY_AXIS = (0, 1)
        NO_OBJ_ID = -1
        # Maps object ID -> [indices in packed tensor]
        tracked_objects_packed_idx = defaultdict(list)
        # Maps object ID -> [indices in padded tensor (abs frame index)]
        tracked_objects_frame_idx = defaultdict(list)
        total_num_preds = 0
        # This will hold the packed representation of predictions.
        vid_preds_packed: List[TensorDict] = []
        vid_masklets_rle_packed: List[Optional[Dict]] = []
        video_id = (
            -1
        )  # We assume single video postprocessing, this ID should be unique in the datapoint.

        for frame_idx, (frame_outs, meta) in enumerate(
            zip(find_stages, find_metadatas)
        ):
            # only store keys we need to extract the results
            frame_outs_td = TensorDict(
                {k: frame_outs[k] for k in self.EXPECTED_KEYS}
            ).auto_batch_size_()  # Shape is [P,Q,...]
            meta_td = TensorDict(
                dataclasses.asdict(meta)
            ).auto_batch_size_()  # Shape is [P,...]
            unique_vid_id = meta.original_image_id.unique()
            assert unique_vid_id.size(0) == 1
            if video_id == -1:
                video_id = unique_vid_id.item()
            else:
                assert (
                    video_id == unique_vid_id.item()
                ), "We can only postprocess one video per datapoint"
            # keeping track of which objects appear in the current frame
            obj_ids_per_frame = frame_outs["pred_object_ids"]
            assert obj_ids_per_frame.size(-1) == frame_outs["pred_logits"].size(-2)
            if self.prob_thresh is not None:
                # only keep the predictions on this frame with probability above the threshold
                # (remove those predictions during the keep-alive period of a tracking query,
                # where its "pred_object_ids" is still the tracked object ID rather than -1)
                pred_probs = frame_outs["pred_logits"].sigmoid().squeeze(-1)
                obj_ids_per_frame = torch.where(
                    pred_probs >= self.prob_thresh, obj_ids_per_frame, NO_OBJ_ID
                )
            tracked_obj_ids_idx = torch.where(obj_ids_per_frame != NO_OBJ_ID)
            # Object id is a tuple of (prompt_idx, obj_id). This is because the model can assign same obj_id for two different prompts.
            tracked_obj_ids = [
                (p_id.item(), obj_ids_per_frame[p_id, q_id].item())
                for p_id, q_id in zip(
                    tracked_obj_ids_idx[PROMPT_AXIS],
                    tracked_obj_ids_idx[OBJ_QUERY_AXIS],
                )
            ]
            if len(tracked_obj_ids) == 0:
                continue
            # For each object, we keep track of the packed and padded (frame index) indices
            for oid in tracked_obj_ids:
                tracked_objects_packed_idx[oid].append(total_num_preds)
                tracked_objects_frame_idx[oid].append(frame_idx)
                total_num_preds += 1

            # Since we have P*Q masks per frame, mask interpolation is the GPU memory bottleneck or time bottleneck in case of cpu processing.
            # Instead, we first extract results only for tracked objects, reducing the number of masks to K = sum_i(tracked_objs_per_ith_prompt), hopefully <<< P*Q
            tracked_objs_outs_td = frame_outs_td[
                tracked_obj_ids_idx
            ]  # [P,Q,...] --> [K,...]
            meta_td = meta_td[tracked_obj_ids_idx[PROMPT_AXIS].cpu()]
            if self.always_interpolate_masks_on_gpu:
                gpu_device = meta_td["original_size"].device
                assert gpu_device.type == "cuda"
                tracked_objs_outs_td = tracked_objs_outs_td.to(device=gpu_device)
            frame_results_td = self(
                tracked_objs_outs_td.unsqueeze(1),
                (
                    meta_td["original_size"]
                    if self.use_original_sizes
                    else torch.ones_like(meta_td["original_size"])
                ),
                forced_labels=(
                    meta_td["original_category_id"] if self.use_original_ids else None
                ),
                consistent=True,
                ret_tensordict=True,
            ).squeeze(1)
            del tracked_objs_outs_td

            # Optionally, remove "masks" from output tensor dict and directly encode them
            # to RLE format under packed representations
            if self.convert_mask_to_rle_for_video:
                interpolated_binary_masks = frame_results_td.pop("masks")
                rle_list = rle_encode(interpolated_binary_masks, return_areas=True)
                vid_masklets_rle_packed.extend(rle_list)
            # Optionally, move output TensorDict to CPU (do this after RLE encoding step above)
            if self.to_cpu_for_video:
                frame_results_td = frame_results_td.cpu()
            vid_preds_packed.append(frame_results_td)

        if len(vid_preds_packed) == 0:
            logging.debug(f"Video {video_id} has no predictions")
            return {video_id: []}

        vid_preds_packed = torch.cat(vid_preds_packed, dim=0)
        ############### Construct a padded representation of the predictions ###############
        num_preds = len(tracked_objects_packed_idx)
        num_frames = len(find_stages)
        # We zero pad any missing prediction
        # NOTE: here, we also have padded tensors for "scores" and "labels", but we overwrite them later.
        padded_frames_results = TensorDict(
            {
                k: torch.zeros(
                    num_preds, num_frames, *v.shape[1:], device=v.device, dtype=v.dtype
                )
                for k, v in vid_preds_packed.items()
            },
            batch_size=[
                num_preds,
                num_frames,
            ],
        )
        padded_frames_results["scores"][...] = -1e8  # a very low score for empty object
        # Track scores and labels of each pred tracklet, only for frames where the model was able to track that object
        tracklet_scores = []
        tracklet_labels = []
        # Optionally, fill the list of RLEs for masklets
        # note: only frames with actual predicted masks (in packed format) will be
        # filled with RLEs; the rest will remains None in results["masks_rle"]
        if self.convert_mask_to_rle_for_video:
            vid_masklets_rle_padded = [[None] * num_frames for _ in range(num_preds)]
        for o_idx, oid in enumerate(tracked_objects_packed_idx):
            oid2packed_idx = tracked_objects_packed_idx[oid]
            oid2padded_idx = tracked_objects_frame_idx[oid]
            obj_packed_results = vid_preds_packed[oid2packed_idx]
            padded_frames_results[o_idx][oid2padded_idx] = obj_packed_results
            if self.convert_mask_to_rle_for_video:
                for packed_idx, padded_idx in zip(oid2packed_idx, oid2padded_idx):
                    vid_masklets_rle_padded[o_idx][padded_idx] = (
                        vid_masklets_rle_packed[packed_idx]
                    )
            # NOTE: We need a single confidence score per tracklet for the mAP metric.
            # We use the average confidence score across time. (How does this impact AP?)
            tracklet_scores.append(obj_packed_results["scores"].mean())
            # We also need to have a unique category Id per tracklet.
            # This is not a problem for phrase AP, however, for mAP we do majority voting across time.
            tracklet_labels.append(obj_packed_results["labels"].mode()[0])

        results = padded_frames_results.to_dict()
        results["scores"] = torch.stack(tracklet_scores, dim=0)
        results["labels"] = torch.stack(tracklet_labels, dim=0)
        if self.convert_mask_to_rle_for_video:
            results["masks_rle"] = vid_masklets_rle_padded
        # we keep the frame-level scores since it's needed by some evaluation scripts
        results["per_frame_scores"] = padded_frames_results["scores"]

        return {video_id: results}


class PostProcessSemSeg(nn.Module):
    """This module converts the model's output to be evaluated for semantic segmentation"""

    def __init__(
        self,
        use_original_ids: bool = False,
        use_presence_head: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
    ) -> None:
        """
        Args:
            use_original_ids: whether to use the original image ids or the coco ids (used for coco evalutation)
        """
        super().__init__()
        self.use_original_ids = use_original_ids
        self.use_presence_head = use_presence_head
        self.always_interpolate_masks_on_gpu = always_interpolate_masks_on_gpu

    def forward(self, outputs, target_sizes, forced_labels):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            forced_labels: tensor of dimension [batch_size] containing the label to force for each image of the batch
                           This is useful when evaluating the model using standard metrics (eg on COCO, LVIS). In that case,
                           we query the model with every possible class label, so we when we pass the predictions to the evaluator,
                           we want to make sure that the predicted "class" matches the one that was queried.
        """
        all_masks = outputs["semantic_seg"]
        if self.always_interpolate_masks_on_gpu:
            gpu_device = target_sizes.device
            assert gpu_device.type == "cuda"
            all_masks = all_masks.to(device=gpu_device)
        weights = torch.ones(len(all_masks), device=all_masks.device)

        if self.use_presence_head:
            assert "presence_logit" in outputs and outputs["presence_logit"] is not None
            weights = outputs["presence_logit"].to(all_masks.device).flatten().sigmoid()

        assert len(all_masks) == len(target_sizes) == len(forced_labels) == len(weights)
        results = []
        for masks, size, forced_label, cur_w in zip(
            all_masks, target_sizes, forced_labels, weights
        ):
            h, w = size
            out_mask = (
                interpolate(masks[None], (h, w), mode="bilinear", align_corners=False)
                .sigmoid()
                .squeeze(0)
            ) * cur_w
            results.append({"masks": out_mask, "labels": forced_label.view(1)})

        return results

    @torch.no_grad()
    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            detection_results = self(
                outputs,
                meta.original_size,
                forced_labels=(
                    meta.original_category_id
                    if self.use_original_ids
                    else torch.ones(len(meta.original_size))
                ),
            )
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            assert len(detection_results) == len(ids)
            for img_id, result in zip(ids, detection_results):
                if img_id.item() not in results:
                    results[img_id.item()] = result
                else:
                    assert set(results[img_id.item()].keys()) == set(result.keys())
                    for k in result.keys():
                        results[img_id.item()][k] = torch.cat(
                            [results[img_id.item()][k], result[k]], dim=0
                        )

        return results


class PostProcessPresence(nn.Module):
    """This module returns the information about the presence of the object in a [0,1] confidence range"""

    def __init__(
        self,
        use_original_ids: bool = False,
    ) -> None:
        """
        Args:
            use_original_ids: whether to use the original image ids or the coco ids (used for coco evalutation)
        """
        super().__init__()
        self.use_original_ids = use_original_ids

    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            assert "presence_logit" in outputs and outputs["presence_logit"] is not None
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            presence_scores = torch.sigmoid(outputs["presence_logit"]).squeeze(-1)
            assert len(presence_scores) == len(ids)
            for img_id, presence_score in zip(ids, presence_scores):
                results[img_id.item()] = {"presence_score": presence_score}

        return results


class PostProcessTracking(PostProcess):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(
        self,
        max_dets_per_img: int,
        focal_loss: bool = False,
        soft_token: bool = False,
        iou_type="bbox",
        force_single_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(max_dets_per_img, focal_loss, soft_token, iou_type, **kwargs)
        self.force_single_mask = force_single_mask

    def process_results(
        self, find_stages, find_metadatas: BatchedInferenceMetadata, **kwargs
    ):
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            if self.force_single_mask:
                scores, labels = outputs["pred_logits"].max(-1)
                m = []
                for i in range(len(outputs["pred_masks"])):
                    score, idx = scores[i].max(0)
                    m.append(outputs["pred_masks"][i][idx])
                outputs["pred_masks"] = torch.stack(m, 0).unsqueeze(1)
            detection_results = self(outputs, meta.original_size, consistent=False)
            assert len(detection_results) == len(meta.coco_image_id)
            results.update(
                {
                    (media_id.item(), object_id.item(), frame_index.item()): result
                    for media_id, object_id, frame_index, result in zip(
                        meta.original_image_id,
                        meta.object_id,
                        meta.frame_index,
                        detection_results,
                    )
                }
            )
        return results


class PostProcessPromptToTrack(PostProcess):
    """This module converts the Prompt to Track (PT) model's output into the format expected by the coco api."""

    def __init__(
        self,
        max_dets_per_img: int,
        focal_loss: bool = False,
        soft_token: bool = False,
        iou_type="bbox",
        force_single_mask: bool = False,
        is_padded=True,
    ) -> None:
        super().__init__(max_dets_per_img, focal_loss, soft_token, iou_type)
        self.force_single_mask = force_single_mask
        self.is_padded = is_padded

    def process_results(
        self, find_stages, find_metadatas: BatchedInferenceMetadata, **kwargs
    ):
        # TODO: Handle Multistep predictions here instead of assuming that outputs are for a single iteration. This is specific to Prompt to Track model
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            if self.is_padded:
                self._remove_bot_right_padding(
                    outputs, meta
                )  # Removes padding and modifies "pred_masks" in output dict
            if self.force_single_mask:
                scores, labels = outputs["pred_logits"].max(-1)
                m = []
                for i in range(len(outputs["pred_masks"])):
                    score, idx = scores[i].max(0)
                    m.append(outputs["pred_masks"][i][idx])
                outputs["pred_masks"] = torch.stack(m, 0).unsqueeze(1)
            detection_results = self(outputs, meta.original_size)
            assert len(detection_results) == len(meta.coco_image_id)
            results.update(
                {
                    (media_id.item(), object_id.item(), frame_index.item()): result
                    for media_id, object_id, frame_index, result in zip(
                        meta.original_image_id,
                        meta.object_id,
                        meta.frame_index,
                        detection_results,
                    )
                }
            )
        return results

    def _remove_bot_right_padding(self, outputs, meta):
        # Assumes Bottom Right padding
        # To remove padding:
        # 1- First resize the mask prediction to the input size after augmentation and padding
        # 2- Remove the extra padding from mask

        # TODO: Remove padding for all masks in the same image in one operation instead of doing it per object.
        pred_mask_wo_padding = []
        for mask, (ih, iw) in zip(outputs["pred_masks_high_res"], meta.original_size):
            # 1- No need to do step 1, since PT model already outputs the high-res mask
            _, H, W = mask.shape
            assert H == W  # square
            # 2- Remove padding
            orig_image_size = (iw.item(), ih.item())
            oh, ow = get_size_with_aspect_ratio(
                orig_image_size, H, max_size=H
            )  # Shape after resize but before padding
            pred_mask_wo_padding.append(mask[..., :oh, :ow])

        outputs["pred_masks"] = pred_mask_wo_padding


class PostProcessCaptioning(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, outputs, targets, model):
        # TODO: Expose text generator as an overridable argument in the constructor.
        # Different captioning evaluators might use different generators.
        decoded_text = model.text_generator.decode(
            encoder_hidden_states=outputs["encoder_hidden_states"],
            mask=outputs["masks"],
            unprojected_context=outputs["context"],
            encoder_pos_embed=outputs["encoder_pos_embed"],
            context_attention_mask=outputs["context_mask"],
            previous_context_embeds=outputs["previous_context_embeds"],
            previous_context_mask=outputs["previous_context_mask"],
        )
        decoded_text = [clean_pointers(t) for t in decoded_text]
        assert len(decoded_text) == len(targets)
        try:
            captioning_res = [
                (
                    {"image_id": int(target["original_img_id"]), "caption": output}
                    if "original_img_id" in target
                    else {"image_id": int(target["image_id"]), "caption": output}
                )
                # {"image_id": target["image_id"].item(), "caption": output}
                for target, output in zip(targets, decoded_text)
            ]
        except:
            captioning_res = [
                {"image_id": target["original_id"], "caption": output}
                for target, output in zip(targets, decoded_text)
            ]
        for c in captioning_res:
            print(c["image_id"], c["caption"])

        return captioning_res

    def process_results(
        self,
        outputs,
        targets,
        batched_targets,
        detection_results,
        orig_target_sizes,
        model,
    ):
        """Retrieve the results from the postprocessor

        Args:
            postprocessor: Post processor class to use
            outputs: output dictionary from the model
            targets: targets from the dataset
            detection_results: post-processed detection (and possibly segmentation) results
            orig_target_sizes: tensor containing the original dimension of the images
        """
        return self(outputs, targets, model)


class PostProcessChain:
    """This meta-postprocessor chains multiple postprocessors together.
    The first postprocessor will return an "out" result, which will be passed to the next postprocessor.
    As a result, all subsequent postprocessors must have an "out" argument in their process_results method.

    If combine_outputs is True, the outputs from the initial postprocessors will also be returned
    """

    def __init__(self, postprocessors: List[nn.Module], combine_outputs: bool = False):
        self.postprocessors = postprocessors
        self.combine_outputs = combine_outputs

    def process_results(self, *args, **kwargs):
        assert "out" not in kwargs

        all_outs = []
        for p in self.postprocessors:
            out = p.process_results(*args, **kwargs)
            all_outs.append(out)
            kwargs["out"] = out

        if self.combine_outputs:
            out = defaultdict(dict)
            # In case of overlap of keys, the later postprocessors have preference
            for partial_out in all_outs:
                for img_key, result in partial_out.items():
                    out[img_key].update(result)

        return out


class PostProcessSemSegFromInstance(nn.Module):
    """This postprocessor allows to convert instance segmentation into semantic segmentation mask.
    This allows semantic evaluation for models that don't have the native capability."""

    def __init__(self, threshold=0.5) -> None:
        super().__init__()
        self.threshold = 0.5

    def process_results(self, out, **kwargs):
        assert isinstance(
            out, dict
        ), "The output of the detection model must be a dictionary."

        for img_key in out:
            new_labels = torch.unique(out[img_key]["labels"], sorted=False)
            new_masks = []

            for label in new_labels.tolist():
                # We'll take masks which have the correct label and enough confidence
                selection_mask = out[img_key]["labels"] == label
                selection_mask = torch.logical_and(
                    selection_mask, out[img_key]["scores"] >= self.threshold
                )

                new_masks.append(
                    (out[img_key]["masks"][selection_mask] > 0.5).squeeze(1).any(0)
                )

            out[img_key]["labels"] = new_labels
            out[img_key]["masks"] = torch.stack(new_masks)

        return out


class PostProcessCounting(nn.Module):
    """This module converts the model's output to be evaluated for counting tasks"""

    def __init__(
        self,
        use_original_ids: bool = False,
        threshold: float = 0.5,
        use_presence: bool = False,
        use_presence_semgseg: bool = False,
    ) -> None:
        """
        Args:
            use_original_ids: whether to use the original image ids or the coco ids
            threshold: threshold for counting (values above this are counted)
        """
        super().__init__()
        self.use_original_ids = use_original_ids
        self.threshold = threshold
        self.use_presence = use_presence
        self.use_presence_semgseg = use_presence_semgseg

    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
        """
        # Extract scores from model outputs and apply sigmoid
        scores = torch.sigmoid(outputs["pred_logits"]).squeeze(-1)  # [B, N]
        if self.use_presence:
            if self.use_presence_semgseg:
                presence_score = outputs["presence_logit"].sigmoid()
            else:
                presence_score = outputs["presence_logit_dec"].sigmoid()
            if presence_score.ndim == 1:
                presence_score = presence_score.unsqueeze(1)  # [B, 1]
            scores = scores * presence_score  # [B, N]

        # Calculate counts by summing values above threshold
        counts = (scores > self.threshold).float().sum(dim=1)

        assert len(counts) == len(target_sizes)
        results = []
        for count in counts:
            results.append({"count": count.item()})

        return results

    @torch.no_grad()
    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            detection_results = self(
                outputs,
                meta.original_size,
            )
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            assert len(detection_results) == len(ids)
            for img_id, result in zip(ids, detection_results):
                results[img_id.item()] = result

        return results
