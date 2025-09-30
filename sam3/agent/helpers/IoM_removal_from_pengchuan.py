from collections import defaultdict
from typing import List, Tuple

import nevergrad as ng
import torch

import torch.nn.functional as F
from onevision.models.data_gen.datapoint import Datapoint
from onevision.models.detr.video_tracking_with_prompt_utils import (
    _get_connected_components_with_padding,
    fill_holes_in_mask_scores,
)

from onevision.scripts.utils.unionfind import UnionFind

from onevision.utils.masks_ops import mask_iom
from onevision.utils.rle import robust_rle_encode
from tqdm import tqdm

from .predictor import SamPredictorDataclass


class SamPredictorWithCleaningDataclass(SamPredictorDataclass):
    """Similar to SamPredictorDataclass but also cleans up the segmentation data.
    In particular:
    - remove small connected components and holes
    - handle occlusion cases
    - Ensure that the intersection between every pair of masks is 0
    - Use a zero-order search to prompt SAM to discover non-overlapping masks

    This module adds two dependencies:
    - Nevergrad for the zero order search (https://github.com/facebookresearch/nevergrad)
    - cc_torch for connected component finding. You can
    install it via the following command (`TORCH_CUDA_ARCH_LIST=8.0` is for A100 GPUs):
    ```
    pip uninstall -y cc_torch; TORCH_CUDA_ARCH_LIST=8.0 pip install git+https://github.com/ronghanghu/cc_torch
    ```
    """

    def __init__(self, model_type: str, ckpt_path: str, ng_opt_points=3):
        """
        Args:
        - model_type: the model type to use [eg vit_h]
        - ckpt_path: the path to the checkpoint
        - hq_sam: whether to use the high-quality version of SAM
        - ng_opt_points: number of points to use in nevergrad-based optimization when looking for non-overlapping masks
        """
        super().__init__(model_type, ckpt_path)
        self.ng_opt_points = ng_opt_points

    def clean_masks(
        self,
        datapoint: Datapoint,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        size: Tuple[int, int],
        input_size: Tuple[int, int],
        img_embeddings_kwargs,
    ):
        """Given a list of masks and boxes, clean up the masks and return the cleaned masks.

        The masks must be provided before thresholding, and should be at a resolution consistent with the boxes.
        """
        assert len(masks) == len(boxes) == len(datapoint.object_data)

        #######################  PASS #1: Ensure that masks are disjoint  #######################
        # In this pass, we ensure that masks corresponding to the same phrase_id are disjoint
        # First step, collect all masks for each phrase_id
        catid2maskids = defaultdict(list)
        for i, obj in enumerate(datapoint.object_data):
            assert len(obj.labels) == 1, "Only one label per object is supported"
            catid2maskids[obj.labels[0].phrase_id].append(i)

        # Second step, for each phrase_id, ensure that the masks are disjoint
        for maskids in catid2maskids.values():
            if len(maskids) == 1:
                continue
            masks_subset = masks[maskids]
            boxes_subset = boxes[maskids]

            # Compute pair-wise overlap metric. We could use IoU, but this doesn't account for cases where
            # a small mask is completely contained within a larger mask (the IoU would be small, but that's misleading)
            # Instead, we use the Intersection over Minimum (IoM) metric, which is the intersection area divided by the
            # area of the smaller mask.
            binary_masks_subset = masks_subset > self.model.mask_threshold
            pairwise_iom = mask_iom(binary_masks_subset, binary_masks_subset)

            # Now, we want to find clusters of masks that are highly overlapping. We'll do a union find
            uf = UnionFind(len(maskids))
            for i in range(len(maskids)):
                for j in range(i + 1, len(maskids)):
                    if pairwise_iom[i, j] > 0.3:
                        uf.unite(i, j)

            # retrieve the components
            components = defaultdict(list)
            for i in range(len(maskids)):
                components[uf.find(i)].append(i)

            # For each component, workout a solution that reduces overlap
            for comp in components.values():
                if len(comp) == 1:
                    continue
                print(
                    f"Found {len(comp)} overlapping masks for phrase_id {datapoint.object_data[maskids[0]].labels[0].phrase_id}"
                )
                print(f"Overlap matrix: {pairwise_iom[comp][:, comp]}")

                try:
                    de_overlap = self.find_non_overlapping_masks(
                        boxes=boxes_subset[comp],
                        other_masks=masks_subset[
                            ~torch.isin(
                                torch.arange(len(masks_subset)), torch.tensor(comp)
                            )
                        ],
                        img_embeddings_kwargs=img_embeddings_kwargs,
                        size=size,
                        input_size=input_size,
                    )
                    if de_overlap is not None:
                        masks_subset[comp] = de_overlap
                except Exception as e:
                    print(f"Error when trying to find non-overlapping masks: {e}")
                    pass

                print(
                    "Overlap matrix after: ",
                    mask_iom(
                        masks_subset[comp] > self.model.mask_threshold,
                        masks_subset[comp] > self.model.mask_threshold,
                    ),
                )

            # Finally, we ensure that masks are non overlapping
            # Specifically, pixel i belongs to mask j if it has the highest value among all masks and is above the threshold
            binary_masks_subset = masks_subset > self.model.mask_threshold

            masks_winner = masks_subset.float().argmax(dim=0)[None]
            ids = torch.arange(len(masks_subset), device=masks_winner.device)
            # Non-winners gets pushed below threshold
            masks_subset[masks_winner != ids[..., None, None]] = (
                self.model.mask_threshold - 0.1
            )

            masks[maskids] = masks_subset.float()

        #######################  PASS #2: Clean up the masks  #######################
        # In this pass, we clean up the masks by removing small connected components and holes

        masks = fill_holes_in_mask_scores(masks.unsqueeze(1).clone(), 100).squeeze(1)

        return masks

    @torch.inference_mode()
    def find_non_overlapping_masks(
        self,
        boxes: torch.Tensor,
        other_masks: torch.Tensor,
        img_embeddings_kwargs,
        size: Tuple[int, int],
        input_size: Tuple[int, int],
    ):
        """Given a list of boxes, workout a set of non-overlapping masks.
        If we are here, it means that naively prompting SAM with the boxes has resulted in overlapping masks.
        To workaround this problem, we do a zero-order search of additional points that can be used to prompt SAM.
        The objective we minimize comprises the following terms:
        - No overlap between pairs of masks
        - No going outside the box, or overlapping with other masks
        - Each mask should touch the 4 sides of its box (this assumes we have relatively tight boxes)
        - Minimize small holes and connected components (sign of a bad mask)
        - Maximize SAM's quality score for the mask

        Args:
        - boxes: torch tensor of shape (N, 4) in xyxy format
        - other_masks: torch tensor of shape (M, H, W) containing other masks to avoid
        - img_embeddings_kwargs: dict containing the image embeddings
        - size: tuple containing the size of the image
        - input_size: tuple containing the size at the input of the model

        Returns:
        - torch tensor of shape (M, H, W) containing non-overlapping masks
        """
        h, w = size

        # Tensor that contains what should definitely be "outside" each mask
        outside = torch.ones(len(boxes), h, w, dtype=torch.bool, device="cuda")
        # To check if a masks is touching the sides of the box, we'll create a mask
        # that has a cost proportional to the distance to the edge
        inside_t = torch.ones(len(boxes), h, w, dtype=torch.float, device="cuda")
        inside_b = torch.ones(len(boxes), h, w, dtype=torch.float, device="cuda")
        inside_l = torch.ones(len(boxes), h, w, dtype=torch.float, device="cuda")
        inside_r = torch.ones(len(boxes), h, w, dtype=torch.float, device="cuda")

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.cpu().tolist())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            # For each box, set the pixels that are definitely outside the box
            outside[i, y1 : y2 + 1, x1 : x2 + 1].fill_(0)

            inside_t[i, y1 : y2 + 1, x1 : x2 + 1] = torch.linspace(
                0, 1, y2 - y1 + 1, device="cuda"
            )[:, None].repeat(1, x2 - x1 + 1)
            inside_b[i, y1 : y2 + 1, x1 : x2 + 1] = torch.linspace(
                1, 0, y2 - y1 + 1, device="cuda"
            )[:, None].repeat(1, x2 - x1 + 1)
            inside_l[i, y1 : y2 + 1, x1 : x2 + 1] = torch.linspace(
                0, 1, x2 - x1 + 1, device="cuda"
            )[None].repeat(y2 - y1 + 1, 1)
            inside_r[i, y1 : y2 + 1, x1 : x2 + 1] = torch.linspace(
                1, 0, x2 - x1 + 1, device="cuda"
            )[None].repeat(y2 - y1 + 1, 1)

        # The extra masks are also outside
        outside |= (other_masks > self.model.mask_threshold).any(dim=0)[None]

        # The prompting of SAM works as follows:
        # - For each box, we'll have a set of positive points.
        # - For a given box, we'll also use as negative points the positive points of all the other boxes
        # - Note that SAM will predict all masks at once
        # If we have N boxes and use M points, for each mask we'll have N*M points (N positive, (N-1)*M negative)
        # For all masks, the prompted points will be the same, sent as [p_0_0, ..., p_0_M, ..., p_N_M]
        # The labels will differ. For the first mask it will be M times 1 followed by (N-1)*M zeros, etc.

        pts_labels = (
            torch.eye(len(boxes), device="cuda")[:, :, None]
            .repeat(1, 1, self.ng_opt_points)
            .flatten(-2)
        )

        # Create the parametrization for the optimization
        ng_params = {}
        for i, box in enumerate(boxes):
            # Ensure that the points are within the box
            ng_params[f"x{i}"] = ng.p.Array(
                shape=(1, self.ng_opt_points), lower=box[0].item(), upper=box[2].item()
            )
            ng_params[f"y{i}"] = ng.p.Array(
                shape=(1, self.ng_opt_points), lower=box[1].item(), upper=box[3].item()
            )

        instrum = ng.p.Instrumentation(**ng_params)
        optimizer = ng.optimizers.NGOpt(
            parametrization=instrum, budget=5000, num_workers=1
        )

        best = 1e9
        for iter_id in tqdm(range(optimizer.budget + 1)):
            if iter_id < optimizer.budget:
                x = optimizer.ask()
            else:
                x = optimizer.provide_recommendation()

            obj = x[1]
            # Reconstruct all the points from the optimizer
            all_pts = []
            for i in range(len(boxes)):
                all_pts.append(
                    torch.stack(
                        [
                            torch.tensor(obj[f"x{i}"].value),
                            torch.tensor(obj[f"y{i}"].value),
                        ],
                        dim=2,
                    )
                )

            all_pts = torch.cat(all_pts, dim=1).repeat(len(boxes), 1, 1).cuda()

            # Compute the embeddings
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(all_pts, pts_labels),
                boxes=None,  # We don't need the boxes here
                masks=None,
            )
            low_res_masks, scores = self.model.mask_decoder(
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                **img_embeddings_kwargs,
            )
            raw_masks = self.model.postprocess_masks(
                low_res_masks, input_size, size
            ).squeeze(1)

            masks = raw_masks > self.model.mask_threshold

            # Compute the loss

            # Loss 1: Ensure that the masks are disjoint
            pairwise_iom = mask_iom(masks, masks) - torch.eye(len(boxes), device="cuda")
            loss_inter = torch.clamp(pairwise_iom, min=0).sum().item() / (
                len(boxes) * (len(boxes) - 1)
            )
            # If the loss is below a threshold, we can nullify it
            if loss_inter < 0.02:
                loss_inter = 0

            # Loss 2: Ensure that the masks are within the box
            assert len(masks) == len(boxes) == len(outside)
            loss_out = torch.diag(mask_iom(masks, outside)).mean().item()
            if loss_out < 0.02:
                loss_out = 0

            # Loss 3: Penalize small holes and connected components
            loss_cc = 0
            cc_labels_fg, _ = _get_connected_components_with_padding(masks.unsqueeze(1))
            cc_labels_fg = cc_labels_fg.squeeze(1)
            cc_labels_bg, _ = _get_connected_components_with_padding(
                torch.logical_not(masks).unsqueeze(1)
            )
            cc_labels_bg = cc_labels_bg.squeeze(1)
            assert len(masks) == len(cc_labels_bg) == len(cc_labels_fg)
            for m, cc_fg, cc_bg in zip(masks, cc_labels_fg, cc_labels_bg):
                threshold = min(100, m.long().sum().item() // 10)
                bc = torch.bincount(cc_fg.flatten())
                # Penalize small connected components (less than 100 pixels)
                loss_cc += ((bc > 0) & (bc < threshold)).sum() / (bc > 0).sum()

                # Remove sparkles, they shouldn't count for the inside loss
                bc[0] = 0
                bc[bc > threshold] = 0
                m[torch.isin(cc_fg, torch.nonzero(bc)[..., 0])] = 0

                # Redo the same with the inverse mask, to penalize small holes
                bc = torch.bincount(cc_bg.flatten())
                loss_cc += ((bc > 0) & (bc < threshold)).sum() / (bc > 0).sum()
            loss_cc = loss_cc.item() / (2 * len(masks))

            # Loss 4: Ensure that the masks are touching the sides of the box
            not_masks = ~masks
            touching_score = 0
            touching_score += (
                (masks * inside_t + not_masks).flatten(-2).min(-1)[0].float()
            )
            touching_score += (
                (masks * inside_b + not_masks).flatten(-2).min(-1)[0].float()
            )
            touching_score += (
                (masks * inside_l + not_masks).flatten(-2).min(-1)[0].float()
            )
            touching_score += (
                (masks * inside_r + not_masks).flatten(-2).min(-1)[0].float()
            )
            loss_touch = 0.25 * touching_score.mean().item()

            # Loss 5: Maximize SAM's quality score
            loss_score = (1 - scores).mean().item()

            loss = loss_inter + loss_out + loss_touch + 0.1 * loss_cc + 0.1 * loss_score

            if loss < best:
                best = loss
                print(
                    f"New best loss: {loss} (inter: {loss_inter}, out: {loss_out}, touch: {loss_touch}, cc: {loss_cc}, score: {loss_score})"
                )

            if iter_id < optimizer.budget:
                optimizer.tell(x, loss)
            else:
                print(
                    f"Final loss: {loss} (inter: {loss_inter}, out: {loss_out}, touch: {loss_touch}, cc: {loss_cc}, score: {loss_score})"
                )
                print("Best loss: ", best)
                if max(loss_inter, loss_out, loss_touch, loss_cc, loss_score) > 0.1:
                    print("Warning: some losses are still high")
                    return None

        return raw_masks

    @torch.inference_mode()
    def forward(self, data: List[Datapoint]) -> List[Datapoint]:
        """Given bounding boxes and image, predict the masks for the objects.

        Required populated fields:
        - image_payload
        - object_data.bbox

        Populating fields:
        - object_data.segmentation
        - object_data.area
        """

        # Optimization: we'll skip images where object_data is empty
        keep = [len(dp.object_data) > 0 for dp in data]
        if sum(keep) == 0:
            return data

        all_imgs = []
        all_boxes = []
        orig_sizes = []
        input_sizes = []
        interm_sizes = []
        ids = []
        for i, dp in enumerate(data):
            if not keep[i]:
                continue
            w, h = dp.image_payload.size
            orig_sizes.append((h, w))
            cur_boxes = torch.tensor([obj.bbox for obj in dp.object_data])
            img, boxes, input_size, interm_size = self.preprocess(
                dp.image_payload, cur_boxes
            )
            all_imgs.append(img)
            all_boxes.append(boxes)
            input_sizes.append(input_size)
            interm_sizes.append(interm_size)
            ids.append(i)

        all_imgs = torch.cat(all_imgs, dim=0)
        features = self.model.image_encoder(all_imgs)

        assert (
            len(all_boxes)
            == len(features)
            == len(orig_sizes)
            == len(input_sizes)
            == len(ids)
            == len(interm_sizes)
            == sum(keep)
        )
        for boxes, feats, cur_id, orig_size, input_size, interm_size in zip(
            all_boxes, features, ids, orig_sizes, input_sizes, interm_sizes
        ):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
            img_embeddings_kwargs = {}

            img_embeddings_kwargs["image_embeddings"] = feats[None]
            img_embeddings_kwargs["image_pe"] = self.model.prompt_encoder.get_dense_pe()

            low_res_masks, _ = self.model.mask_decoder(
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                **img_embeddings_kwargs,
            )

            masks = self.model.postprocess_masks(
                low_res_masks, input_size, interm_size
            ).squeeze(1)
            # masks = masks > self.model.mask_threshold
            masks = self.clean_masks(
                data[cur_id],
                masks,
                boxes,
                interm_size,
                input_size,
                img_embeddings_kwargs,
            )
            try:
                masks = (
                    F.interpolate(
                        masks[None].float(),
                        orig_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    > self.model.mask_threshold
                )
            except Exception as e:
                print(f"Error when resizing masks: {e}")
                torch.cuda.empty_cache()
                masks = (
                    F.interpolate(
                        masks.cpu()[None].float(),
                        orig_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    > self.model.mask_threshold
                )

            areas = masks.cpu().flatten(1).sum(-1)
            assert len(masks) == len(data[cur_id].object_data)

            rles = robust_rle_encode(masks)
            assert len(rles) == len(data[cur_id].object_data) == len(areas)
            for obj, rle, area in zip(data[cur_id].object_data, rles, areas):
                # skip SAM masks for pre-filled segmentation data
                if obj.segmentation != {} and obj.segmentation is not None:
                    continue
                obj.segmentation = rle
                obj.area = area.item()
                obj.source += ". mask generated by refined SAM"

        return data