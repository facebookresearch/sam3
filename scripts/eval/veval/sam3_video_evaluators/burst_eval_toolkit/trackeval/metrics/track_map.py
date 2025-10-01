# flake8: noqa

from functools import partial

import numpy as np

from .. import _timing, utils
from ..utils import TrackEvalException
from ._base_metric import _BaseMetric


class TrackMAP(_BaseMetric):
    """Class which implements the TrackMAP metrics"""

    @staticmethod
    def get_default_metric_config():
        """Default class config values"""
        default_config = {
            "USE_AREA_RANGES": True,  # whether to evaluate for certain area ranges
            "AREA_RANGES": [
                [
                    0**2,
                    32**2,
                ],  # additional area range sets for which TrackMAP is evaluated
                [
                    32**2,
                    96**2,
                ],  # (all area range always included), default values for TAO
                [96**2, 1e5**2],
            ],  # evaluation
            "AREA_RANGE_LABELS": [
                "area_s",
                "area_m",
                "area_l",
            ],  # the labels for the area ranges
            "USE_TIME_RANGES": True,  # whether to evaluate for certain time ranges (length of tracks)
            "TIME_RANGES": [
                [0, 3],
                [3, 10],
                [10, 1e5],
            ],  # additional time range sets for which TrackMAP is evaluated
            # (all time range always included) , default values for TAO evaluation
            "TIME_RANGE_LABELS": [
                "time_s",
                "time_m",
                "time_l",
            ],  # the labels for the time ranges
            "IOU_THRESHOLDS": np.arange(0.5, 0.96, 0.05),  # the IoU thresholds
            "RECALL_THRESHOLDS": np.linspace(
                0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01) + 1), endpoint=True
            ),
            # recall thresholds at which precision is evaluated
            "MAX_DETECTIONS": 0,  # limit the maximum number of considered tracks per sequence (0 for unlimited)
            "PRINT_CONFIG": False,
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(
            config, self.get_default_metric_config(), self.get_name()
        )

        self.num_ig_masks = 1
        self.lbls = ["all"]
        self.use_area_rngs = self.config["USE_AREA_RANGES"]
        if self.use_area_rngs:
            self.area_rngs = self.config["AREA_RANGES"]
            self.area_rng_lbls = self.config["AREA_RANGE_LABELS"]
            self.num_ig_masks += len(self.area_rng_lbls)
            self.lbls += self.area_rng_lbls

        self.use_time_rngs = self.config["USE_TIME_RANGES"]
        if self.use_time_rngs:
            self.time_rngs = self.config["TIME_RANGES"]
            self.time_rng_lbls = self.config["TIME_RANGE_LABELS"]
            self.num_ig_masks += len(self.time_rng_lbls)
            self.lbls += self.time_rng_lbls

        self.array_labels = self.config["IOU_THRESHOLDS"]
        self.rec_thrs = self.config["RECALL_THRESHOLDS"]

        self.maxDet = self.config["MAX_DETECTIONS"]
        self.float_array_fields = ["AP_" + lbl for lbl in self.lbls] + [
            "AR_" + lbl for lbl in self.lbls
        ]
        self.fields = self.float_array_fields
        self.summary_fields = self.float_array_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates GT and Tracker matches for one sequence for TrackMAP metrics. Adapted from
        https://github.com/TAO-Dataset/"""

        # Initialise results to zero for each sequence as the fields are only defined over the set of all sequences
        res = {}
        for field in self.fields:
            res[field] = [0 for _ in self.array_labels]

        gt_ids, dt_ids = data["gt_track_ids"], data["dt_track_ids"]

        if len(gt_ids) == 0 and len(dt_ids) == 0:
            for idx in range(self.num_ig_masks):
                res[idx] = None
            return res

        # get track data
        gt_tr_areas = data.get("gt_track_areas", None) if self.use_area_rngs else None
        gt_tr_lengths = (
            data.get("gt_track_lengths", None) if self.use_time_rngs else None
        )
        gt_tr_iscrowd = data.get("gt_track_iscrowd", None)
        dt_tr_areas = data.get("dt_track_areas", None) if self.use_area_rngs else None
        dt_tr_lengths = (
            data.get("dt_track_lengths", None) if self.use_time_rngs else None
        )
        is_nel = data.get("not_exhaustively_labeled", False)

        # compute ignore masks for different track sets to eval
        gt_ig_masks = self._compute_track_ig_masks(
            len(gt_ids),
            track_lengths=gt_tr_lengths,
            track_areas=gt_tr_areas,
            iscrowd=gt_tr_iscrowd,
        )
        dt_ig_masks = self._compute_track_ig_masks(
            len(dt_ids),
            track_lengths=dt_tr_lengths,
            track_areas=dt_tr_areas,
            is_not_exhaustively_labeled=is_nel,
            is_gt=False,
        )

        boxformat = data.get("boxformat", "xywh")
        ious = self._compute_track_ious(
            data["dt_tracks"],
            data["gt_tracks"],
            iou_function=data["iou_type"],
            boxformat=boxformat,
        )

        for mask_idx in range(self.num_ig_masks):
            gt_ig_mask = gt_ig_masks[mask_idx]

            # Sort gt ignore last
            gt_idx = np.argsort([g for g in gt_ig_mask], kind="mergesort")
            gt_ids = [gt_ids[i] for i in gt_idx]

            ious_sorted = ious[:, gt_idx] if len(ious) > 0 else ious

            num_thrs = len(self.array_labels)
            num_gt = len(gt_ids)
            num_dt = len(dt_ids)

            # Array to store the "id" of the matched dt/gt
            gt_m = np.zeros((num_thrs, num_gt)) - 1
            dt_m = np.zeros((num_thrs, num_dt)) - 1

            gt_ig = np.array([gt_ig_mask[idx] for idx in gt_idx])
            dt_ig = np.zeros((num_thrs, num_dt))

            for iou_thr_idx, iou_thr in enumerate(self.array_labels):
                if len(ious_sorted) == 0:
                    break

                for dt_idx, _dt in enumerate(dt_ids):
                    iou = min([iou_thr, 1 - 1e-10])
                    # information about best match so far (m=-1 -> unmatched)
                    # store the gt_idx which matched for _dt
                    m = -1
                    for gt_idx, _ in enumerate(gt_ids):
                        # if this gt already matched continue
                        if gt_m[iou_thr_idx, gt_idx] > 0:
                            continue
                        # if _dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious_sorted[dt_idx, gt_idx] < iou - np.finfo("float").eps:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious_sorted[dt_idx, gt_idx]
                        m = gt_idx

                    # No match found for _dt, go to next _dt
                    if m == -1:
                        continue

                    # if gt to ignore for some reason update dt_ig.
                    # Should not be used in evaluation.
                    dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                    # _dt match found, update gt_m, and dt_m with "id"
                    dt_m[iou_thr_idx, dt_idx] = gt_ids[m]
                    gt_m[iou_thr_idx, m] = _dt

            dt_ig_mask = dt_ig_masks[mask_idx]

            dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
            dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt

            # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
            dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == -1, dt_ig_mask))
            # store results for given video and category
            res[mask_idx] = {
                "dt_ids": dt_ids,
                "gt_ids": gt_ids,
                "dt_matches": dt_m,
                "gt_matches": gt_m,
                "dt_scores": data["dt_track_scores"],
                "gt_ignore": gt_ig,
                "dt_ignore": dt_ig,
            }

        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences. Computes precision and recall values based on track matches.
        Adapted from https://github.com/TAO-Dataset/
        """
        num_thrs = len(self.array_labels)
        num_recalls = len(self.rec_thrs)

        # -1 for absent categories
        precision = -np.ones((num_thrs, num_recalls, self.num_ig_masks))
        recall = -np.ones((num_thrs, self.num_ig_masks))

        for ig_idx in range(self.num_ig_masks):
            ig_idx_results = [
                res[ig_idx] for res in all_res.values() if res[ig_idx] is not None
            ]

            # Remove elements which are None
            if len(ig_idx_results) == 0:
                continue

            # Append all scores: shape (N,)
            # limit considered tracks for each sequence if maxDet > 0
            if self.maxDet == 0:
                dt_scores = np.concatenate(
                    [res["dt_scores"] for res in ig_idx_results], axis=0
                )

                dt_idx = np.argsort(-dt_scores, kind="mergesort")

                dt_m = np.concatenate(
                    [e["dt_matches"] for e in ig_idx_results], axis=1
                )[:, dt_idx]
                dt_ig = np.concatenate(
                    [e["dt_ignore"] for e in ig_idx_results], axis=1
                )[:, dt_idx]
            elif self.maxDet > 0:
                dt_scores = np.concatenate(
                    [res["dt_scores"][0 : self.maxDet] for res in ig_idx_results],
                    axis=0,
                )

                dt_idx = np.argsort(-dt_scores, kind="mergesort")

                dt_m = np.concatenate(
                    [e["dt_matches"][:, 0 : self.maxDet] for e in ig_idx_results],
                    axis=1,
                )[:, dt_idx]
                dt_ig = np.concatenate(
                    [e["dt_ignore"][:, 0 : self.maxDet] for e in ig_idx_results], axis=1
                )[:, dt_idx]
            else:
                raise Exception(
                    "Number of maximum detections must be >= 0, but is set to %i"
                    % self.maxDet
                )

            gt_ig = np.concatenate([res["gt_ignore"] for res in ig_idx_results])
            # num gt anns to consider
            num_gt = np.count_nonzero(gt_ig == 0)

            if num_gt == 0:
                continue

            tps = np.logical_and(dt_m != -1, np.logical_not(dt_ig))
            fps = np.logical_and(dt_m == -1, np.logical_not(dt_ig))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

            for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                num_tp = len(tp)
                rc = tp / num_gt
                if num_tp:
                    recall[iou_thr_idx, ig_idx] = rc[-1]
                else:
                    recall[iou_thr_idx, ig_idx] = 0

                # np.spacing(1) ~= eps
                pr = tp / (fp + tp + np.spacing(1))
                pr = pr.tolist()

                # Ensure precision values are monotonically decreasing
                for i in range(num_tp - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                # find indices at the predefined recall values
                rec_thrs_insert_idx = np.searchsorted(rc, self.rec_thrs, side="left")

                pr_at_recall = [0.0] * num_recalls

                try:
                    for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                        pr_at_recall[_idx] = pr[pr_idx]
                except IndexError:
                    pass

                precision[iou_thr_idx, :, ig_idx] = np.array(pr_at_recall)

        res = {"precision": precision, "recall": recall}

        # compute the precision and recall averages for the respective alpha thresholds and ignore masks
        for lbl in self.lbls:
            res["AP_" + lbl] = np.zeros((len(self.array_labels)), dtype=float)
            res["AR_" + lbl] = np.zeros((len(self.array_labels)), dtype=float)

        for a_id, alpha in enumerate(self.array_labels):
            for lbl_idx, lbl in enumerate(self.lbls):
                p = precision[a_id, :, lbl_idx]
                if len(p[p > -1]) == 0:
                    mean_p = -1
                else:
                    mean_p = np.mean(p[p > -1])
                res["AP_" + lbl][a_id] = mean_p
                res["AR_" + lbl][a_id] = recall[a_id, lbl_idx]

        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=True):
        """Combines metrics across all classes by averaging over the class values
        Note mAP is not well defined for 'empty classes' so 'ignore empty classes' is always true here.
        """
        res = {}
        for field in self.fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=float)
            field_stacked = np.array([res[field] for res in all_res.values()])

            for a_id, alpha in enumerate(self.array_labels):
                values = field_stacked[:, a_id]
                if len(values[values > -1]) == 0:
                    mean = -1
                else:
                    mean = np.mean(values[values > -1])
                res[field][a_id] = mean
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""

        res = {}
        for field in self.fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=float)
            field_stacked = np.array([res[field] for res in all_res.values()])

            for a_id, alpha in enumerate(self.array_labels):
                values = field_stacked[:, a_id]
                if len(values[values > -1]) == 0:
                    mean = -1
                else:
                    mean = np.mean(values[values > -1])
                res[field][a_id] = mean
        return res

    def _compute_track_ig_masks(
        self,
        num_ids,
        track_lengths=None,
        track_areas=None,
        iscrowd=None,
        is_not_exhaustively_labeled=False,
        is_gt=True,
    ):
        """
        Computes ignore masks for different track sets to evaluate
        :param num_ids: the number of track IDs
        :param track_lengths: the lengths of the tracks (number of timesteps)
        :param track_areas: the average area of a track
        :param iscrowd: whether a track is marked as crowd
        :param is_not_exhaustively_labeled: whether the track category is not exhaustively labeled
        :param is_gt: whether it is gt
        :return: the track ignore masks
        """
        # for TAO tracks for classes which are not exhaustively labeled are not evaluated
        if not is_gt and is_not_exhaustively_labeled:
            track_ig_masks = [
                [1 for _ in range(num_ids)] for i in range(self.num_ig_masks)
            ]
        else:
            # consider all tracks
            track_ig_masks = [[0 for _ in range(num_ids)]]

            # consider tracks with certain area
            if self.use_area_rngs:
                for rng in self.area_rngs:
                    track_ig_masks.append(
                        [
                            0
                            if rng[0] - np.finfo("float").eps
                            <= area
                            <= rng[1] + np.finfo("float").eps
                            else 1
                            for area in track_areas
                        ]
                    )

            # consider tracks with certain duration
            if self.use_time_rngs:
                for rng in self.time_rngs:
                    track_ig_masks.append(
                        [
                            0
                            if rng[0] - np.finfo("float").eps
                            <= length
                            <= rng[1] + np.finfo("float").eps
                            else 1
                            for length in track_lengths
                        ]
                    )

        # for YouTubeVIS evaluation tracks with crowd tag are not evaluated
        if is_gt and iscrowd:
            track_ig_masks = [np.logical_or(mask, iscrowd) for mask in track_ig_masks]

        return track_ig_masks

    @staticmethod
    def _compute_bb_track_iou(dt_track, gt_track, boxformat="xywh"):
        """
        Calculates the track IoU for one detected track and one ground truth track for bounding boxes
        :param dt_track: the detected track (format: dictionary with frame index as keys and
                            numpy arrays as values)
        :param gt_track: the ground truth track (format: dictionary with frame index as keys and
                        numpy array as values)
        :param boxformat: the format of the boxes
        :return: the track IoU
        """
        intersect = 0
        union = 0
        image_ids = set(gt_track.keys()) | set(dt_track.keys())
        for image in image_ids:
            g = gt_track.get(image, None)
            d = dt_track.get(image, None)
            if boxformat == "xywh":
                if d is not None and g is not None:
                    dx, dy, dw, dh = d
                    gx, gy, gw, gh = g
                    w = max(min(dx + dw, gx + gw) - max(dx, gx), 0)
                    h = max(min(dy + dh, gy + gh) - max(dy, gy), 0)
                    i = w * h
                    u = dw * dh + gw * gh - i
                    intersect += i
                    union += u
                elif d is None and g is not None:
                    union += g[2] * g[3]
                elif d is not None and g is None:
                    union += d[2] * d[3]
            elif boxformat == "x0y0x1y1":
                if d is not None and g is not None:
                    dx0, dy0, dx1, dy1 = d
                    gx0, gy0, gx1, gy1 = g
                    w = max(min(dx1, gx1) - max(dx0, gx0), 0)
                    h = max(min(dy1, gy1) - max(dy0, gy0), 0)
                    i = w * h
                    u = (dx1 - dx0) * (dy1 - dy0) + (gx1 - gx0) * (gy1 - gy0) - i
                    intersect += i
                    union += u
                elif d is None and g is not None:
                    union += (g[2] - g[0]) * (g[3] - g[1])
                elif d is not None and g is None:
                    union += (d[2] - d[0]) * (d[3] - d[1])
            else:
                raise TrackEvalException("BoxFormat not implemented")
        if intersect > union:
            raise TrackEvalException(
                "Intersection value > union value. Are the box values corrupted?"
            )
        return intersect / union if union > 0 else 0

    @staticmethod
    def _compute_mask_track_iou(dt_track, gt_track):
        """
        Calculates the track IoU for one detected track and one ground truth track for segmentation masks
        :param dt_track: the detected track (format: dictionary with frame index as keys and
                            pycocotools rle encoded masks as values)
        :param gt_track: the ground truth track (format: dictionary with frame index as keys and
                            pycocotools rle encoded masks as values)
        :return: the track IoU
        """
        # only loaded when needed to reduce minimum requirements
        from pycocotools import mask as mask_utils

        intersect = 0.0
        union = 0.0
        image_ids = set(gt_track.keys()) | set(dt_track.keys())
        for image in image_ids:
            g = gt_track.get(image, None)
            d = dt_track.get(image, None)
            if d and g:
                intersect += mask_utils.area(mask_utils.merge([d, g], True))
                union += mask_utils.area(mask_utils.merge([d, g], False))
            elif not d and g:
                union += mask_utils.area(g)
            elif d and not g:
                union += mask_utils.area(d)
        if union < 0.0 - np.finfo("float").eps:
            raise TrackEvalException("Union value < 0. Are the segmentaions corrupted?")
        if intersect > union:
            raise TrackEvalException(
                "Intersection value > union value. Are the segmentations corrupted?"
            )
        iou = intersect / union if union > 0.0 + np.finfo("float").eps else 0.0
        return iou

    @staticmethod
    def _compute_track_ious(dt, gt, iou_function="bbox", boxformat="xywh"):
        """
        Calculate track IoUs for a set of ground truth tracks and a set of detected tracks
        """

        if len(gt) == 0 and len(dt) == 0:
            return []

        if iou_function == "bbox":
            track_iou_function = partial(
                TrackMAP._compute_bb_track_iou, boxformat=boxformat
            )
        elif iou_function == "mask":
            track_iou_function = partial(TrackMAP._compute_mask_track_iou)
        else:
            raise Exception("IoU function not implemented")

        ious = np.zeros([len(dt), len(gt)])
        for i, j in np.ndindex(ious.shape):
            ious[i, j] = track_iou_function(dt[i], gt[j])
        return ious

    @staticmethod
    def _row_print(*argv):
        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = "%-40s" % argv[0]
        for v in argv[1:]:
            to_print += "%-12s" % str(v)
        print(to_print)
