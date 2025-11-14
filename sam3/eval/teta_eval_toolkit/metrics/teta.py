# fmt: off
# flake8: noqa

"""Track Every Thing Accuracy metric."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .. import _timing
from ._base_metric import _BaseMetric

EPS = np.finfo("float").eps  # epsilon


class TETA(_BaseMetric):
    """TETA metric."""

    def __init__(self, exhaustive=False, config=None):
        """Initialize metric."""
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.0, 0.99, 0.05)
        self.cls_array_labels = np.arange(0.5, 0.99, 0.05)

        self.integer_array_fields = [
            "Loc_TP",
            "Loc_FN",
            "Loc_FP",
            "Cls_TP",
            "Cls_FN",
            "Cls_FP",
        ]
        self.float_array_fields = (
            ["TETA", "LocA", "AssocA", "ClsA"]
            + ["LocRe", "LocPr"]
            + ["AssocRe", "AssocPr"]
            + ["ClsRe", "ClsPr"]
        )
        self.fields = self.float_array_fields + self.integer_array_fields
        self.summary_fields = self.float_array_fields
        self.exhaustive = exhaustive

    def compute_global_assignment(self, data_thr, alpha=0.5):
        """Compute global assignment of TP."""
        res = {
            thr: {t: {} for t in range(data_thr[thr]["num_timesteps"])}
            for thr in data_thr
        }

        for thr in data_thr:
            data = data_thr[thr]
            # return empty result if tracker or gt sequence is empty
            if data["num_tk_overlap_dets"] == 0 or data["num_gt_dets"] == 0:
                return res

            # global alignment score
            ga_score, _, _ = self.compute_global_alignment_score(data)

            # calculate scores for each timestep
            for t, (gt_ids_t, tk_ids_t) in enumerate(
                zip(data["gt_ids"], data["tk_ids"])
            ):
                # get matches optimizing for TETA
                amatch_rows, amatch_cols = self.compute_matches(
                    data, t, ga_score, gt_ids_t, tk_ids_t, alpha=alpha
                )
                gt_ids = [data["gt_id_map"][tid] for tid in gt_ids_t[amatch_rows[0]]]
                matched_ids = [
                    data["tk_id_map"][tid] for tid in tk_ids_t[amatch_cols[0]]
                ]
                res[thr][t] = dict(zip(gt_ids, matched_ids))

        return res

    def eval_sequence_single_thr(self, data, cls, cid2clsname, cls_fp_thr, thr):
        """Computes TETA metric for one threshold for one sequence."""
        res = {}
        class_info_list = []
        for field in self.float_array_fields + self.integer_array_fields:
            if field.startswith("Cls"):
                res[field] = np.zeros(len(self.cls_array_labels), dtype=float)
            else:
                res[field] = np.zeros((len(self.array_labels)), dtype=float)

        # return empty result if tracker or gt sequence is empty
        if data["num_tk_overlap_dets"] == 0:
            res["Loc_FN"] = data["num_gt_dets"] * np.ones(
                (len(self.array_labels)), dtype=float
            )
            if self.exhaustive:
                cls_fp_thr[cls] = data["num_tk_cls_dets"] * np.ones(
                    (len(self.cls_array_labels)), dtype=float
                )
            res = self._compute_final_fields(res)
            return res, cls_fp_thr, class_info_list

        if data["num_gt_dets"] == 0:
            if self.exhaustive:
                cls_fp_thr[cls] = data["num_tk_cls_dets"] * np.ones(
                    (len(self.cls_array_labels)), dtype=float
                )
            res = self._compute_final_fields(res)
            return res, cls_fp_thr, class_info_list

        # global alignment score
        ga_score, gt_id_count, tk_id_count = self.compute_global_alignment_score(data)
        matches_counts = [np.zeros_like(ga_score) for _ in self.array_labels]

        # calculate scores for each timestep
        for t, (gt_ids_t, tk_ids_t, tk_overlap_ids_t, tk_cls_ids_t) in enumerate(
            zip(
                data["gt_ids"],
                data["tk_ids"],
                data["tk_overlap_ids"],
                data["tk_class_eval_tk_ids"],
            )
        ):
            # deal with the case that there are no gt_det/tk_det in a timestep
            if len(gt_ids_t) == 0:
                if self.exhaustive:
                    cls_fp_thr[cls] += len(tk_cls_ids_t)
                continue

            # get matches optimizing for TETA
            amatch_rows, amatch_cols = self.compute_matches(
                data, t, ga_score, gt_ids_t, tk_ids_t, list(self.array_labels)
            )

            # map overlap_ids to original ids.
            if len(tk_overlap_ids_t) != 0:
                sorter = np.argsort(tk_ids_t)
                indexes = sorter[
                    np.searchsorted(tk_ids_t, tk_overlap_ids_t, sorter=sorter)
                ]
                sim_t = data["sim_scores"][t][:, indexes]
                fpl_candidates = tk_overlap_ids_t[(sim_t >= (thr / 100)).any(axis=0)]
                fpl_candidates_ori_ids_t = np.array(
                    [data["tk_id_map"][tid] for tid in fpl_candidates]
                )
            else:
                fpl_candidates_ori_ids_t = []

            if self.exhaustive:
                cls_fp_thr[cls] += len(tk_cls_ids_t) - len(tk_overlap_ids_t)

            # calculate and accumulate basic statistics
            for a, alpha in enumerate(self.array_labels):
                match_row, match_col = amatch_rows[a], amatch_cols[a]
                num_matches = len(match_row)
                matched_ori_ids = set(
                    [data["tk_id_map"][tid] for tid in tk_ids_t[match_col]]
                )
                match_tk_cls = data["tk_classes"][t][match_col]
                wrong_tk_cls = match_tk_cls[match_tk_cls != data["gt_classes"][t]]

                num_class_and_det_matches = np.sum(
                    match_tk_cls == data["gt_classes"][t]
                )

                if alpha >= 0.5:
                    for cid in wrong_tk_cls:
                        if cid in cid2clsname:
                            cname = cid2clsname[cid]
                            cls_fp_thr[cname][a - 10] += 1
                    res["Cls_TP"][a - 10] += num_class_and_det_matches
                    res["Cls_FN"][a - 10] += num_matches - num_class_and_det_matches

                res["Loc_TP"][a] += num_matches
                res["Loc_FN"][a] += len(gt_ids_t) - num_matches
                res["Loc_FP"][a] += len(set(fpl_candidates_ori_ids_t) - matched_ori_ids)

                if num_matches > 0:
                    matches_counts[a][gt_ids_t[match_row], tk_ids_t[match_col]] += 1

        # calculate AssocA, AssocRe, AssocPr
        self.compute_association_scores(res, matches_counts, gt_id_count, tk_id_count)

        # calculate final scores
        res = self._compute_final_fields(res)
        return res, cls_fp_thr, class_info_list

    def compute_global_alignment_score(self, data):
        """Computes global alignment score."""
        num_matches = np.zeros((data["num_gt_ids"], data["num_tk_ids"]))
        gt_id_count = np.zeros((data["num_gt_ids"], 1))
        tk_id_count = np.zeros((1, data["num_tk_ids"]))

        # loop through each timestep and accumulate global track info.
        for t, (gt_ids_t, tk_ids_t) in enumerate(zip(data["gt_ids"], data["tk_ids"])):
            # count potential matches between ids in each time step
            # these are normalized, weighted by match similarity
            sim = data["sim_scores"][t]
            sim_iou_denom = sim.sum(0, keepdims=True) + sim.sum(1, keepdims=True) - sim
            sim_iou = np.zeros_like(sim)
            mask = sim_iou_denom > (0 + EPS)
            sim_iou[mask] = sim[mask] / sim_iou_denom[mask]
            num_matches[gt_ids_t[:, None], tk_ids_t[None, :]] += sim_iou

            # calculate total number of dets for each gt_id and tk_id.
            gt_id_count[gt_ids_t] += 1
            tk_id_count[0, tk_ids_t] += 1

        # Calculate overall Jaccard alignment score between IDs
        ga_score = num_matches / (gt_id_count + tk_id_count - num_matches)
        return ga_score, gt_id_count, tk_id_count

    def compute_matches(self, data, t, ga_score, gt_ids, tk_ids, alpha):
        """Compute matches based on alignment score."""
        sim = data["sim_scores"][t]
        score_mat = ga_score[gt_ids[:, None], tk_ids[None, :]] * sim
        # Hungarian algorithm to find best matches
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        if not isinstance(alpha, list):
            alpha = [alpha]
        alpha_match_rows, alpha_match_cols = [], []
        for a in alpha:
            matched_mask = sim[match_rows, match_cols] >= a - EPS
            alpha_match_rows.append(match_rows[matched_mask])
            alpha_match_cols.append(match_cols[matched_mask])
        return alpha_match_rows, alpha_match_cols

    def compute_association_scores(self, res, matches_counts, gt_id_count, tk_id_count):
        """Calculate association scores for each alpha.

        First calculate scores per gt_id/tk_id combo,
        and then average over the number of detections.
        """
        for a, _ in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(
                1, gt_id_count + tk_id_count - matches_count
            )
            res["AssocA"][a] = np.sum(matches_count * ass_a) / np.maximum(
                1, res["Loc_TP"][a]
            )
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res["AssocRe"][a] = np.sum(matches_count * ass_re) / np.maximum(
                1, res["Loc_TP"][a]
            )
            ass_pr = matches_count / np.maximum(1, tk_id_count)
            res["AssocPr"][a] = np.sum(matches_count * ass_pr) / np.maximum(
                1, res["Loc_TP"][a]
            )

    @_timing.time
    def eval_sequence(self, data, cls, cls_id_name_mapping, cls_fp):
        """Evaluate a single sequence across all thresholds."""
        res = {}
        class_info_dict = {}

        for thr in data:
            res[thr], cls_fp[thr], cls_info = self.eval_sequence_single_thr(
                data[thr], cls, cls_id_name_mapping, cls_fp[thr], thr
            )
            class_info_dict[thr] = cls_info

        return res, cls_fp, class_info_dict

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences."""
        data = {}
        res = {}

        if all_res:
            thresholds = list(list(all_res.values())[0].keys())
        else:
            thresholds = [50]
        for thr in thresholds:
            data[thr] = {}
            for seq_key in all_res:
                data[thr][seq_key] = all_res[seq_key][thr]
        for thr in thresholds:
            res[thr] = self._combine_sequences_thr(data[thr])

        return res

    def _combine_sequences_thr(self, all_res):
        """Combines sequences over each threshold."""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ["AssocRe", "AssocPr", "AssocA"]:
            res[field] = self._combine_weighted_av(
                all_res, field, res, weight_field="Loc_TP"
            )
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty=False):
        """Combines metrics across all classes by averaging over classes.

        If 'ignore_empty' is True, then it only sums over classes
        with at least one gt or predicted detection.
        """
        data = {}
        res = {}
        if all_res:
            thresholds = list(list(all_res.values())[0].keys())
        else:
            thresholds = [50]
        for thr in thresholds:
            data[thr] = {}
            for cls_key in all_res:
                data[thr][cls_key] = all_res[cls_key][thr]
        for thr in data:
            res[thr] = self._combine_classes_class_averaged_thr(
                data[thr], ignore_empty=ignore_empty
            )
        return res

    def _combine_classes_class_averaged_thr(self, all_res, ignore_empty=False):
        """Combines classes over each threshold."""
        res = {}

        def check_empty(val):
            """Returns True if empty."""
            return not (val["Loc_TP"] + val["Loc_FN"] + val["Loc_FP"] > 0 + EPS).any()

        for field in self.integer_array_fields:
            if ignore_empty:
                res_field = {k: v for k, v in all_res.items() if not check_empty(v)}
            else:
                res_field = {k: v for k, v in all_res.items()}
            res[field] = self._combine_sum(res_field, field)

        for field in self.float_array_fields:
            if ignore_empty:
                res_field = [v[field] for v in all_res.values() if not check_empty(v)]
            else:
                res_field = [v[field] for v in all_res.values()]
            res[field] = np.mean(res_field, axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over detections."""
        data = {}
        res = {}
        if all_res:
            thresholds = list(list(all_res.values())[0].keys())
        else:
            thresholds = [50]
        for thr in thresholds:
            data[thr] = {}
            for cls_key in all_res:
                data[thr][cls_key] = all_res[cls_key][thr]
        for thr in data:
            res[thr] = self._combine_classes_det_averaged_thr(data[thr])
        return res

    def _combine_classes_det_averaged_thr(self, all_res):
        """Combines detections over each threshold."""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ["AssocRe", "AssocPr", "AssocA"]:
            res[field] = self._combine_weighted_av(
                all_res, field, res, weight_field="Loc_TP"
            )
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate final metric values.

        This function is used both for both per-sequence calculation,
        and in combining values across sequences.
        """
        # LocA
        res["LocRe"] = res["Loc_TP"] / np.maximum(1, res["Loc_TP"] + res["Loc_FN"])
        res["LocPr"] = res["Loc_TP"] / np.maximum(1, res["Loc_TP"] + res["Loc_FP"])
        res["LocA"] = res["Loc_TP"] / np.maximum(
            1, res["Loc_TP"] + res["Loc_FN"] + res["Loc_FP"]
        )

        # ClsA
        res["ClsRe"] = res["Cls_TP"] / np.maximum(1, res["Cls_TP"] + res["Cls_FN"])
        res["ClsPr"] = res["Cls_TP"] / np.maximum(1, res["Cls_TP"] + res["Cls_FP"])
        res["ClsA"] = res["Cls_TP"] / np.maximum(
            1, res["Cls_TP"] + res["Cls_FN"] + res["Cls_FP"]
        )

        res["ClsRe"] = np.mean(res["ClsRe"])
        res["ClsPr"] = np.mean(res["ClsPr"])
        res["ClsA"] = np.mean(res["ClsA"])

        res["TETA"] = (res["LocA"] + res["AssocA"] + res["ClsA"]) / 3

        return res

    def print_summary_table(self, thr_res, thr, tracker, cls):
        """Prints summary table of results."""
        print("")
        metric_name = self.get_name()
        self._row_print(
            [f"{metric_name}{str(thr)}: {tracker}-{cls}"] + self.summary_fields
        )
        self._row_print(["COMBINED"] + thr_res)
