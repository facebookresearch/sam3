# flake8: noqa

from abc import ABC, abstractmethod

import numpy as np

from .. import _timing
from ..utils import TrackEvalException


class _BaseMetric(ABC):
    @abstractmethod
    def __init__(self):
        self.plottable = False
        self.integer_fields = []
        self.float_fields = []
        self.array_labels = []
        self.integer_array_fields = []
        self.float_array_fields = []
        self.fields = []
        self.summary_fields = []
        self.registered = False

    #####################################################################
    # Abstract functions for subclasses to implement

    @_timing.time
    @abstractmethod
    def eval_sequence(self, data): ...

    @abstractmethod
    def combine_sequences(self, all_res): ...

    @abstractmethod
    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False): ...

    @abstractmethod
    def combine_classes_det_averaged(self, all_res): ...

    def plot_single_tracker_results(self, all_res, tracker, output_folder, cls):
        """Plot results of metrics, only valid for metrics with self.plottable"""
        if self.plottable:
            raise NotImplementedError(
                "plot_results is not implemented for metric %s" % self.get_name()
            )
        else:
            pass

    #####################################################################
    # Helper functions which are useful for all metrics:

    @classmethod
    def get_name(cls):
        return cls.__name__

    @staticmethod
    def _combine_sum(all_res, field):
        """Combine sequence results via sum"""
        return sum([all_res[k][field] for k in all_res.keys()])

    @staticmethod
    def _combine_weighted_av(all_res, field, comb_res, weight_field):
        """Combine sequence results via weighted average"""
        return sum(
            [all_res[k][field] * all_res[k][weight_field] for k in all_res.keys()]
        ) / np.maximum(1.0, comb_res[weight_field])

    def print_table(
        self, table_res, tracker, cls, res_field="COMBINED_SEQ", output_lable="COMBINED"
    ):
        """Prints table of results for all sequences"""
        print("")
        metric_name = self.get_name()
        self._row_print(
            [metric_name + ": " + tracker + "-" + cls] + self.summary_fields
        )
        for seq, results in sorted(table_res.items()):
            if seq.startswith("COMBINED_SEQ"):
                continue
            summary_res = self._summary_row(results)
            self._row_print([seq] + summary_res)
        summary_res = self._summary_row(table_res[res_field])
        self._row_print([output_lable] + summary_res)

    def _summary_row(self, results_):
        vals = []
        for h in self.summary_fields:
            if h in self.float_array_fields:
                vals.append("{0:1.5g}".format(100 * np.mean(results_[h])))
            elif h in self.float_fields:
                vals.append("{0:1.5g}".format(100 * float(results_[h])))
            elif h in self.integer_fields:
                vals.append("{0:d}".format(int(results_[h])))
            else:
                raise NotImplementedError(
                    "Summary function not implemented for this field type."
                )
        return vals

    @staticmethod
    def _row_print(*argv):
        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = "%-35s" % argv[0]
        for v in argv[1:]:
            to_print += "%-10s" % str(v)
        print(to_print)

    def summary_results(self, table_res):
        """Returns a simple summary of final results for a tracker"""
        return dict(
            zip(self.summary_fields, self._summary_row(table_res["COMBINED_SEQ"]))
        )

    def detailed_results(self, table_res):
        """Returns detailed final results for a tracker"""
        # Get detailed field information
        detailed_fields = self.float_fields + self.integer_fields
        for h in self.float_array_fields + self.integer_array_fields:
            for alpha in [int(100 * x) for x in self.array_labels]:
                detailed_fields.append(h + "___" + str(alpha))
            detailed_fields.append(h + "___AUC")

        # Get detailed results
        detailed_results = {}
        for seq, res in table_res.items():
            detailed_row = self._detailed_row(res)
            if len(detailed_row) != len(detailed_fields):
                raise TrackEvalException(
                    "Field names and data have different sizes (%i and %i)"
                    % (len(detailed_row), len(detailed_fields))
                )
            detailed_results[seq] = dict(zip(detailed_fields, detailed_row))
        return detailed_results

    def _detailed_row(self, res):
        detailed_row = []
        for h in self.float_fields + self.integer_fields:
            detailed_row.append(res[h])
        for h in self.float_array_fields + self.integer_array_fields:
            for i, alpha in enumerate([int(100 * x) for x in self.array_labels]):
                detailed_row.append(res[h][i])
            detailed_row.append(np.mean(res[h]))
        return detailed_row
