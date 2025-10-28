# fmt: off
# flake8: noqa

import csv
import os
from collections import OrderedDict


def validate_metrics_list(metrics_list):
    """Get names of metric class and ensures they are unique, further checks that the fields within each metric class
    do not have overlapping names.
    """
    metric_names = [metric.get_name() for metric in metrics_list]
    # check metric names are unique
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException(
            "Code being run with multiple metrics of the same name"
        )
    fields = []
    for m in metrics_list:
        fields += m.fields
    # check metric fields are unique
    if len(fields) != len(set(fields)):
        raise TrackEvalException(
            "Code being run with multiple metrics with fields of the same name"
        )
    return metric_names


def get_track_id_str(ann):
    """Get name of track ID in annotation."""
    if "track_id" in ann:
        tk_str = "track_id"
    elif "instance_id" in ann:
        tk_str = "instance_id"
    elif "scalabel_id" in ann:
        tk_str = "scalabel_id"
    else:
        assert False, "No track/instance ID."
    return tk_str


class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""

    ...
