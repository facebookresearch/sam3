import logging
import math
import os
import random

from collections import defaultdict
from typing import List, MutableSequence, Optional, Union

import torch

from sam3.train.data.modulated_detection_api_v2 import (
    Datapoint,
    FindQuery,
    GetQuery,
    Object,
    QueryContent,
    QueryType,
)


class FindNode:
    def __init__(self, idx, find_children, get_children):
        self.idx = idx
        self.find_children = find_children
        self.get_children = get_children


class FilterDataPointQueries:
    find_ids_to_filter: set = None
    get_ids_to_filter: set = None
    obj_ids_to_filter: set = None  # stored as pairs (img_id, obj_id)

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        """
        Compute set of query ids to keep, for both find and get queries
        """
        raise NotImplementedError

    def _do_filter_query(self, query: Union[FindQuery, GetQuery], query_id: int):
        assert (
            self.find_ids_to_filter is not None and self.get_ids_to_filter is not None
        )
        # print("debugg@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", query)

        # if isinstance(query, FindQuery):
        return query_id in self.find_ids_to_filter
        # elif isinstance(query, GetQuery):
        #    return query_id in self.get_ids_to_filter
        # else:
        #    raise NotImplementedError


class FilterQueryWithText(FilterDataPointQueries):
    """
    Filter all datapoints which have query text in a specified list of exluded terms
    """

    def __init__(
        self, exclude_find_keys: List[str] = None, exclude_get_keys: List[str] = None
    ):
        self.find_filter_keys = exclude_find_keys if exclude_find_keys else []
        self.get_filter_keys = exclude_get_keys if exclude_get_keys else []

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        del_find_ids = []
        del_get_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if f_q.query_text in self.find_filter_keys:
                del_find_ids.append(i)
        for i, g_q in enumerate(datapoint.get_queries):
            if g_q.query_text in self.get_filter_keys:
                del_get_ids.append(i)

        self.find_ids_to_filter = set(del_find_ids)
        self.get_ids_to_filter = set(del_get_ids)


class KeepMaxNumFindQueries(FilterDataPointQueries):
    def __init__(
        self, max_num_find_queries: int, retain_positive_queries: bool = False
    ):
        self.max_num_find_queries = max_num_find_queries
        self.retain_positive_queries = retain_positive_queries

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        num_find_queries = len(datapoint.find_queries)
        if num_find_queries <= self.max_num_find_queries:
            self.find_ids_to_filter = set()  # keep all find queries
            self.get_ids_to_filter = set()  # keep all get queries
            return

        if not self.retain_positive_queries:
            all_find_query_ids = list(range(num_find_queries))
            num_queries_to_filter = max(0, num_find_queries - self.max_num_find_queries)
            query_ids_to_filter = random.sample(
                all_find_query_ids, k=num_queries_to_filter
            )
        else:
            # keep up to max_num_find_queries postive find queries and fill
            # the remaining slots (if any) with negative find queries
            pos_find_ids, neg_find_ids = [], []
            for i, f_q in enumerate(datapoint.find_queries):
                # Negative finds return an empty list of object_ids_output
                if len(f_q.object_ids_output) == 0:
                    neg_find_ids.append(i)
                else:
                    pos_find_ids.append(i)

            if len(pos_find_ids) >= self.max_num_find_queries:
                # we have more positive find queries than `max_num_find_queries`,
                # so we subsample postive find queries and remove all negative find queries
                num_queries_to_filter = len(pos_find_ids) - self.max_num_find_queries
                query_ids_to_filter = random.sample(
                    pos_find_ids, k=num_queries_to_filter
                )
                query_ids_to_filter.extend(neg_find_ids)
            else:
                # we have fewer positive find queries than `max_num_find_queries`
                # so we need to fill the remaining with negative find queries
                num_queries_to_filter = num_find_queries - self.max_num_find_queries
                query_ids_to_filter = random.sample(
                    neg_find_ids, k=num_queries_to_filter
                )

        assert len(query_ids_to_filter) == num_find_queries - self.max_num_find_queries
        self.find_ids_to_filter = set(query_ids_to_filter)
        self.get_ids_to_filter = set(
            []
        )  # Keep all get queries which don't depend on filtered finds


class KeepMaxNumFindQueriesVideo(FilterDataPointQueries):
    def __init__(
        self,
        video_mosaic_max_num_find_queries_per_frame: int,
        retain_positive_queries: bool = False,
    ):
        self.video_mosaic_max_num_find_queries_per_frame = (
            video_mosaic_max_num_find_queries_per_frame
        )
        self.retain_positive_queries = retain_positive_queries

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        num_find_queries = len(datapoint.find_queries)

        findQueries_to_imageIds = defaultdict(list)
        max_queries_per_frame = True
        for i, f_q in enumerate(datapoint.find_queries):
            findQueries_to_imageIds[f_q.image_id].append(i)
            if (
                len(findQueries_to_imageIds[f_q.image_id])
                > self.video_mosaic_max_num_find_queries_per_frame
            ):
                max_queries_per_frame = False

        if max_queries_per_frame:
            self.find_ids_to_filter = set()
            self.get_ids_to_filter = set()  # keep all get queries
            return

        num_frames = len(findQueries_to_imageIds)
        findQueries_0 = findQueries_to_imageIds[0]
        num_find_queries_0 = len(findQueries_0)
        max_num_find_queries_per_frame = (
            self.video_mosaic_max_num_find_queries_per_frame
        )
        if not self.retain_positive_queries:
            find_query_ids_0 = list(range(num_find_queries_0))
            num_queries_to_filter = max(
                0, num_find_queries_0 - max_num_find_queries_per_frame
            )
            query_ids_to_filter_0 = random.sample(
                find_query_ids_0, k=num_queries_to_filter
            )
        else:
            # keep up to max_num_find_queries postive find queries and fill
            # the remaining slots (if any) with negative find queries
            pos_find_ids_0, neg_find_ids_0 = [], []
            for i, f_q_id in enumerate(findQueries_0):
                f_q = datapoint.find_queries[f_q_id]
                # Negative finds return an empty list of object_ids_output
                if len(f_q.object_ids_output) == 0:
                    neg_find_ids_0.append(i)
                else:
                    pos_find_ids_0.append(i)

            if len(pos_find_ids_0) >= max_num_find_queries_per_frame:
                # we have more positive find queries than `max_num_find_queries`,
                # so we subsample postive find queries and remove all negative find queries
                num_queries_to_filter = (
                    len(pos_find_ids_0) - max_num_find_queries_per_frame
                )
                query_ids_to_filter_0 = random.sample(
                    pos_find_ids_0, k=num_queries_to_filter
                )
                query_ids_to_filter_0.extend(neg_find_ids_0)
            else:
                # we have fewer positive find queries than `max_num_find_queries`
                # so we need to fill the remaining with negative find queries
                num_queries_to_filter = (
                    num_find_queries_0 - max_num_find_queries_per_frame
                )
                query_ids_to_filter_0 = random.sample(
                    neg_find_ids_0, k=num_queries_to_filter
                )

        # get based on frame 0 all find queries from all the frames with the same indices as in frame 0
        query_ids_to_filter = []
        for i in range(num_frames):
            findQueries_i = findQueries_to_imageIds[i]
            query_ids_to_filter.extend(
                [findQueries_i[j] for j in query_ids_to_filter_0]
            )

        assert (
            len(query_ids_to_filter)
            == num_find_queries
            - self.video_mosaic_max_num_find_queries_per_frame * num_frames
        )
        self.find_ids_to_filter = set(query_ids_to_filter)
        self.get_ids_to_filter = set(
            []
        )  # Keep all get queries which don't depend on filtered finds


class KeepMaxNumGetQueries(FilterDataPointQueries):
    def __init__(self, max_num_get_queries: int):
        self.max_num_get_queries = max_num_get_queries

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = set()

        num_get_queries = len(datapoint.get_queries)
        if num_get_queries <= self.max_num_get_queries:
            self.get_ids_to_filter = set()  # keep all get queries
            return

        all_get_query_ids = list(range(num_get_queries))
        num_queries_to_filter = max(0, num_get_queries - self.max_num_get_queries)
        query_ids_to_filter = random.sample(all_get_query_ids, k=num_queries_to_filter)

        assert len(query_ids_to_filter) == num_get_queries - self.max_num_find_queries
        self.get_ids_to_filter = set(query_ids_to_filter)


class KeepSemanticFindQueriesOnly(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = {
            i for i, q in enumerate(datapoint.find_queries) if q.input_bbox is not None
        }  # filter (remove) geometric find queries (whose input_bbox is not None)

        # Keep all get queries which don't depend on filtered finds
        self.get_ids_to_filter = set()


class KeepUnaryFindQueriesOnly(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = {
            i
            for i, q in enumerate(datapoint.find_queries)
            if q.ptr_x is not None or q.ptr_y is not None
        }  # filter (remove) relational find queries (whose ptr_x or ptr_y is not None)

        # Keep all get queries which don't depend on filtered finds
        self.get_ids_to_filter = set()


class KeepKNegativeQueries(FilterDataPointQueries):
    """
    Keep a fixed number of negatives per datapoint
    """

    def __init__(self, num_negatives_to_keep: int = None, force=False):
        self.num_negatives_to_keep = num_negatives_to_keep
        self.force = force

    def identify_queries_to_filter(self, datapoint: Datapoint):
        self.obj_ids_to_filter = set()
        neg_find_ids = []
        neg_get_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            # Negative finds return an empty list of object_ids_output
            if len(f_q.object_ids_output) == 0:
                neg_find_ids.append(i)

        for i, g_q in enumerate(datapoint.get_queries):
            # Negative gets return 'no' as the text output
            if g_q.text_output == "no":
                neg_get_ids.append(i)

        n_finds_to_keep, n_gets_to_keep = self._how_many_neg_find_gets_to_keep(
            len(neg_find_ids), len(neg_get_ids), self.num_negatives_to_keep
        )
        n_finds_to_keep = min(n_finds_to_keep, len(neg_find_ids))
        n_gets_to_keep = min(n_gets_to_keep, len(neg_get_ids))
        if (
            not self.force
            and len(datapoint.find_queries) == len(neg_find_ids)
            and n_finds_to_keep == 0
        ):
            # Looks like all finds are negative and we're not keeping any - this would result in
            # an empty datapoint, which is not good. So, keep about a third of the negative finds
            # to avoid this
            n_finds_to_keep = int(math.ceil(len(neg_find_ids) / 3))
        neg_finds_to_keep = random.sample(neg_find_ids, k=n_finds_to_keep)
        neg_gets_to_keep = random.sample(neg_get_ids, k=n_gets_to_keep)

        self.find_ids_to_filter = set(neg_find_ids) - set(neg_finds_to_keep)
        self.get_ids_to_filter = set(neg_get_ids) - set(neg_gets_to_keep)

    def _how_many_neg_find_gets_to_keep(
        self, n_neg_finds: int, n_neg_gets: int, num_negs_to_keep: int
    ):
        """
        Compute how many negatives to keep out from the negative finds and the negative gets
        Such that, in total, there are num_negatives_to_keep negatives in the DataPoint
        """

        # If num_negs_to_keep is greater than total_length, take all from both lists
        total_length = n_neg_finds + n_neg_gets
        if num_negs_to_keep >= total_length or total_length == 0:
            return n_neg_finds, n_neg_gets

        if num_negs_to_keep == 0:
            return 0, 0

        # Step 1: Compute proportions
        proportion_neg_finds = n_neg_finds / total_length
        proportion_neg_gets = n_neg_gets / total_length

        # Step 2: Distribute the n samples based on the proportions
        n1 = round(num_negs_to_keep * proportion_neg_finds)
        n2 = round(num_negs_to_keep * proportion_neg_gets)

        # Step 3: Handle edge cases
        # If desired n1 or n2 exceeds list length
        if n1 > n_neg_finds:
            n1 = n_neg_finds
            n2 = num_negs_to_keep - n1  # adjust n2

        if n2 > n_neg_gets:
            n2 = n_neg_gets
            n1 = num_negs_to_keep - n2  # adjust n1

        return n1, n2


class KeepKNegativeQueriesPerPositive(KeepKNegativeQueries):
    """
    Keep a fixed number of negatives per datapoint
    """

    def __init__(self, negatives_to_keep_per_positive: float = None):
        self.num_negatives_to_keep_per_positive = negatives_to_keep_per_positive
        self.num_negatives_to_keep = None

    def identify_queries_to_filter(self, datapoint: Datapoint):
        self.obj_ids_to_filter = set()
        self.num_negatives_to_keep = self._compute_num_negs_to_keep(datapoint)
        super(KeepKNegativeQueriesPerPositive, self).identify_queries_to_filter(
            datapoint
        )

    def _compute_num_negs_to_keep(self, datapoint: Datapoint):
        num_positives = 0
        for f_q in datapoint.find_queries:
            # Positive finds return a non-empty list of object_ids_output
            if len(f_q.object_ids_output) > 0:
                num_positives += 1

        for g_q in datapoint.get_queries:
            # Negative gets return 'no' as the text output
            if g_q.text_output != "no":
                num_positives += 1

        num_negs_to_keep = int(self.num_negatives_to_keep_per_positive * num_positives)
        if num_positives == 0:
            num_negs_to_keep = int(
                math.ceil(
                    (len(datapoint.find_queries) + len(datapoint.get_queries)) / 3
                )
            )

        return num_negs_to_keep


class FilterZeroBoxQueries(FilterDataPointQueries):
    """
    Filters all find queries which predict a box with zero area
    """

    @staticmethod
    def _is_zero_area_object(obj: Object):
        # Check if height or width of bounding box is zero
        bbox = obj.bbox  # Assume in XYXY format
        height = bbox[..., 3].item() - bbox[..., 1].item()
        width = bbox[..., 2].item() - bbox[..., 0].item()

        return height == 0 or width == 0

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # Find objects with zero area
        # Assume only one image per datapoint
        image_objects = datapoint.images[0].objects
        exclude_objects = {
            obj_id
            for obj_id, obj in enumerate(image_objects)
            if self._is_zero_area_object(obj)
        }

        # If a query predicts an object with zero area, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            f_q_objects = set(f_q.object_ids_output)
            if len(exclude_objects.intersection(f_q_objects)) > 0:
                del_find_ids.append(i)

        self.find_ids_to_filter = set(del_find_ids)
        self.get_ids_to_filter = set()


class FilterFindQueriesWithTooManyOut(FilterDataPointQueries):
    """
    Filters all find queries which have more than a specified number of objects in the output
    """

    def __init__(self, max_num_objects: int):
        self.max_num_objects = max_num_objects

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # If a query predicts more than max_num_objects, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if len(f_q.object_ids_output) > self.max_num_objects:
                del_find_ids.append(i)

        self.find_ids_to_filter = set(del_find_ids)
        self.get_ids_to_filter = set()


class FilterEmptyTargets(FilterDataPointQueries):
    """
    Filters all targets which have zero area
    """

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        for img_id in range(len(datapoint.images)):
            for obj_id, obj in enumerate(datapoint.images[img_id].objects):
                if obj.area < 1e-6:
                    self.obj_ids_to_filter.add((img_id, obj_id))
        self.find_ids_to_filter = set()
        self.get_ids_to_filter = set()


class FilterContentQueries(FilterDataPointQueries):
    """
    Filters all queries with the specified query type and content type
    """

    def __init__(self, content_type_filter: str = None, query_type_filter: str = None):
        try:
            self.query_type_filter = QueryType[query_type_filter]
        except KeyError:
            raise KeyError(
                f"Exception: {query_type_filter} is not a valid type of QueryType. Possible options are: {list(QueryType.__members__.keys())}"
            )
        try:
            self.content_type_filter = QueryContent[content_type_filter]
        except KeyError:
            raise KeyError(
                f"Exception: {content_type_filter} is not a valid type of QueryContent. Possible options are: {list(QueryContent.__members__.keys())}"
            )

    def identify_queries_to_filter(self, datapoint):
        if self.query_type_filter == QueryType.GetQuery:
            list_queries = datapoint.get_queries
        elif self.query_type_filter == QueryType.FindQuery:
            list_queries = datapoint.find_queries
        else:
            raise ValueError(f"{self.query_type_filter} filtering is not implemented")

        # If a query predicts an object with zero area, drop the whole find query
        del_ids = []
        for i, f_q in enumerate(list_queries):
            if f_q.query_content == self.content_type_filter:
                del_ids.append(i)

        self.obj_ids_to_filter = set()
        self.get_ids_to_filter = set()
        self.find_ids_to_filter = set()
        if self.query_type_filter == QueryType.GetQuery:
            self.get_ids_to_filter = set(del_ids)
        elif self.query_type_filter == QueryType.FindQuery:
            self.find_ids_to_filter = set(del_ids)


class FilterNonExhaustiveFindQueries(FilterDataPointQueries):
    """
    Filters all find queries which are non-exhaustive
    """

    def __init__(self, exhaustivity_type: str):
        """
        Args:
            exhaustivity_type: Can be "pixel" or "instance":
                -pixel: filter queries where the union of all segments covers every pixel belonging to target class
                -instance: filter queries where there are non-separable or non annotated instances
        Note that instance exhaustivity implies pixel exhaustivity
        """
        assert exhaustivity_type in ["pixel", "instance"]
        self.exhaustivity_type = exhaustivity_type

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # If a query predicts more than max_num_objects, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if self.exhaustivity_type == "instance":
                if not f_q.is_exhaustive:
                    del_find_ids.append(i)
            elif self.exhaustivity_type == "pixel":
                if f_q.is_pixel_exhaustive is not None and not f_q.is_pixel_exhaustive:
                    del_find_ids.append(i)
            else:
                raise RuntimeError(
                    f"Unknown exhaustivity type {self.exhaustivity_type}"
                )

        self.find_ids_to_filter = set(del_find_ids)
        self.get_ids_to_filter = set()


class FilterInvalidGeometricQueries(FilterDataPointQueries):
    """
    Filters geometric queries whose output got deleted (eg due to cropping)
    """

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # If a query predicts more than max_num_objects, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if f_q.input_bbox is not None and f_q.query_text == "geometric":
                if len(f_q.object_ids_output) == 0:
                    del_find_ids.append(i)
        self.find_ids_to_filter = set(del_find_ids)
        self.get_ids_to_filter = set()


class FlexibleFilterFindGetQueries:
    def __init__(
        self, query_filter: FilterDataPointQueries, enabled: bool = True
    ) -> None:
        self.query_filter = query_filter
        self.enabled = enabled

    def delete_all_children(self, datapoint, find_id, graph_id_to_node):
        if find_id not in graph_id_to_node:
            return
        for g_i in graph_id_to_node[find_id].get_children:
            datapoint.get_queries[g_i] = None
        for f_i in graph_id_to_node[find_id].find_children:
            datapoint.find_queries[f_i] = None
            self.delete_all_children(datapoint, f_i, graph_id_to_node)

    def get_parent_node(self, node_id, graph_id_to_node):
        if node_id not in graph_id_to_node:
            graph_id_to_node[node_id] = FindNode(node_id, [], [])

        return graph_id_to_node[node_id]

    def __call__(self, datapoint, **kwargs):
        if not self.enabled:
            return datapoint

        # Identify all queries to filter
        self.query_filter.identify_queries_to_filter(datapoint=datapoint)

        # We run an additional closure step: if there are objects that are filtered, we remove all queries that are
        # pointing to them
        for i, q in enumerate(datapoint.find_queries):
            if q.ptr_x is not None:
                pointed_query = datapoint.find_queries[q.ptr_x.query_id]
                pointed_obj = (pointed_query.image_id, q.ptr_x.object_id)
                if pointed_obj in self.query_filter.obj_ids_to_filter:
                    self.query_filter.find_ids_to_filter.add(i)

            if q.ptr_y is not None:
                pointed_query = datapoint.find_queries[q.ptr_y.query_id]
                pointed_obj = (pointed_query.image_id, q.ptr_y.object_id)
                if pointed_obj in self.query_filter.obj_ids_to_filter:
                    self.query_filter.find_ids_to_filter.add(i)

        for i, q in enumerate(datapoint.get_queries):
            if q.ptr_x is not None:
                pointed_query = datapoint.find_queries[q.ptr_x.query_id]
                pointed_obj = (pointed_query.image_id, q.ptr_x.object_id)
                if pointed_obj in self.query_filter.obj_ids_to_filter:
                    self.query_filter.get_ids_to_filter.add(i)

            if q.ptr_y is not None:
                pointed_query = datapoint.find_queries[q.ptr_y.query_id]
                pointed_obj = (pointed_query.image_id, q.ptr_y.object_id)
                if pointed_obj in self.query_filter.obj_ids_to_filter:
                    self.query_filter.get_ids_to_filter.add(i)

        # build find graph
        graph_id_to_node = {}
        for i, q in enumerate(datapoint.find_queries):
            if q.ptr_x is not None:
                parent_node = self.get_parent_node(q.ptr_x.query_id, graph_id_to_node)
                parent_node.find_children.append(i)

            if q.ptr_y is not None:
                parent_node = self.get_parent_node(q.ptr_y.query_id, graph_id_to_node)
                parent_node.find_children.append(i)

        for i, q in enumerate(datapoint.get_queries):
            if q.ptr_x is not None:
                parent_node = self.get_parent_node(q.ptr_x.query_id, graph_id_to_node)
                parent_node.get_children.append(i)

            if q.ptr_y is not None:
                parent_node = self.get_parent_node(q.ptr_y.query_id, graph_id_to_node)
                parent_node.get_children.append(i)

        del_find_ids = []
        del_get_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if self.query_filter._do_filter_query(f_q, i):
                datapoint.find_queries[i] = None
                del_find_ids.append(i)
        for i, g_q in enumerate(datapoint.get_queries):
            if self.query_filter._do_filter_query(g_q, i):
                datapoint.get_queries[i] = None
                del_get_ids.append(i)

        for d_f_i in del_find_ids:
            self.delete_all_children(datapoint, d_f_i, graph_id_to_node)

        new_find_queries = []
        new_get_queries = []

        find_old_to_new_map = {}
        get_old_to_new_map = {}

        find_counter = 0
        get_counter = 0

        for i, f_q in enumerate(datapoint.find_queries):
            if f_q is not None:
                find_old_to_new_map[i] = find_counter
                find_counter += 1
                new_find_queries.append(f_q)

        for i, g_q in enumerate(datapoint.get_queries):
            if g_q is not None:
                get_old_to_new_map[i] = get_counter
                get_counter += 1
                new_get_queries.append(g_q)

        for n_f_q in new_find_queries:
            if n_f_q.ptr_x is not None:
                n_f_q.ptr_x.query_id = find_old_to_new_map[n_f_q.ptr_x.query_id]

            if n_f_q.ptr_y is not None:
                n_f_q.ptr_y.query_id = find_old_to_new_map[n_f_q.ptr_y.query_id]

        start_with_zero_check = False
        for n_f_q in new_find_queries:
            if n_f_q.query_processing_order == 0:
                start_with_zero_check = True
                break

        if len(new_find_queries) == 0:
            start_with_zero_check = True

        assert (
            start_with_zero_check
        ), "Invalid Find queries, they need to start at query_processing_order = 0"

        for n_g_q in new_get_queries:
            if n_g_q.ptr_x is not None:
                n_g_q.ptr_x.query_id = find_old_to_new_map[n_g_q.ptr_x.query_id]

            if n_g_q.ptr_y is not None:
                n_g_q.ptr_y.query_id = find_old_to_new_map[n_g_q.ptr_y.query_id]

        datapoint.find_queries = new_find_queries
        datapoint.get_queries = new_get_queries

        if len(datapoint.find_queries) == 0:
            print("Warning: No find queries left in datapoint, this is not allowed")
            print("Filtering function:", self.query_filter)
            print("Datapoint:", datapoint)
            raise ValueError

        # The deletion may have removed intermediate steps, so we need to remap to make them contiguous again
        all_stages = sorted(
            list(set(q.query_processing_order for q in datapoint.find_queries))
        )
        stage_map = {qpo: i for i, qpo in enumerate(all_stages)}
        for i in range(len(datapoint.find_queries)):
            qpo = datapoint.find_queries[i].query_processing_order
            datapoint.find_queries[i].query_processing_order = stage_map[qpo]

        # Final step, clear up objects that are not used anymore
        for img_id in range(len(datapoint.images)):
            all_objects_ids = set(
                i
                for find in datapoint.find_queries
                for i in find.object_ids_output
                if find.image_id == img_id
            )
            unused_ids = (
                set(range(len(datapoint.images[img_id].objects))) - all_objects_ids
            )
            for tgt_img_id, tgt_obj_id in self.query_filter.obj_ids_to_filter:
                if tgt_img_id == img_id:
                    unused_ids.add(tgt_obj_id)

            if len(unused_ids) > 0:
                old_objects = datapoint.images[img_id].objects
                object_old_to_new_map = {}
                new_objects = []
                for i, o in enumerate(old_objects):
                    if i not in unused_ids:
                        object_old_to_new_map[i] = len(new_objects)
                        new_objects.append(o)

                datapoint.images[img_id].objects = new_objects

                # Remap the outputs of the find queries
                affected_find_queries_ids = set()
                object_old_to_new_map_per_query = {}
                for fid, find in enumerate(datapoint.find_queries):
                    if find.image_id == img_id:
                        old_object_ids_output = find.object_ids_output
                        object_old_to_new_map_per_query[fid] = {}
                        find.object_ids_output = []
                        for oid, old_obj_id in enumerate(old_object_ids_output):
                            if old_obj_id not in unused_ids:
                                new_obj_id = object_old_to_new_map[old_obj_id]
                                find.object_ids_output.append(new_obj_id)
                                object_old_to_new_map_per_query[fid][oid] = (
                                    len(find.object_ids_output) - 1
                                )
                        affected_find_queries_ids.add(fid)

                # Remap the pointers in find queries
                final_find = []
                for find in datapoint.find_queries:
                    try:
                        if (
                            find.ptr_x is not None
                            and find.ptr_x.query_id in affected_find_queries_ids
                        ):
                            find.ptr_x.object_id = object_old_to_new_map_per_query[
                                find.ptr_x.query_id
                            ][find.ptr_x.object_id]

                        if (
                            find.ptr_y is not None
                            and find.ptr_y.query_id in affected_find_queries_ids
                        ):
                            find.ptr_y.object_id = object_old_to_new_map_per_query[
                                find.ptr_y.query_id
                            ][find.ptr_y.object_id]
                        final_find.append(find)
                    except KeyError:
                        # This means the pointed object doesn't exist anymore
                        # We just skip that query
                        pass
                datapoint.find_queries = final_find

                # Remap the pointers of the get queries
                final_get = []
                for get in datapoint.get_queries:
                    try:
                        if (
                            get.ptr_x is not None
                            and get.ptr_x.query_id in affected_find_queries_ids
                        ):
                            get.ptr_x.object_id = object_old_to_new_map_per_query[
                                get.ptr_x.query_id
                            ][get.ptr_x.object_id]

                        if (
                            get.ptr_y is not None
                            and get.ptr_y.query_id in affected_find_queries_ids
                        ):
                            get.ptr_y.object_id = object_old_to_new_map_per_query[
                                get.ptr_y.query_id
                            ][get.ptr_y.object_id]
                        final_get.append(get)
                    except KeyError:
                        # This means the pointed object doesn't exist anymore
                        # We just skip that query
                        pass
                datapoint.get_queries = final_get

        # finally remove unused images
        all_imgs_to_keep = set()
        for f_q in datapoint.find_queries:
            all_imgs_to_keep.add(f_q.image_id)

        old_img_id_to_new_img_id = {}
        new_images = []
        for img_id, img in enumerate(datapoint.images):
            if img_id in all_imgs_to_keep:
                old_img_id_to_new_img_id[img_id] = len(new_images)
                new_images.append(img)
        datapoint.images = new_images

        for f_q in datapoint.find_queries:
            f_q.image_id = old_img_id_to_new_img_id[f_q.image_id]

        return datapoint


class AddPrefixSuffixToFindText:
    """
    Add prefix or suffix strings to find query text on the fly.

    If `condition_on_text` is True, the prefix or suffix strings are only added
    to those find query text in `condition_text_list` (case-insensitive).
    """

    def __init__(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        condition_on_text: bool = False,
        condition_text_list: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.condition_on_text = condition_on_text
        if self.condition_on_text:
            assert condition_text_list is not None
            self.condition_text_set = {s.lower().strip() for s in condition_text_list}
        self.enabled = enabled
        if self.enabled:
            logging.info(
                f"AddPrefixSuffixToFindText: prefix={prefix}, suffix={suffix}, "
                f"condition_on_text={condition_on_text}, condition_text_list={condition_text_list}"
            )

    def __call__(self, datapoint, **kwargs):
        if not self.enabled:
            return datapoint

        for find in datapoint.find_queries:
            if find.query_text == "geometric":
                # skip geometric find queries
                continue
            if (
                self.condition_on_text
                and find.query_text.lower().strip() not in self.condition_text_set
            ):
                # if condition_on_text is True, skip those queries not in condition_text_set
                continue

            # add prefix and/or suffix strings to the find query text
            if self.prefix is not None:
                find.query_text = self.prefix + find.query_text
            if self.suffix is not None:
                find.query_text = find.query_text + self.suffix

        return datapoint


class FilterCrowds(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        """
        Compute set of query ids to keep, for both find and get queries
        """
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = set()
        self.get_ids_to_filter = set()
        for img_id, img in enumerate(datapoint.images):
            for obj_id, obj in enumerate(img.objects):
                if obj.is_crowd:
                    self.obj_ids_to_filter.add((img_id, obj_id))


class BinarizeGetQuery:
    """
    For Get queries that query for a category, binarize them, i.e., transform them into a yes/no question.

    If `keep_original_query`, the binarized query is created on top of the original one, otherwise it replaces it.
    If `ratio_negatives` > 0, negatives (i.e., queries where the ground truth answer is "no") are generated for
    every positive binary query. The negatives are selected from a list provided either directly or through a .txt file.
    """

    def __init__(
        self,
        keep_original_query: bool = False,
        ratio_negatives: int = 0,
        list_negatives: Optional[Union[str, MutableSequence]] = None,
    ) -> None:
        self.keep_original_query = keep_original_query
        self.ratio_negatives = ratio_negatives

        if self.ratio_negatives > 0:
            assert list_negatives is not None
        if isinstance(list_negatives, str):
            assert os.path.isfile(list_negatives)
            with open(list_negatives, "r") as f:
                list_negatives = f.read().splitlines()
        self.list_negatives = list_negatives

    def __call__(self, datapoint, **kwargs):
        new_get_queries = []
        for get_query in datapoint.get_queries:
            if get_query.query_text == "category":
                text_output = get_query.text_output

                # Formulate as binary yes/no get
                query_text_pos = f"is {text_output}"
                text_output_pos = "yes"

                if self.keep_original_query:
                    new_query = GetQuery(
                        query_type=get_query.query_type,
                        query_text=query_text_pos,
                        ptr_x=get_query.ptr_x,
                        ptr_y=get_query.ptr_y,
                        text_output=text_output_pos,
                        query_content=QueryContent.ContentCategory,
                    )
                    new_get_queries.append(new_query)

                else:
                    get_query.query_text = query_text_pos
                    get_query.text_output = text_output_pos
                    get_query.query_content = QueryContent.ContentCategory

                if self.ratio_negatives > 0:
                    text_output_neg = "no"
                    text_ignore = {text_output}

                    for _neg_id in range(self.ratio_negatives):
                        candidate_negatives = tuple(
                            set(self.list_negatives) - text_ignore
                        )
                        if len(candidate_negatives) <= 0:
                            break
                        query_text_neg = random.choice(candidate_negatives)
                        query_text_neg = f"is {query_text_neg}"
                        new_negative_query = GetQuery(
                            query_type=get_query.query_type,
                            query_text=query_text_neg,
                            ptr_x=get_query.ptr_x,
                            ptr_y=get_query.ptr_y,
                            text_output=text_output_neg,
                            query_content=QueryContent.ContentCategory,
                        )
                        new_get_queries.append(new_negative_query)
                        text_ignore.add(query_text_neg)

        for n_g_q in new_get_queries:
            datapoint.get_queries.append(n_g_q)

        return datapoint


class TextQueryToVisual:
    """
    Transform a test query to a visual query (with some proba), using any of the output targets as the prompt
    """

    def __init__(self, probability) -> None:
        self.probability = probability
        assert 0 <= probability <= 1

    def __call__(self, datapoint: Datapoint, **kwargs):
        for find in datapoint.find_queries:
            if find.input_bbox is not None or find.input_points is not None:
                # skip geometric find queries
                continue

            if len(find.object_ids_output) == 0:
                # Can't create a visual query, skip
                continue

            if find.query_processing_order > 0:
                # Second stage query, can't use
                continue

            if random.random() > self.probability:
                continue

            selected_vq_id = random.choice(find.object_ids_output)
            img_id = find.image_id

            find.input_bbox = datapoint.images[img_id].objects[selected_vq_id].bbox
            find.input_bbox_label = torch.ones(1, dtype=torch.bool)
            find.query_text = "visual"

        return datapoint


class RemoveInputBoxes:
    """
    Remove input boxes from find queries
    """

    def __init__(self) -> None:
        pass

    def __call__(self, datapoint: Datapoint, **kwargs):
        for find in datapoint.find_queries:
            if find.input_bbox is None:
                continue

            if find.query_text == "geometric":
                print("Warning: removing input box from geometric find query")

            find.input_bbox = None
        return datapoint


class DropWikiNegs(FilterDataPointQueries):
    """
    Transform to drop some of the wiki negs
    """

    def __init__(self, probability, all_at_once=True) -> None:
        self.probability = probability
        self.all_at_once = all_at_once
        assert 0 <= probability <= 1

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        """
        Compute set of query ids to keep, for both find and get queries
        """
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = set()
        self.get_ids_to_filter = set()

        drop_all = False
        if self.all_at_once and random.random() < self.probability:
            drop_all = True
        for i, find in enumerate(datapoint.find_queries):
            if len(find.object_ids_output) != 0:
                # Not a negative
                continue

            if find.query_processing_order > 0:
                # Second stage query, can't use
                continue

            if not isinstance(find.source, str):
                continue

            assert find.source is not None
            if "wiki" in find.source.lower():
                if drop_all or random.random() < self.probability:
                    self.find_ids_to_filter.add(i)


class OverwriteTextQuery:
    """
    With some probability, overwrite the text query with a custom text
    """

    def __init__(self, target_text, probability=1.0) -> None:
        self.probability = probability
        self.target_text = target_text
        assert 0 <= probability <= 1

    def __call__(self, datapoint: Datapoint, **kwargs):
        for find in datapoint.find_queries:
            if random.random() > self.probability:
                continue

            find.query_text = self.target_text

        return datapoint


class FilterPosNegContext(FilterDataPointQueries):
    """
    Select the right number of pos/neg from the context
    """

    def __init__(self, num_pos, num_neg):
        self.num_pos = num_pos
        self.num_neg = num_neg

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        self.get_ids_to_filter = set()
        self.find_ids_to_filter = set()

        last_stage = max(q.query_processing_order for q in datapoint.find_queries)
        assert (
            last_stage >= self.num_neg + self.num_pos
        ), f"Found  {last_stage} stages. Expected {self.num_neg} + {self.num_pos} stages"

        found_pos, found_neg = 0, 0
        for i, f_q in enumerate(datapoint.find_queries):
            qpo = f_q.query_processing_order
            if (
                qpo != last_stage
                and f_q.input_bbox_label is not None
                and len(f_q.input_bbox_label) > 0
                and f_q.input_bbox_label[0] == 1
            ):
                # Positive query
                if found_pos < self.num_pos:
                    found_pos += 1
                else:
                    self.find_ids_to_filter.add(i)
            elif (
                qpo != last_stage
                and f_q.input_bbox_label is not None
                and len(f_q.input_bbox_label) > 0
                and f_q.input_bbox_label[0] == 0
            ):
                # Negative query
                if found_neg < self.num_neg:
                    found_neg += 1
                else:
                    self.find_ids_to_filter.add(i)

        if not found_pos == self.num_pos or not found_neg == self.num_neg:
            raise ValueError(
                f"Found {found_pos} pos and {found_neg} neg, expected {self.num_pos} pos and {self.num_neg} neg"
            )


class DeleteContextQueries(FilterDataPointQueries):
    """
    Select the right number of pos/neg from the context
    """

    def __init__(self):
        pass

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        self.get_ids_to_filter = set()
        self.find_ids_to_filter = set()

        last_stage = max(q.query_processing_order for q in datapoint.find_queries)
        i = 0
        for i, f_q in enumerate(datapoint.find_queries):
            qpo = f_q.query_processing_order
            if qpo != last_stage:
                self.find_ids_to_filter.add(i)
            else:
                f_q.query_processing_order = 0

        assert len(datapoint.find_queries) - len(self.find_ids_to_filter) == 1


class DropGeoPromptsInLastStage:
    """
    Transform to drop some of the wiki negs
    """

    def __init__(self) -> None:
        pass

    def __call__(self, datapoint, **kwargs):
        last_stage = max(q.query_processing_order for q in datapoint.find_queries)
        for fq in datapoint.find_queries:
            if fq.query_processing_order == last_stage:
                # Drop the input box
                fq.input_bbox = None
                fq.input_bbox_label = None
                fq.input_points = None
        return datapoint
