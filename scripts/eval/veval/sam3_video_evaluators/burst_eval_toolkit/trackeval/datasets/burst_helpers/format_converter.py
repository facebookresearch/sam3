# flake8: noqa

import json
import os
from typing import Union

import pycocotools.mask as cocomask
from tabulate import tabulate


def _global_track_id(
    *, local_track_id: Union[str, int], video_id: Union[str, int], track_id_mapping
) -> int:
    # remap local track ids into globally unique ids
    return track_id_mapping[str(video_id)][str(local_track_id)]


class GroundTruthBURSTFormatToTAOFormatConverter:
    def __init__(self, ali_format):
        self._ali_format = ali_format
        self._split = ali_format["split"]
        self._categories = self._make_categories()
        self._videos = []
        self._annotations = []
        self._tracks = {}
        self._images = []
        self._next_img_id = 0
        self._next_ann_id = 0

        self._track_id_mapping = self._load_track_id_mapping()

        for seq in ali_format["sequences"]:
            self._visit_seq(seq)

    def _load_track_id_mapping(self):
        id_map = {}
        next_global_track_id = 1
        for seq in self._ali_format["sequences"]:
            seq_id = seq["id"]
            seq_id_map = {}
            id_map[str(seq_id)] = seq_id_map
            for local_track_id in seq["track_category_ids"]:
                seq_id_map[str(local_track_id)] = next_global_track_id
                next_global_track_id += 1
        return id_map

    def global_track_id(
        self, *, local_track_id: Union[str, int], video_id: Union[str, int]
    ) -> int:
        return _global_track_id(
            local_track_id=local_track_id,
            video_id=video_id,
            track_id_mapping=self._track_id_mapping,
        )

    def _visit_seq(self, seq):
        self._make_video(seq)
        imgs = self._make_images(seq)
        self._make_annotations_and_tracks(seq, imgs)

    def _make_images(self, seq):
        imgs = []
        for img_path in seq["annotated_image_paths"]:
            video = self._split + "/" + seq["dataset"] + "/" + seq["seq_name"]
            file_name = video + "/" + img_path

            # TODO: once python 3.9 is more common, we can use this nicer and safer code
            # stripped = img_path.removesuffix('.jpg').removesuffix('.png').removeprefix('frame')
            stripped = (
                img_path.replace(".jpg", "").replace(".png", "").replace("frame", "")
            )

            last = stripped.split("_")[-1]
            frame_idx = int(last)

            img = {
                "id": self._next_img_id,
                "video": video,
                "width": seq["width"],
                "height": seq["height"],
                "file_name": file_name,
                "frame_index": frame_idx,
                "video_id": seq["id"],
            }
            self._next_img_id += 1
            self._images.append(img)
            imgs.append(img)
        return imgs

    def _make_video(self, seq):
        video_id = seq["id"]
        dataset = seq["dataset"]
        seq_name = seq["seq_name"]
        name = f"{self._split}/" + dataset + "/" + seq_name
        video = {
            "id": video_id,
            "width": seq["width"],
            "height": seq["height"],
            "neg_category_ids": seq["neg_category_ids"],
            "not_exhaustive_category_ids": seq["not_exhaustive_category_ids"],
            "name": name,
            "metadata": {"dataset": dataset},
        }
        self._videos.append(video)

    def _make_annotations_and_tracks(self, seq, imgs):
        video_id = seq["id"]
        segs = seq["segmentations"]
        assert len(segs) == len(imgs), (len(segs), len(imgs))
        for frame_segs, img in zip(segs, imgs):
            for local_track_id, seg in frame_segs.items():
                distractors = {
                    20,
                    63,
                    108,
                    180,
                    188,
                    204,
                    212,
                    247,
                    303,
                    403,
                    407,
                    415,
                    490,
                    504,
                    507,
                    513,
                    529,
                    567,
                    569,
                    588,
                    672,
                    691,
                    702,
                    708,
                    711,
                    720,
                    736,
                    737,
                    798,
                    813,
                    815,
                    827,
                    831,
                    851,
                    877,
                    883,
                    912,
                    971,
                    976,
                    1130,
                    1133,
                    1134,
                    1169,
                    1184,
                    1220,
                }
                global_track_id = self.global_track_id(
                    local_track_id=local_track_id, video_id=seq["id"]
                )
                rle = seg["rle"]
                segmentation = {"counts": rle, "size": [img["height"], img["width"]]}
                image_id = img["id"]
                category_id = int(seq["track_category_ids"][local_track_id])
                if category_id in distractors:
                    continue
                coco_bbox = cocomask.toBbox(segmentation)
                bbox = [int(x) for x in coco_bbox]
                ann = {
                    "segmentation": segmentation,
                    "id": self._next_ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "track_id": global_track_id,
                    "video_id": video_id,
                    "bbox": bbox,
                }
                self._next_ann_id += 1
                self._annotations.append(ann)

                if global_track_id not in self._tracks:
                    track = {
                        "id": global_track_id,
                        "category_id": category_id,
                        "video_id": video_id,
                    }
                    self._tracks[global_track_id] = track

    def convert(self):
        tracks = sorted(self._tracks.values(), key=lambda t: t["id"])
        return {
            "videos": self._videos,
            "annotations": self._annotations,
            "tracks": tracks,
            "images": self._images,
            "categories": self._categories,
            "track_id_mapping": self._track_id_mapping,
            "split": self._split,
        }

    def _make_categories(self):
        tao_categories_path = os.path.join(
            os.path.dirname(__file__),
            "tao_categories.json",
        )
        # tao_categories_path = os.path.join(os.path.dirname(__file__), 'tao_categories.json')
        with open(os.path.realpath(tao_categories_path)) as f:
            return json.load(f)


class PredictionBURSTFormatToTAOFormatConverter:
    def __init__(self, gt, ali_format, exemplar_guided):
        self._gt = gt
        self._ali_format = ali_format
        if "split" in ali_format:
            self._split = ali_format["split"]
            gt_split = self._gt["split"]
            assert self._split == gt_split, (self._split, gt_split)
        else:
            self._split = self._gt["split"]
        self._exemplar_guided = exemplar_guided
        self._result = []
        self._next_det_id = 0

        self._img_by_filename = {}
        for img in self._gt["images"]:
            file_name = img["file_name"]
            assert file_name not in self._img_by_filename
            self._img_by_filename[file_name] = img

        self._gt_track_by_track_id = {}
        for track in self._gt["tracks"]:
            self._gt_track_by_track_id[int(track["id"])] = track

        self._filtered_out_track_ids = set()

        for seq in ali_format["sequences"]:
            self._visit_seq(seq)

        if exemplar_guided and len(self._filtered_out_track_ids) > 0:
            self.print_filter_out_debug_info(ali_format)

    def print_filter_out_debug_info(self, ali_format):
        track_ids_in_pred = set()
        a_dict_for_debugging = {}
        for seq in ali_format["sequences"]:
            for local_track_id in seq["track_category_ids"]:
                global_track_id = _global_track_id(
                    local_track_id=local_track_id,
                    video_id=seq["id"],
                    track_id_mapping=self._gt["track_id_mapping"],
                )
                track_ids_in_pred.add(global_track_id)
                a_dict_for_debugging[global_track_id] = {
                    "seq": seq,
                    "local_track_id": local_track_id,
                }
        print("Number of Track ids in pred:", len(track_ids_in_pred))
        print(
            "Exemplar Guided: Filtered out",
            len(self._filtered_out_track_ids),
            "tracks which were not found in the ground truth.",
        )
        track_ids_after_filtering = set(d["track_id"] for d in self._result)
        print("Number of tracks after filtering:", len(track_ids_after_filtering))
        problem_tracks = list(
            track_ids_in_pred - track_ids_after_filtering - self._filtered_out_track_ids
        )
        if len(problem_tracks) > 0:
            print(
                "\nWARNING:",
                len(problem_tracks),
                "object tracks are not present. There could be a number of reasons for this:\n"
                "(1) If you are running evaluation for the box/point exemplar-guided task then this is to be expected"
                " because your tracker probably didn't predict masks for every ground-truth object instance.\n"
                "(2) If you are running evaluation for the mask exemplar-guided task, then this could indicate a "
                "problem. Assume that you copied the given first-frame object mask to your predicted result, this "
                "should not happen. It could be that your predictions are at the wrong frame-rate i.e. you have no "
                "predicted masks for video frames which will be evaluated.\n",
            )

            rows = []
            for xx in problem_tracks:
                rows.append(
                    [
                        a_dict_for_debugging[xx]["seq"]["dataset"],
                        a_dict_for_debugging[xx]["seq"]["seq_name"],
                        a_dict_for_debugging[xx]["local_track_id"],
                    ]
                )

            print(
                "For your reference, the sequence name and track IDs for these missing tracks are:"
            )
            print(tabulate(rows, ["Dataset", "Sequence Name", "Track ID"]))

    def _visit_seq(self, seq):
        dataset = seq["dataset"]
        seq_name = seq["seq_name"]
        assert len(seq["segmentations"]) == len(seq["annotated_image_paths"])
        for frame_segs, img_path in zip(
            seq["segmentations"], seq["annotated_image_paths"]
        ):
            for local_track_id_str, track_det in frame_segs.items():
                rle = track_det["rle"]

                file_name = (
                    self._split + "/" + dataset + "/" + seq_name + "/" + img_path
                )
                # the result might have a higher frame rate than the ground truth
                if file_name not in self._img_by_filename:
                    continue

                img = self._img_by_filename[file_name]
                img_id = img["id"]
                height = img["height"]
                width = img["width"]
                segmentation = {"counts": rle, "size": [height, width]}

                local_track_id = int(local_track_id_str)
                if self._exemplar_guided:
                    global_track_id = _global_track_id(
                        local_track_id=local_track_id,
                        video_id=seq["id"],
                        track_id_mapping=self._gt["track_id_mapping"],
                    )
                else:
                    global_track_id = local_track_id
                coco_bbox = cocomask.toBbox(segmentation)
                bbox = [int(x) for x in coco_bbox]
                det = {
                    "id": self._next_det_id,
                    "image_id": img_id,
                    "track_id": global_track_id,
                    "bbox": bbox,
                    "segmentation": segmentation,
                }
                if self._exemplar_guided:
                    if global_track_id not in self._gt_track_by_track_id:
                        self._filtered_out_track_ids.add(global_track_id)
                        continue
                    gt_track = self._gt_track_by_track_id[global_track_id]
                    category_id = gt_track["category_id"]
                    det["category_id"] = category_id
                elif "category_id" in track_det:
                    det["category_id"] = track_det["category_id"]
                else:
                    category_id = seq["track_category_ids"][local_track_id_str]
                    det["category_id"] = category_id
                self._next_det_id += 1
                if "score" in track_det:
                    det["score"] = track_det["score"]
                else:
                    det["score"] = 1.0
                self._result.append(det)

    def convert(self):
        return self._result
