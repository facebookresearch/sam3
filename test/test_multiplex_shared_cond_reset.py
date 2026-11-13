# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Regression tests for the multiplex shared-conditioning-frame removal bug.

Scenario (model-free, data-free; object/frame ids are arbitrary synthetic values):
several objects in a multiplex bucket share a single conditioning ("cond") frame.
One object (the "owner") supplied the user input that created that cond frame; the
others ("freeloaders") were added on already-tracked frames, never created their own
cond, and rely on the bucket having a cond frame plus their own non-conditioning
("non_cond") memory.

Bug: removing the owner empties the bucket's only cond frame inside
``clear_all_points_in_frame`` (the owner's input frame gets downgraded), which used
to trigger a full ``_reset_tracking_results`` -- wiping *every* object's memory,
including the freeloaders'. Propagation then raises
"No points are provided; please add points first".

These tests construct a minimal ``inference_state`` and call the real, unmodified
state-machine method ``clear_all_points_in_frame`` (exactly what ``remove_objects``
invokes for a removed object's input frames). They assert the *fixed* behaviour, so
they FAIL on buggy code and PASS once the fix promotes the first frame that contains
a surviving (non-owner) participant to be the new cond frame instead of resetting.
"""

import unittest
from types import SimpleNamespace

from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo


# --- synthetic, arbitrary ids (no relation to any real data) ---
OWNER = 0
FREELOADERS = [1, 2]
OBJ_IDS = [OWNER] + FREELOADERS

COND_FRAME = 0  # owner's input frame -> the bucket's only cond frame
FIRST_PARTICIPANT_FRAME = 2  # earliest frame where a non-owner appears
LATEST_FRAME = 4

# which objects are present in each tracked frame's output (local_obj_id_to_idx)
FRAME_MEMBERS = {
    0: [0],  # owner only (this is the cond frame)
    1: [0],  # owner only
    2: [0, 1],  # freeloader 1 first appears here
    3: [0, 1, 2],  # freeloader 2 first appears here
    4: [0, 1, 2],
}


def _out(members):
    """A placeholder frame output. The real code only does dict ops / reads
    ``local_obj_id_to_idx`` on this in the path under test -- no tensors needed."""
    return {
        "_marker": tuple(members),
        "local_obj_id_to_idx": {oid: i for i, oid in enumerate(members)},
    }


def _bare_predictor():
    """An instance without ``__init__`` (no model load). Every method exercised here
    operates only on ``inference_state``."""
    pred = object.__new__(Sam3VideoTrackingMultiplexDemo)
    pred.is_dynamic_model = True
    return pred


def _build_shared_cond_state(obj_ids, frame_members, cond_frame, owner):
    idx = {oid: i for i, oid in enumerate(obj_ids)}
    out_per_obj = {}
    for oid, i in idx.items():
        cond = {cond_frame: _out([owner])} if oid == owner else {}
        non_cond = {
            f: _out(members)
            for f, members in frame_members.items()
            if f != cond_frame and oid in members
        }
        out_per_obj[i] = {
            "cond_frame_outputs": cond,
            "non_cond_frame_outputs": non_cond,
        }
    return {
        "multiplex_state": SimpleNamespace(total_valid_entries=len(obj_ids)),
        "obj_id_to_idx": dict(idx),
        "obj_idx_to_id": {i: oid for oid, i in idx.items()},
        "obj_ids": list(obj_ids),
        "tracking_has_started": True,
        "first_ann_frame_idx": cond_frame,
        # only the owner has a real input, on the cond frame
        "point_inputs_per_obj": {
            idx[oid]: ({cond_frame: {"pt": True}} if oid == owner else {})
            for oid in obj_ids
        },
        "mask_inputs_per_obj": {i: {} for i in idx.values()},
        "temp_output_dict_per_obj": {
            i: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            for i in idx.values()
        },
        "output_dict_per_obj": out_per_obj,
        "output_dict": {
            "cond_frame_outputs": {cond_frame: _out([owner])},
            "non_cond_frame_outputs": {
                f: _out(members)
                for f, members in frame_members.items()
                if f != cond_frame
            },
        },
        # invariant in real code: consolidated_frame_inds == frames with user input
        "consolidated_frame_inds": {
            "cond_frame_outputs": {cond_frame},
            "non_cond_frame_outputs": set(),
        },
        "frames_already_tracked": {f: {"reverse": False} for f in frame_members},
    }


class TestMultiplexSharedCondReset(unittest.TestCase):
    def test_removing_cond_owner_preserves_survivors(self):
        """Removing the cond-owner must NOT wipe the freeloaders' memory, and must
        leave the bucket with a cond frame so propagation does not raise No-points."""
        pred = _bare_predictor()
        state = _build_shared_cond_state(OBJ_IDS, FRAME_MEMBERS, COND_FRAME, OWNER)
        idx = state["obj_id_to_idx"]

        # This is exactly what remove_objects() calls for the owner's input frame.
        pred.clear_all_points_in_frame(state, COND_FRAME, OWNER, need_output=False)

        cond = state["output_dict"]["cond_frame_outputs"]
        # 1) bucket still has a cond frame -> no "No points" crash later
        self.assertGreater(len(cond), 0, "bucket lost its only cond frame")
        # 2) tracking was not destructively reset
        self.assertTrue(state["tracking_has_started"], "tracking_has_started was reset")
        # 3) every surviving freeloader still has memory
        for fl in FREELOADERS:
            d = state["output_dict_per_obj"][idx[fl]]
            self.assertTrue(
                d["cond_frame_outputs"] or d["non_cond_frame_outputs"],
                f"freeloader {fl} lost all memory",
            )

    def test_promoted_frame_is_first_participant_frame(self):
        """The promoted cond frame should be the earliest frame containing a
        surviving participant (keeps the normal 'early anchor + forward chain'
        structure), not the latest frame."""
        pred = _bare_predictor()
        state = _build_shared_cond_state(OBJ_IDS, FRAME_MEMBERS, COND_FRAME, OWNER)

        pred.clear_all_points_in_frame(state, COND_FRAME, OWNER, need_output=False)

        cond = state["output_dict"]["cond_frame_outputs"]
        self.assertIn(
            FIRST_PARTICIPANT_FRAME, cond, "first-participant frame was not promoted"
        )
        self.assertNotIn(LATEST_FRAME, cond, "latest frame was promoted instead")

    def test_non_cond_chain_is_preserved(self):
        """Promotion is a move, not a wipe: surviving objects' non_cond history stays."""
        pred = _bare_predictor()
        state = _build_shared_cond_state(OBJ_IDS, FRAME_MEMBERS, COND_FRAME, OWNER)
        idx = state["obj_id_to_idx"]
        before = sum(
            len(state["output_dict_per_obj"][idx[fl]]["non_cond_frame_outputs"])
            for fl in FREELOADERS
        )

        pred.clear_all_points_in_frame(state, COND_FRAME, OWNER, need_output=False)

        after = sum(
            len(state["output_dict_per_obj"][idx[fl]]["cond_frame_outputs"])
            + len(state["output_dict_per_obj"][idx[fl]]["non_cond_frame_outputs"])
            for fl in FREELOADERS
        )
        self.assertEqual(after, before, "freeloaders' memory frame count changed")

    def test_removing_only_object_still_resets(self):
        """Boundary: if removing the owner leaves NO surviving object, the full reset
        is still correct (there is genuinely nothing left to track). Guards against the
        fix over-triggering. Passes both before and after the fix."""
        pred = _bare_predictor()
        single = _build_shared_cond_state([OWNER], {0: [0], 1: [0]}, COND_FRAME, OWNER)

        pred.clear_all_points_in_frame(single, COND_FRAME, OWNER, need_output=False)

        self.assertEqual(len(single["output_dict"]["cond_frame_outputs"]), 0)
        self.assertFalse(single["tracking_has_started"])


if __name__ == "__main__":
    unittest.main()
