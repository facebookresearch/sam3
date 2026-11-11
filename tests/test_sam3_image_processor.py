# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Tests for detection fusion functionality in Sam3Processor.

Note: These tests must be run in an environment with the sam3 package dependencies
installed (torch, torchvision, etc.). Run with:
    conda activate sam3
    pytest tests/test_sam3_image_processor.py
"""

import pytest

# Check for required dependencies
try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

try:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.perflib.masks_ops import mask_iou
except ImportError as e:
    pytest.skip(f"sam3 package not available: {e}", allow_module_level=True)


class TestSam3ProcessorFusion:
    """Tests for detection fusion functionality in Sam3Processor"""

    def test_fusion_disabled_by_default(self):
        """Test that fusion is disabled when fuse_detections_iou_threshold is None"""
        # Create a mock model (we'll test the fusion logic directly)
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        assert processor.fuse_detections_iou_threshold is None

    def test_fusion_enabled_when_threshold_set(self):
        """Test that fusion is enabled when fuse_detections_iou_threshold is set"""
        class MockModel:
            pass

        processor = Sam3Processor(
            MockModel(), device="cpu", fuse_detections_iou_threshold=0.3
        )
        assert processor.fuse_detections_iou_threshold == 0.3

    def test_fuse_detections_empty_input(self):
        """Test fusion with empty detections"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        scores = torch.tensor([], dtype=torch.float32)
        masks = torch.zeros((0, 1, 10, 10))
        boxes = torch.zeros((0, 4))

        fused_scores, fused_masks, fused_boxes = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=0.3
        )

        assert len(fused_scores) == 0
        assert fused_masks.shape[0] == 0
        assert len(fused_boxes) == 0

    def test_fuse_detections_single_detection(self):
        """Test fusion with a single detection (should remain unchanged)"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        scores = torch.tensor([0.8], dtype=torch.float32)
        masks = torch.zeros((1, 1, 10, 10))
        masks[0, 0, 2:5, 2:5] = 1.0  # Small square mask
        boxes = torch.tensor([[2.0, 2.0, 5.0, 5.0]], dtype=torch.float32)

        fused_scores, fused_masks, fused_boxes = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=0.3
        )

        assert len(fused_scores) == 1
        assert fused_scores[0] == scores[0]
        assert fused_masks.shape == (1, 1, 10, 10)
        torch.testing.assert_close(fused_masks, masks)
        assert len(fused_boxes) == 1

    def test_fuse_detections_non_overlapping(self):
        """Test that non-overlapping detections are not fused"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        scores = torch.tensor([0.8, 0.7], dtype=torch.float32)
        masks = torch.zeros((2, 1, 20, 20))
        # Two non-overlapping masks
        masks[0, 0, 2:5, 2:5] = 1.0  # Top-left
        masks[1, 0, 15:18, 15:18] = 1.0  # Bottom-right
        boxes = torch.tensor(
            [[2.0, 2.0, 5.0, 5.0], [15.0, 15.0, 18.0, 18.0]], dtype=torch.float32
        )

        fused_scores, fused_masks, fused_boxes = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=0.3
        )

        # Should still have 2 detections (not fused)
        assert len(fused_scores) == 2
        assert fused_masks.shape[0] == 2
        assert len(fused_boxes) == 2

    def test_fuse_detections_overlapping(self):
        """Test that overlapping detections are fused"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        scores = torch.tensor([0.8, 0.7], dtype=torch.float32)
        masks = torch.zeros((2, 1, 20, 20))
        # Two overlapping masks with high overlap (same center, different sizes)
        masks[0, 0, 5:10, 5:10] = 1.0  # 5x5 square
        masks[1, 0, 6:11, 6:11] = 1.0  # 5x5 square, overlaps by 4x4 = 16 pixels
        boxes = torch.tensor(
            [[5.0, 5.0, 10.0, 10.0], [6.0, 6.0, 11.0, 11.0]], dtype=torch.float32
        )

        fused_scores, fused_masks, fused_boxes = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=0.3
        )

        # Should be fused into 1 detection
        assert len(fused_scores) == 1
        assert fused_masks.shape[0] == 1
        assert len(fused_boxes) == 1
        # Score should be max of the two
        assert fused_scores[0] == 0.8

    def test_fuse_detections_mask_union(self):
        """Test that fused masks are the union of overlapping masks"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        scores = torch.tensor([0.8, 0.7], dtype=torch.float32)
        masks = torch.zeros((2, 1, 20, 20))
        # Two overlapping masks with sufficient overlap
        masks[0, 0, 5:10, 5:10] = 1.0  # Left square
        masks[1, 0, 6:11, 6:11] = 1.0  # Right square, overlaps significantly
        boxes = torch.tensor(
            [[5.0, 5.0, 10.0, 10.0], [6.0, 6.0, 11.0, 11.0]], dtype=torch.float32
        )

        fused_scores, fused_masks, fused_boxes = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=0.3
        )

        # Should be fused
        assert len(fused_scores) == 1
        
        # Check that fused mask contains union of both masks
        fused_mask_binary = fused_masks[0, 0] > 0.5
        original_mask1_binary = masks[0, 0] > 0.5
        original_mask2_binary = masks[1, 0] > 0.5
        union_binary = original_mask1_binary | original_mask2_binary

        # Fused mask should be at least as large as union
        assert (fused_mask_binary >= union_binary).all()

    def test_fuse_detections_multiple_groups(self):
        """Test fusion with multiple groups of overlapping detections"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
        masks = torch.zeros((4, 1, 30, 30))
        # Group 1: masks 0 and 1 overlap significantly
        masks[0, 0, 5:10, 5:10] = 1.0
        masks[1, 0, 6:11, 6:11] = 1.0  # High overlap
        # Group 2: masks 2 and 3 overlap significantly (separate location)
        masks[2, 0, 20:25, 20:25] = 1.0
        masks[3, 0, 21:26, 21:26] = 1.0  # High overlap
        boxes = torch.tensor(
            [
                [5.0, 5.0, 10.0, 10.0],
                [6.0, 6.0, 11.0, 11.0],
                [20.0, 20.0, 25.0, 25.0],
                [21.0, 21.0, 26.0, 26.0],
            ],
            dtype=torch.float32,
        )

        fused_scores, fused_masks, fused_boxes = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=0.3
        )

        # Should have 2 fused detections (one per group)
        assert len(fused_scores) == 2
        assert fused_masks.shape[0] == 2
        assert len(fused_boxes) == 2
        # Scores should be max of each group
        assert fused_scores[0] == 0.9  # max(0.9, 0.8)
        assert fused_scores[1] == 0.7  # max(0.7, 0.6)

    def test_fuse_detections_iou_threshold(self):
        """Test that IoU threshold correctly controls fusion"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        scores = torch.tensor([0.8, 0.7], dtype=torch.float32)
        masks = torch.zeros((2, 1, 20, 20))
        # Two masks with low overlap
        masks[0, 0, 5:10, 5:10] = 1.0
        masks[1, 0, 9:14, 9:14] = 1.0  # Small overlap

        # Compute actual IoU
        mask1_binary = masks[0, 0] > 0.5
        mask2_binary = masks[1, 0] > 0.5
        ious = mask_iou(mask1_binary.unsqueeze(0), mask2_binary.unsqueeze(0))
        actual_iou = ious[0, 0].item()

        boxes = torch.tensor(
            [[5.0, 5.0, 10.0, 10.0], [9.0, 9.0, 14.0, 14.0]], dtype=torch.float32
        )

        # With threshold below actual IoU - should fuse
        fused_scores_low, _, _ = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=actual_iou - 0.1
        )
        assert len(fused_scores_low) == 1

        # With threshold above actual IoU - should not fuse
        fused_scores_high, _, _ = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=actual_iou + 0.1
        )
        assert len(fused_scores_high) == 2

    def test_fuse_detections_score_ordering(self):
        """Test that fusion preserves the highest score from each group"""
        class MockModel:
            pass

        processor = Sam3Processor(MockModel(), device="cpu")
        # Lower score first to test that max is used
        scores = torch.tensor([0.6, 0.9], dtype=torch.float32)
        masks = torch.zeros((2, 1, 20, 20))
        masks[0, 0, 5:10, 5:10] = 1.0
        masks[1, 0, 6:11, 6:11] = 1.0  # High overlap
        boxes = torch.tensor(
            [[5.0, 5.0, 10.0, 10.0], [6.0, 6.0, 11.0, 11.0]], dtype=torch.float32
        )

        fused_scores, _, _ = processor._fuse_detections(
            scores, masks, boxes, iou_threshold=0.3
        )

        # Should use max score (0.9) even though it was second
        assert len(fused_scores) == 1
        assert fused_scores[0] == 0.9
