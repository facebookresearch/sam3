# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for io_utils extensionless video file handling (D99228861)."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sam3.model.io_utils import load_video_frames


class TestLoadVideoFramesRouting(unittest.TestCase):
    """Test that load_video_frames routes paths correctly based on extension."""

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mp4_extension_routes_to_video_loader(self, mock_load_video):
        """Paths with .mp4 extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/test_video.mp4",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mov_extension_routes_to_video_loader(self, mock_load_video):
        """Paths with .mov extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        load_video_frames(
            video_path="/tmp/test_video.mov",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_oil_path_routes_to_video_loader(self, mock_load_video):
        """Extensionless OIL paths should attempt video loading (D99228861 fix)."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="oil://fb_permanent/abc123def456",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_bare_hash_routes_to_video_loader(self, mock_load_video):
        """Bare hash paths without extension should attempt video loading."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/data/videos/abc123def456",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_path_raises_on_decode_failure(self, mock_load_video):
        """Extensionless path that fails to decode should raise NotImplementedError."""
        mock_load_video.side_effect = RuntimeError("Could not decode video")
        with self.assertRaises(NotImplementedError) as ctx:
            load_video_frames(
                video_path="oil://fb_permanent/corrupted_file",
                image_size=256,
                offload_video_to_cpu=True,
            )
        self.assertIn("failed to load", str(ctx.exception))
        self.assertIn("oil://fb_permanent/corrupted_file", str(ctx.exception))

    @patch("sam3.model.io_utils.load_video_frames_from_image_folder")
    def test_directory_routes_to_image_folder_loader(self, mock_load_folder):
        """Directory paths should route to load_video_frames_from_image_folder."""
        mock_load_folder.return_value = ("frames", 480, 640)
        with tempfile.TemporaryDirectory() as tmpdir:
            load_video_frames(
                video_path=tmpdir,
                image_size=256,
                offload_video_to_cpu=True,
            )
            mock_load_folder.assert_called_once()

    def test_dummy_video_pattern(self):
        """<load-dummy-video-N> pattern should return dummy frames."""
        frames, h, w = load_video_frames(
            video_path="<load-dummy-video-5>",
            image_size=64,
            offload_video_to_cpu=True,
        )
        self.assertEqual(frames.shape[0], 5)  # 5 frames
        self.assertEqual(h, 480)
        self.assertEqual(w, 640)

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_unknown_extension_routes_to_video_loader(self, mock_load_video):
        """Paths with unrecognized extensions should attempt video loading."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/video.xyz",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))
