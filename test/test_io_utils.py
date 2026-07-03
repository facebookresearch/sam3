# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for io_utils extensionless video file handling (D99228861)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sam3.model.io_utils import load_video_frames


class TestLoadVideoFramesRouting(unittest.TestCase):
    """Test that load_video_frames routes paths correctly based on extension."""

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mp4_extension_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
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
    def test_mov_extension_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Paths with .mov extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        load_video_frames(
            video_path="/tmp/test_video.mov",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_oil_path_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
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
    def test_extensionless_bare_hash_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
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
    def test_extensionless_path_raises_on_decode_failure(
        self, mock_load_video: MagicMock
    ) -> None:
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
    def test_directory_routes_to_image_folder_loader(
        self, mock_load_folder: MagicMock
    ) -> None:
        """Directory paths should route to load_video_frames_from_image_folder."""
        mock_load_folder.return_value = ("frames", 480, 640)
        with tempfile.TemporaryDirectory() as tmpdir:
            load_video_frames(
                video_path=tmpdir,
                image_size=256,
                offload_video_to_cpu=True,
            )
            mock_load_folder.assert_called_once()

    def test_dummy_video_pattern(self) -> None:
        """<load-dummy-video-N> pattern should return dummy frames."""
        frames, h, w = load_video_frames(
            video_path="<load-dummy-video-5>",
            image_size=64,
            offload_video_to_cpu=True,
        )
        self.assertEqual(frames.shape[0], 5)  # 5 frames
        self.assertEqual(h, 480)
        self.assertEqual(w, 640)

    def test_cv2_video_file_loader_scales_before_normalization(self) -> None:
        """OpenCV video loading should match normalized decoded uint8 frames."""
        try:
            import cv2
        except ImportError as exc:
            self.skipTest(f"OpenCV is required for this test: {exc}")

        image_size = 8
        source_height = 6
        source_width = 10
        img_mean = (0.5, 0.25, 0.75)
        img_std = (0.5, 0.25, 0.25)
        yy, xx = np.indices((source_height, source_width), dtype=np.uint16)
        frames_rgb = [
            np.stack(
                (
                    (xx * 23 + yy * 7) % 256,
                    (xx * 11 + yy * 17 + 3) % 256,
                    (xx * 5 + yy * 29 + 9) % 256,
                ),
                axis=-1,
            ).astype(np.uint8),
            np.stack(
                (
                    (xx * 13 + yy * 19 + 31) % 256,
                    (xx * 3 + yy * 41 + 47) % 256,
                    (xx * 37 + yy * 2 + 61) % 256,
                ),
                axis=-1,
            ).astype(np.uint8),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "tiny.avi")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                2.0,
                (source_width, source_height),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV could not create a temporary MJPG video")

            for frame_rgb in frames_rgb:
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            writer.release()

            decoded_frames = []
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                decoded_frames.append(
                    cv2.resize(
                        frame_rgb,
                        (image_size, image_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
            cap.release()
            self.assertEqual(len(decoded_frames), len(frames_rgb))

            expected = torch.from_numpy(np.stack(decoded_frames, axis=0))
            expected = expected.permute(0, 3, 1, 2).to(dtype=torch.float16)
            expected /= 255.0
            expected -= torch.tensor(img_mean, dtype=torch.float16).view(1, 3, 1, 1)
            expected /= torch.tensor(img_std, dtype=torch.float16).view(1, 3, 1, 1)

            frames, height, width = load_video_frames(
                video_path=video_path,
                image_size=image_size,
                offload_video_to_cpu=True,
                img_mean=img_mean,
                img_std=img_std,
                video_loader_type="cv2",
            )

        self.assertEqual((height, width), (source_height, source_width))
        self.assertEqual(frames.dtype, torch.float16)
        torch.testing.assert_close(frames, expected, rtol=0, atol=1e-6)

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_unknown_extension_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Paths with unrecognized extensions should attempt video loading."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/video.xyz",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))
