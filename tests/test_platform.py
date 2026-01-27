# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for sam3.platform module."""

import sys
from unittest import mock

import pytest
from sam3.platform import (
    check_platform_compatibility,
    get_platform_info,
    get_pytorch_index_url,
    get_recommended_python,
    is_jetson,
)


class TestIsJetson:
    """Tests for is_jetson() function."""

    def test_returns_bool(self):
        """is_jetson() should return a boolean."""
        result = is_jetson()
        assert isinstance(result, bool)

    def test_detects_jetson_when_file_exists(self):
        """Should return True when /etc/nv_tegra_release exists."""
        with mock.patch("os.path.exists", return_value=True):
            assert is_jetson() is True

    def test_detects_non_jetson_when_file_missing(self):
        """Should return False when /etc/nv_tegra_release doesn't exist."""
        with mock.patch("os.path.exists", return_value=False):
            assert is_jetson() is False


class TestGetPlatformInfo:
    """Tests for get_platform_info() function."""

    def test_returns_dict(self):
        """get_platform_info() should return a dictionary."""
        info = get_platform_info()
        assert isinstance(info, dict)

    def test_contains_required_keys(self):
        """Result should contain all required keys."""
        info = get_platform_info()
        required_keys = [
            "is_jetson",
            "l4t_release",
            "device_model",
            "python_version",
            "platform_machine",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_python_version_matches_current(self):
        """python_version should match current Python."""
        info = get_platform_info()
        expected = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert info["python_version"] == expected


class TestGetRecommendedPython:
    """Tests for get_recommended_python() function."""

    def test_returns_string(self):
        """get_recommended_python() should return a string."""
        result = get_recommended_python()
        assert isinstance(result, str)

    def test_returns_valid_version(self):
        """Should return either 3.10 (Jetson) or 3.12 (x86)."""
        result = get_recommended_python()
        assert result in ["3.10", "3.12"]

    def test_returns_310_for_jetson(self):
        """Should return 3.10 when on Jetson."""
        with mock.patch("sam3.platform.is_jetson", return_value=True):
            assert get_recommended_python() == "3.10"

    def test_returns_312_for_x86(self):
        """Should return 3.12 when not on Jetson."""
        with mock.patch("sam3.platform.is_jetson", return_value=False):
            assert get_recommended_python() == "3.12"


class TestGetPytorchIndexUrl:
    """Tests for get_pytorch_index_url() function."""

    def test_returns_string(self):
        """get_pytorch_index_url() should return a string."""
        result = get_pytorch_index_url()
        assert isinstance(result, str)

    def test_returns_https_url(self):
        """Should return a valid HTTPS URL."""
        result = get_pytorch_index_url()
        assert result.startswith("https://")

    def test_returns_jetson_url_on_jetson(self):
        """Should return Jetson AI Lab URL on Jetson."""
        with mock.patch("sam3.platform.is_jetson", return_value=True):
            url = get_pytorch_index_url()
            assert "jetson-ai-lab" in url

    def test_returns_pytorch_url_on_x86(self):
        """Should return PyTorch URL on x86."""
        with mock.patch("sam3.platform.is_jetson", return_value=False):
            url = get_pytorch_index_url()
            assert "pytorch.org" in url


class MockVersionInfo:
    """Mock for sys.version_info with named attributes."""

    def __init__(self, major: int, minor: int, micro: int = 0):
        self.major = major
        self.minor = minor
        self.micro = micro

    def __getitem__(self, index):
        return (self.major, self.minor, self.micro)[index]


class TestCheckPlatformCompatibility:
    """Tests for check_platform_compatibility() function."""

    def test_returns_none_when_compatible(self):
        """Should return None when Python version matches platform."""
        # Mock current Python as 3.10 on Jetson
        mock_version = MockVersionInfo(3, 10, 0)
        with mock.patch("sam3.platform.is_jetson", return_value=True):
            with mock.patch("sam3.platform.sys.version_info", mock_version):
                result = check_platform_compatibility(warn=False)
                assert result is None

    def test_returns_message_when_incompatible(self):
        """Should return warning message when Python version mismatches."""
        # Mock current Python as 3.12 on Jetson (should warn)
        mock_version = MockVersionInfo(3, 12, 0)
        with mock.patch("sam3.platform.is_jetson", return_value=True):
            with mock.patch("sam3.platform.sys.version_info", mock_version):
                result = check_platform_compatibility(warn=False)
                assert result is not None
                assert "3.10" in result

    def test_emits_warning_when_warn_true(self):
        """Should emit UserWarning when warn=True and incompatible."""
        mock_version = MockVersionInfo(3, 12, 0)
        with mock.patch("sam3.platform.is_jetson", return_value=True):
            with mock.patch("sam3.platform.sys.version_info", mock_version):
                with pytest.warns(UserWarning):
                    check_platform_compatibility(warn=True)

    def test_no_warning_when_warn_false(self):
        """Should not emit warning when warn=False."""
        mock_version = MockVersionInfo(3, 12, 0)
        with mock.patch("sam3.platform.is_jetson", return_value=True):
            with mock.patch("sam3.platform.sys.version_info", mock_version):
                # This should not raise any warnings
                import warnings

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    check_platform_compatibility(warn=False)
                    # Filter for UserWarnings from our module
                    user_warnings = [
                        x for x in w if issubclass(x.category, UserWarning)
                    ]
                    assert len(user_warnings) == 0
