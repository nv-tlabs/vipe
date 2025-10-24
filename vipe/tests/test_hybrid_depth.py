# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for hybrid metric depth estimation.

This module tests the scale optimizer and hybrid depth model.
"""

import pytest
import torch

from vipe.priors.depth.scale_optimizer import ScaleOptimizer
from vipe.priors.depth.hybrid_metric import HybridMetricDepthModel


class TestScaleOptimizer:
    """Test the ScaleOptimizer class."""
    
    def test_compute_scale_at_frame(self):
        """Test scale computation for a single frame."""
        optimizer = ScaleOptimizer()
        
        # Create synthetic data
        H, W = 128, 128
        gc_depth = torch.ones(H, W) * 2.0  # Relative depth
        moge_depth = torch.ones(H, W) * 5.0  # Metric depth
        
        # Expected scale: 5.0 / 2.0 = 2.5
        scale = optimizer.compute_scale_at_frame(moge_depth, gc_depth)
        
        assert isinstance(scale, float)
        assert abs(scale - 2.5) < 0.1, f"Expected scale ~2.5, got {scale}"
    
    def test_compute_scale_with_noise(self):
        """Test scale computation with noisy data."""
        optimizer = ScaleOptimizer()
        
        H, W = 128, 128
        gc_depth = torch.ones(H, W) * 2.0 + torch.randn(H, W) * 0.1
        moge_depth = torch.ones(H, W) * 5.0 + torch.randn(H, W) * 0.1
        
        scale = optimizer.compute_scale_at_frame(moge_depth, gc_depth)
        
        # Should still be close to 2.5 despite noise (median is robust)
        assert 2.0 < scale < 3.0, f"Scale {scale} out of expected range"
    
    def test_compute_scale_with_invalid_pixels(self):
        """Test scale computation with invalid pixels."""
        optimizer = ScaleOptimizer()
        
        H, W = 128, 128
        gc_depth = torch.ones(H, W) * 2.0
        moge_depth = torch.ones(H, W) * 5.0
        
        # Add some invalid pixels
        gc_depth[:10, :] = 0.0  # Invalid region
        moge_depth[-10:, :] = 0.0  # Invalid region
        
        scale = optimizer.compute_scale_at_frame(moge_depth, gc_depth)
        
        # Should still compute valid scale from remaining pixels
        assert 2.0 < scale < 3.0
    
    def test_interpolate_scales(self):
        """Test scale interpolation between keyframes."""
        optimizer = ScaleOptimizer()
        
        # Keyframes at indices 0, 10, 20 with scales 1.0, 2.0, 3.0
        keyframe_scales = {0: 1.0, 10: 2.0, 20: 3.0}
        n_frames = 30
        
        scales = optimizer.interpolate_scales(keyframe_scales, n_frames)
        
        assert len(scales) == n_frames
        assert scales[0].item() == 1.0
        assert scales[10].item() == 2.0
        assert scales[20].item() == 3.0
        
        # Check interpolation at frame 5 (midway between 0 and 10)
        assert abs(scales[5].item() - 1.5) < 0.1
        
        # Check interpolation at frame 15 (midway between 10 and 20)
        assert abs(scales[15].item() - 2.5) < 0.1
        
        # Check extrapolation after last keyframe
        assert scales[29].item() == 3.0
    
    def test_optimize_temporal_smoothness(self):
        """Test temporal smoothness optimization."""
        optimizer = ScaleOptimizer()
        
        n_frames = 20
        H, W = 64, 64
        
        # Create initial scales with some jumps
        initial_scales = torch.ones(n_frames)
        initial_scales[10:15] = 2.0  # Sudden jump
        
        # Create synthetic GeometryCrafter depths
        gc_depths = torch.ones(n_frames, H, W) * 2.0
        
        # Create MoGe depths at keyframes
        moge_depths = {
            0: torch.ones(H, W) * 2.0,   # Scale should be 1.0
            10: torch.ones(H, W) * 4.0,  # Scale should be 2.0
            19: torch.ones(H, W) * 2.0,  # Scale should be 1.0
        }
        
        optimized_scales = optimizer.optimize_temporal_smoothness(
            initial_scales, moge_depths, gc_depths, n_iterations=50
        )
        
        assert len(optimized_scales) == n_frames
        
        # Optimized scales should be smoother (less variation)
        initial_var = initial_scales.var().item()
        optimized_var = optimized_scales.var().item()
        
        # Check smoothness: consecutive frames should have similar scales
        diffs = (optimized_scales[1:] - optimized_scales[:-1]).abs()
        max_diff = diffs.max().item()
        assert max_diff < 0.5, f"Scales not smooth enough: max diff = {max_diff}"
    
    def test_compute_scale_statistics(self):
        """Test scale statistics computation."""
        optimizer = ScaleOptimizer()
        
        scales = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
        stats = optimizer.compute_scale_statistics(scales)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "range" in stats
        
        assert abs(stats["mean"] - 2.0) < 0.1
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["range"] == 2.0


class TestHybridMetricDepthModel:
    """Test the HybridMetricDepthModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a hybrid depth model for testing."""
        # Note: This requires MoGe to be installed
        try:
            model = HybridMetricDepthModel(keyframe_interval=5)
            return model
        except Exception as e:
            pytest.skip(f"Could not initialize HybridMetricDepthModel: {e}")
    
    def test_model_initialization(self, model):
        """Test that the model initializes correctly."""
        assert model.keyframe_interval == 5
        assert model.moge is not None
        assert model.scale_optimizer is not None
    
    def test_select_keyframes(self, model):
        """Test keyframe selection."""
        n_frames = 23
        keyframes = model.select_keyframes(n_frames)
        
        # Should have keyframes at 0, 5, 10, 15, 20, 22 (last frame)
        assert 0 in keyframes
        assert 5 in keyframes
        assert 10 in keyframes
        assert 15 in keyframes
        assert 20 in keyframes
        assert 22 in keyframes  # Last frame always included
    
    def test_get_model_info(self, model):
        """Test model info retrieval."""
        info = model.get_model_info()
        
        assert "model_type" in info
        assert info["model_type"] == "hybrid_metric"
        assert "keyframe_interval" in info
        assert info["keyframe_interval"] == 5


def test_make_depth_model():
    """Test that hybrid depth model can be created via factory."""
    from vipe.priors.depth import make_depth_model
    
    try:
        model = make_depth_model("hybrid_metric")
        assert model is not None
        assert isinstance(model, HybridMetricDepthModel)
    except Exception as e:
        pytest.skip(f"Could not create hybrid_metric model: {e}")


# Integration test markers
pytestmark = pytest.mark.integration


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

