# SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
# SPDX-License-Identifier: MIT
#
# Source: https://github.com/princeton-vl/DPVO
# See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

# Modified for MetricTrail integration.
# The following Apache-2.0 terms apply to NVIDIA modifications only.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vipe.metrictrail.dpvo_deps import CfgNode as CN  # [vipe] was: from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 4096

# bias patch selection towards high gradient regions?
_C.CENTROID_SEL_STRAT = 'RANDOM'

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

# Optional metric inverse-depth guidance. These defaults leave the official
# DPVO behavior unchanged; the MoGe-2 benchmark enables them in its config.
_C.METRIC_DEPTH = False
_C.METRIC_DEPTH_MIN = 0.2
_C.METRIC_DEPTH_MAX = 500.0
_C.METRIC_DEPTH_PRIOR_WEIGHT = 1.0
_C.METRIC_DEPTH_PRIOR_HUBER = 0.3
_C.METRIC_DEPTH_PRIOR_WARMUP = 12

# Loop closure
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15

# Classic loop closure
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3
_C.LOOP_RETR_THRESH = 0.04

# =============================================================================
# Metric global BA backend (MoGe-2 metric-depth mode, indoor handheld sequences).
#
# Every key below defaults to the ORIGINAL DPVO behaviour: with all of them at
# their defaults the code paths of dpvo/dpvo.py are byte-identical to the base
# fork (the fixed-lag metric local-BA mode, including its raises).  The three
# shipped presets are configs/dpvo/local.yaml (local only, none of these keys),
# configs/dpvo/gba.yaml (global BA group only: 7-Scenes / Wild-SLAM) and
# configs/dpvo/gba_refine.yaml (every group: TUM-style long handheld sequences with
# revisits and drift), relative to the MetricTrail repository root.  Section
# numbers below refer to the internal research notes the flags came from.
# =============================================================================

# --- 1. Global BA (EFFECTIVE_CHANGES §1, §1.2) -------------------------------
# METRIC_GLOBAL_BA: together with METRIC_DEPTH and LOOP_CLOSURE, DPV-SLAM's
# proximity loop edges (patchgraph.edges_loop) and its efficient global BA run
# WITHOUT pg.normalize() (which would wash out the MoGe metric scale) and WITH
# the same robust log-inverse-depth unary as the local BA; keyframe() retains
# the out-of-window factors for the backend.  CLASSIC_LOOP_CLOSURE (DBoW)
# stays forbidden with metric depth.
_C.METRIC_GLOBAL_BA = False
# Poses optimised by the metric global BA: 'active_min' = DPV-SLAM faithful
# t0 = min(active ii) (poses before the oldest active edge stay fixed);
# 'all' = t0 = 1 (every pose except the identity anchor at frame 0 is free).
_C.METRIC_GLOBAL_BA_T0 = 'active_min'

# --- 2. Scheduling (§5, variant V1) -----------------------------------------
# GBA_SCHEDULE_K > 0: also run the metric global BA every K accepted keyframes
# when stored inactive factors exist, independent of proximity loop edges
# (drift is pulled back before it accumulates).  0 = only when edges_loop fires.
_C.GBA_SCHEDULE_K = 0

# --- 3. Culled-frame interpolation (§6, fix F2; dpvo/burst_fixes.py) --------
# CULLED_INTERP: at terminate(), spread the discrepancy between the cull-time
# relative-pose chain of each run of culled frames and the final BA pose of
# the next kept keyframe along that run (keyframe poses untouched).  Removes
# the trajectory "snap" at the first kept keyframe after a culled run.
_C.CULLED_INTERP = False

# --- 4. Final re-measurement (§9, §12.1, §12.3; dpvo/remeasure_*.py) ---------
# GBA_REMEASURE_FINAL: final re-measured global BA in terminate() after the 12
# update passes and BEFORE the culled-frame interpolation.  'off' | 'one' (one
# pass of 8 x [refresh every stored+active factor with the update operator
# under the current poses, then global BA t0=1]) | 'two' (passes of 8 and 24
# iterations; the ViPE run(7)+run(24) analogue).  Requires METRIC_GLOBAL_BA.
_C.GBA_REMEASURE_FINAL = 'off'
# FMAP_STORE: fmap ring-buffer length (self.mem) while GBA_REMEASURE_FINAL is
# on: min(pmem, FMAP_STORE), so the matching features of every kept keyframe
# stay resident for refresh_factors (fp16, ~3.3 MB per keyframe at 378x518).
# Inert while GBA_REMEASURE_FINAL is 'off' (the ring stays at 36).
_C.FMAP_STORE = 512
# REMEASURE_MIN_AGE > 0: only edges with jj - ii >= MIN_AGE (keyframe-index
# age) are re-measured; the rest keep their stored target/weight.  Trades
# floor for plant/360 (§12.1); 0 in the shipped TUM preset.
_C.REMEASURE_MIN_AGE = 0
# Cold-start removal of the re-measurement passes (§12.3, arm f_carry):
# REMEASURE_BURNIN > 0: at the start of every pass run this many
#   refresh_factors calls WITHOUT a global BA in between (the GRU state settles
#   on the converged poses before any BA sees re-measured targets);
# REMEASURE_WARM_ACTIVE: pass 0 starts the GRU state of the still-active edges
#   from pg.net (their tracking-time state) instead of zeros;
# REMEASURE_CARRY_STATE: later passes keep the GRU state of the previous pass
#   instead of zeroing it.
_C.REMEASURE_BURNIN = 0
_C.REMEASURE_WARM_ACTIVE = False
_C.REMEASURE_CARRY_STATE = False

# --- 5. Scale latent (§12.2 proposal #3, §12.3 arm s1; dpvo/scale_latent.py) -
# GBA_SCALE_LATENT: at every global BA call (prior on) solve a zero-mean,
# parallax-gated, smoothed, clamped per-keyframe log-scale latent s_t from the
# globally adjusted patch depths vs the RAW MoGe prior and hand the global BA
# depth_prior = metric_disps_ * exp(s_t).  The weighted mean of s is removed,
# so the Sim(3) gauge stays pinned by the raw prior; the tracking-time local
# BA keeps the raw prior.  Fixes per-frame prior scale drift (floor).
_C.GBA_SCALE_LATENT = False
_C.GBA_SCALE_SMOOTH = 20.0           # lambda_smooth on (s_{t+1} - s_t)^2
_C.GBA_SCALE_CLAMP = 0.35            # |s_t| bound and |m_t| data-term gate (log units)
_C.GBA_SCALE_MIN_PATCHES = 32        # frames with fewer valid prior patches have no data term
_C.GBA_SCALE_MIN_PARALLAX_PX = 2.0   # frames whose mean tonly flow into t-1 is below this have no data term
_C.GBA_SCALE_LOCAL_USE = 'off'       # 'off' only ('frozen' = local BA reuses the latent: reserved, rejected at init)

# --- 6. Graph structure (§12.2 proposals #2 / #4, §12.3 arm rh; dpvo/graph_guards.py)
# LOOP_REVERSE_EDGES: for every (i old, j new) pair accepted by
# patchgraph.edges_loop also add the reverse edges (frame j's M patches
# observed in frame i) when frame i's fmap stays resident for the edge's
# active lifetime.  Requires GBA_REMEASURE_FINAL != 'off' (long fmap ring).
_C.LOOP_REVERSE_EDGES = False
# KEYFRAME_MAX_CULL > 0: keyframe() keeps the candidate regardless of the flow
# test once >= this many consecutive candidates were culled since the last
# kept keyframe AND their summed rotation path >= KEYFRAME_MAX_CULL_ROT_DEG
# (closes the 16-30 frame keyframe holes that carried the room/teddy breaks).
_C.KEYFRAME_MAX_CULL = 0
_C.KEYFRAME_MAX_CULL_ROT_DEG = 10.0

cfg = _C
