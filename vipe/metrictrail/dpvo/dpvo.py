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

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .metric_depth import patch_metric_priors, prepare_metric_disparity
# Metric global BA backend helpers (dpvo/config.py, block "metric global BA backend"; every flag
# defaults off = base DPVO behaviour, so with the defaults none of these is ever called).
from .burst_fixes import culled_interp, robust_log_scale_offsets   # CULLED_INTERP; robust per-frame log-scale offsets
from .remeasure_gates import age_mask                               # REMEASURE_MIN_AGE
from .scale_latent import gba_scale_latent                          # GBA_SCALE_LATENT
from .remeasure_burnin import (                                     # GBA_REMEASURE_FINAL passes, REMEASURE_BURNIN / WARM_ACTIVE / CARRY_STATE
    pass_schedule, warm_start_active, mean_weight_by_age_class, oob_count, move_stats, keyframe_scale_summary,
)
from .graph_guards import (TAG_LOOP_FWD, TAG_LOOP_REV, TAG_NAMES, HoleGuard, loop_pairs,   # LOOP_REVERSE_EDGES, KEYFRAME_MAX_CULL
                           reverse_loop_edges, weight_mean_by_tag_and_age)
from .utils import *

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            raise ValueError("CLASSIC_LOOP_CLOSURE must be disabled")
        if self.cfg.METRIC_DEPTH and self.cfg.LOOP_CLOSURE and not self.cfg.METRIC_GLOBAL_BA:
            raise ValueError("metric-depth proximity edges require METRIC_GLOBAL_BA=True")
        if self.cfg.METRIC_GLOBAL_BA:
            if not (self.cfg.METRIC_DEPTH and self.cfg.LOOP_CLOSURE):
                raise ValueError(
                    "METRIC_GLOBAL_BA requires METRIC_DEPTH=True and "
                    "LOOP_CLOSURE=True (proximity loop edges feed the backend)"
                )
            if self.cfg.METRIC_GLOBAL_BA_T0 not in ("active_min", "all"):
                raise ValueError(
                    "METRIC_GLOBAL_BA_T0 must be 'active_min' or 'all', got "
                    f"{self.cfg.METRIC_GLOBAL_BA_T0!r}"
                )
        # Scheduling (GBA_SCHEDULE_K) only makes sense with the metric global BA.
        if not self.cfg.METRIC_GLOBAL_BA and int(self.cfg.GBA_SCHEDULE_K) > 0:
            raise ValueError("GBA_SCHEDULE_K requires METRIC_GLOBAL_BA=True")
        # Final re-measurement (M1): guards only; defaults are off.
        if str(self.cfg.GBA_REMEASURE_FINAL) not in ("off", "one", "two"):
            raise ValueError("GBA_REMEASURE_FINAL must be 'off', 'one' or 'two'")
        if str(self.cfg.GBA_REMEASURE_FINAL) != "off" and not self.cfg.METRIC_GLOBAL_BA:
            raise ValueError("GBA_REMEASURE_FINAL requires METRIC_GLOBAL_BA=True")
        if int(self.cfg.FMAP_STORE) < 36:
            raise ValueError("FMAP_STORE must be at least 36 (the default fmap ring)")
        if int(self.cfg.REMEASURE_BURNIN) < 0:
            raise ValueError("REMEASURE_BURNIN must be >= 0")
        # Graph-structure guards (defaults off).  Reverse edges read the OLD frame's fmap
        # in corr() while active, so the long fmap ring of M1 is required.
        if bool(self.cfg.LOOP_REVERSE_EDGES):
            if not (self.cfg.METRIC_GLOBAL_BA and self.cfg.LOOP_CLOSURE):
                raise ValueError("LOOP_REVERSE_EDGES requires METRIC_GLOBAL_BA=True and LOOP_CLOSURE=True")
            if str(self.cfg.GBA_REMEASURE_FINAL) == "off":
                raise ValueError("LOOP_REVERSE_EDGES requires GBA_REMEASURE_FINAL != 'off' "
                                 "(self.mem = min(pmem, FMAP_STORE) keeps old fmaps resident)")
        if int(self.cfg.KEYFRAME_MAX_CULL) < 0 or float(self.cfg.KEYFRAME_MAX_CULL_ROT_DEG) < 0.0:
            raise ValueError("KEYFRAME_MAX_CULL / KEYFRAME_MAX_CULL_ROT_DEG must be >= 0")
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0
        self.metric_prior_valid_patches = 0
        self.metric_prior_total_patches = 0
        self.metric_prior_valid_pixel_fractions = []
        self.metric_keyframes_seen = 0
        self.metric_prior_regularization_enabled = False
        self.metric_prior_ba_calls = 0

        # METRIC_GLOBAL_BA diagnostics (counters only; reported by
        # metric_prior_diagnostics() solely when the flag is on).
        self.global_ba_calls = 0
        self.global_ba_calls_with_metric_prior = 0
        self.global_ba_t0_history = []
        self.global_ba_edge_history = []
        self.loop_edges_added = 0
        self.loop_edge_batches = 0
        self.inactive_factors_peak = 0
        self.inactive_factors_at_terminate = None
        self.active_factors_at_terminate = None

        # Scheduled global BA diagnostics (GBA_SCHEDULE_K; reported by
        # metric_prior_diagnostics only when the key is non-default).
        self.global_ba_scheduled_calls = 0
        self._gba_sched_last_n = -1               # once per n value
        self.keyframe_prior_diag = None           # terminate-time sidecar: per-keyframe m_t / parallax_t

        # Culled-frame interpolation diagnostics (CULLED_INTERP, dpvo/burst_fixes.py).
        self.culled_interp_stats = None

        # Final re-measurement (M1) diagnostics (reported only when GBA_REMEASURE_FINAL != 'off').
        self.remeasure_passes = []                # M1: one dict per pass
        self.remeasure_seconds = 0.0              # M1: wall seconds added in terminate()
        self.remeasure_edges_missing_fmap = 0     # M1: edge-refreshes skipped (features evicted)

        # Scale-latent guards and diagnostics (reported only when GBA_SCALE_LATENT is on).
        if str(self.cfg.GBA_SCALE_LOCAL_USE) not in ("off", "frozen"):
            raise ValueError("GBA_SCALE_LOCAL_USE must be 'off' or 'frozen'")
        if bool(self.cfg.GBA_SCALE_LATENT) and str(self.cfg.GBA_SCALE_LOCAL_USE) != "off":
            raise NotImplementedError("GBA_SCALE_LOCAL_USE 'frozen' is not implemented in DPVO_h (run 'off' first)")
        if bool(self.cfg.GBA_SCALE_LATENT) and not self.cfg.METRIC_GLOBAL_BA:
            raise ValueError("GBA_SCALE_LATENT needs METRIC_GLOBAL_BA")
        self.scale_latent_calls = 0               # GBA calls where s_t was solved and applied
        self.scale_latent_std_hist = []           # std(s) per call
        self.scale_latent_maxabs_hist = []        # max|s| per call
        self.scale_latent_gated_hist = []         # [parallax-gated frac, min-patch-gated frac, clamp-gated frac] per call
        self.scale_latent_wmean_abs_max = 0.0     # max over calls of |sum w s / sum w| after the clamp
        self.scale_latent_clamped_total = 0
        self.scale_latent_solve_seconds = 0.0
        self.scale_latent_last = None             # per-keyframe snapshot of the last call

        # Graph-structure diagnostics (reported only when LOOP_REVERSE_EDGES or KEYFRAME_MAX_CULL is non-default).
        self.loop_forward_pairs = 0               # unique (i, j) pairs accepted by edges_loop
        self.loop_reverse_pairs = 0               # pairs that received a reverse edge set
        self.loop_reverse_pairs_skipped = 0       # pairs skipped (old fmap not resident for the edge lifetime)
        self.loop_reverse_edges_added = 0
        self.loop_reverse_batches = 0
        self.tag_w_sum = {t: 0.0 for t in TAG_NAMES}   # running mean of update() weights per tag
        self.tag_w_cnt = {t: 0 for t in TAG_NAMES}
        self.remeasure_final_weight_by_tag = None       # M1 last pass, final refreshed weights per tag / age
        self.hole_guard = HoleGuard(int(self.cfg.KEYFRAME_MAX_CULL), float(self.cfg.KEYFRAME_MAX_CULL_ROT_DEG))

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory
        if self.cfg.METRIC_GLOBAL_BA and str(self.cfg.GBA_REMEASURE_FINAL) != "off":
            # M1: keep the frame-side matching features (fmap1_/fmap2_) of
            # every kept keyframe so stored factors can be re-measured at the
            # end. Ring semantics are unchanged (corr() indexes jj % mem and
            # keyframe() shifts slot i % mem); only the length grows.
            self.mem = min(self.pmem, int(self.cfg.FMAP_STORE))

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)
        # Per-edge tag beside pg.net / pg.weight_inac, aligned with pg.ii /
        # pg.ii_inac: 0 regular, 1 forward loop (edges_loop), 2 reverse loop.
        # Always allocated and maintained; only read by the flag-guarded
        # backend paths, never with every backend flag off.
        self.pg.tag = torch.zeros(0, dtype=torch.long, device="cuda")
        self.pg.tag_inac = torch.zeros(0, dtype=torch.long, device="cuda")


        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()


    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):


        if self.cfg.LOOP_CLOSURE:
            lii, ljj = self.pg.edges_loop()
            self.append_factors(lii, ljj, tag=TAG_LOOP_FWD)
            self._count_loop_edges(lii)
            if bool(self.cfg.LOOP_REVERSE_EDGES):
                self._append_reverse_loop_edges(lii, ljj)

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        if self.cfg.METRIC_GLOBAL_BA and str(self.cfg.GBA_REMEASURE_FINAL) != "off":
            # M1: final re-measured global BA (ViPE run(7)+run(24) analogue).
            # Order: 12 updates -> M1 here -> trajectory extraction -> culled-frame
            # interpolation LAST (it must see the keyframe poses as left by these passes).
            self._remeasure_final()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        if self.cfg.CULLED_INTERP:
            # F2: spread the BA correction of every kept keyframe over the run
            # of culled frames preceding it (keyframe poses untouched). Done on
            # the CPU: ~one small SE3 op per run.
            data = poses.data.detach().cpu()
            kept = torch.zeros(self.counter, dtype=torch.bool)
            kept[[int(t) for t in self.traj.keys()]] = True
            data, stats = culled_interp(data, kept, self.pg.delta_next)
            self.culled_interp_stats = stats
            poses = SE3(data)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        if self.cfg.METRIC_GLOBAL_BA:
            # The backend factors have served their last global BA above; the
            # trajectory is already extracted. Record their count, then release
            # them so the post-terminate diagnostics truthfully report zero
            # retained inactive factors (metric_prior_diagnostics documents
            # the pre-release count under inactive_factors_at_terminate).
            self.inactive_factors_at_terminate = int(self.pg.ii_inac.numel())
            self.active_factors_at_terminate = int(self.pg.ii.numel())
            # keyframe_prior sidecar (read-only; poses/patches are final here): per-keyframe
            # robust log-scale offset vs the MoGe prior and parallax. Never fatal.
            try:
                self.keyframe_prior_diag = self._keyframe_prior_diagnostics()
            except Exception as error:  # noqa: BLE001
                self.keyframe_prior_diag = {"error": repr(error)[:300]}
            self._release_inactive_factors()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def _count_loop_edges(self, lii):
        """Diagnostics only: edges appended through patchgraph.edges_loop."""
        if lii.numel() > 0:
            self.loop_edges_added += int(lii.numel())
            self.loop_edge_batches += 1

    def _append_reverse_loop_edges(self, lii, ljj):
        """LOOP_REVERSE_EDGES: for every accepted edges_loop pair (i old,
        j new) also observe frame j's M patches in frame i (kk_r = j*M + m,
        jj_r = i; append_factors derives ii = ix[kk] = j), tag 2.  Only pairs
        whose old frame's fmap stays resident for the edge's whole active
        lifetime are added: (n-1-i) < mem - (REMOVAL_WINDOW+1)  [spec: mem-1;
        identical on TUM fr1 where <= 166 keyframes are kept vs mem 512].
        jj - ii < 0 for these edges, so keyframe()'s lc keep-rule never applies
        and they are stored by the normal ix[kk] < n - REMOVAL_WINDOW rule."""
        if lii.numel() == 0:
            return
        pairs = loop_pairs(self.ix[lii[::self.M]], ljj[::self.M])
        self.loop_forward_pairs += int(pairs.shape[0])
        kk_r, jj_r, n_used, n_skip = reverse_loop_edges(
            pairs, self.n, self.mem, self.M, int(self.cfg.REMOVAL_WINDOW) + 1)
        self.loop_reverse_pairs += n_used
        self.loop_reverse_pairs_skipped += n_skip
        if kk_r.numel() > 0:
            self.append_factors(kk_r, jj_r, tag=TAG_LOOP_REV)
            self.loop_reverse_edges_added += int(kk_r.numel())
            self.loop_reverse_batches += 1

    def _tag_weight_stats(self, weight):
        """Diagnostics: running mean of update()'s in-graph weights per tag."""
        w = weight[0].float().mean(dim=-1)
        for t in TAG_NAMES:
            m = self.pg.tag == t
            c = int(m.sum().item())
            if c:
                self.tag_w_sum[t] += float(w[m].sum().item())
                self.tag_w_cnt[t] += c

    def _release_inactive_factors(self):
        """Drop the retained backend factors (METRIC_GLOBAL_BA, after terminate)."""
        device = self.pg.ii_inac.device
        self.pg.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.pg.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.pg.kk_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.pg.weight_inac = self.pg.weight_inac[:, :0]
        self.pg.target_inac = self.pg.target_inac[:, :0]
        self.pg.tag_inac = self.pg.tag_inac[:0]

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    @torch.no_grad()
    def refresh_factors(self, ii, jj, kk, target, weight, net, block_frames=40, refresh_mask=None):
        """M1: re-measure factors under the current poses/patches. Re-runs the
        correlation lookup + update operator for every edge whose features are
        still resident, in blocks grouped by source frame ii (the operator
        aggregates over edges sharing kk and sharing (ii, jj); both groups live
        entirely inside one source-frame block, loop edges included).
        target / weight / net are updated IN PLACE (net = GRU state carried
        across the calls of one pass). Edges whose frame features were evicted
        from the rings keep their stored target/weight; they are counted.
        refresh_mask (bool, E; REMEASURE_MIN_AGE age gate, default None = every
        edge): only masked-in edges are refreshed; the rest are neither read nor
        written (n_missing counts eligible edges only).  Only _remeasure_final
        calls this method; the tracking-time update() never does.
        Also counts (diagnostic only) the refreshed edges whose reprojected
        patch centre lies outside the 1/4-res image.
        Returns (n_refreshed, n_missing, n_oob)."""
        E = int(ii.numel())
        if E == 0:
            return 0, 0, 0
        n = self.n
        resident = self._resident_mask(ii, jj)
        if refresh_mask is not None:
            n_missing = int((~resident & refresh_mask).sum().item())
            resident = resident & refresh_mask
        else:
            n_missing = int((~resident).sum().item())
        n_refreshed = 0
        n_oob = torch.zeros((), dtype=torch.long, device=ii.device)
        wd4, ht4 = self.wd // self.RES, self.ht // self.RES
        P = self.P
        for b0 in range(0, n, block_frames):
            sel = resident & (ii >= b0) & (ii < b0 + block_frames)
            idx = sel.nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            ii_b, jj_b, kk_b = ii[idx], jj[idx], kk[idx]
            coords = self.reproject(indicies=(ii_b, jj_b, kk_b))
            with autocast(enabled=self.cfg.MIXED_PRECISION):
                corr = self.corr(coords, indicies=(kk_b, jj_b))
                ctx = self.imap[:, kk_b % (self.M * self.pmem)]
                net_b, (delta, weight_b, _) = \
                    self.network.update(net[:, idx], ctx, corr, None, ii_b, jj_b, kk_b)
            net[:, idx] = net_b.to(net.dtype)
            target[:, idx] = coords[..., P // 2, P // 2] + delta.float()
            weight[:, idx] = weight_b.float()
            n_refreshed += int(idx.numel())
            n_oob = n_oob + oob_count(coords[..., P // 2, P // 2], wd4, ht4)
        return n_refreshed, n_missing, int(n_oob.item())

    def _resident_mask(self, ii, jj):
        """Edges whose frame features (jj, ring self.mem) and patch features (ii, ring self.pmem) are still resident:
        frame f's fmap is resident iff n-1-f < mem; patch features iff n-1-f < pmem."""
        n = self.n
        return ((n - 1 - jj) < self.mem) & ((n - 1 - ii) < self.pmem) & (jj >= 0) & (ii >= 0)

    @torch.no_grad()
    def _remeasure_final(self):
        """Final re-measurement driver (M1; terminate(), after the 12 update passes).
        NOTE: callers typically wrap slam(...) in torch.inference_mode() but call
        terminate() outside it; without no_grad the in-place net/target writes
        of every block of every iteration would chain into one retained autograd
        graph (tens of GiB on a long sequence). no_grad changes no value.
        'one' = one pass
        of 8 x [refresh_factors over inactive+active edges, global BA t0=1];
        'two' = passes of 8 and 24 iterations (remeasure_burnin.pass_schedule),
        GRU state zeroed at the start of each pass unless REMEASURE_CARRY_STATE.
        The metric prior enters exactly as in every other global BA call (scale
        latent included when on). Diagnostics per pass: median / p90 of
        ||target_new - target_old|| over the inactive edges after the first
        refresh (vs the stored targets) and over the whole pass, stale fraction
        (> 3 px), mean weight before/after, seconds.
        Cold-start removal (all defaults off):
          REMEASURE_WARM_ACTIVE: pass 0 starts the active rows of the GRU
            state from pg.net; REMEASURE_CARRY_STATE: later passes keep it;
          REMEASURE_BURNIN: refresh_factors calls without BA at pass start.
        REMEASURE_MIN_AGE > 0 restricts the refresh to edges with jj - ii >= age.
        Extra per-pass diagnostics: per-iteration target movement / inactive
        mean weight, burn-in last-step movement, refreshed weights by edge age
        class at pass start (after burn-in) and end, out-of-image centres."""
        import time
        t_start = time.time()
        schedule = pass_schedule(str(self.cfg.GBA_REMEASURE_FINAL))
        burnin = int(self.cfg.REMEASURE_BURNIN)
        warm_active = bool(self.cfg.REMEASURE_WARM_ACTIVE)
        carry_state = bool(self.cfg.REMEASURE_CARRY_STATE)
        n_inac = int(self.pg.ii_inac.numel())
        n_act = int(self.pg.ii.numel())
        E = n_inac + n_act
        if E == 0:
            self.remeasure_passes.append({"skipped": "no factors"})
            self.remeasure_seconds += time.time() - t_start
            return
        ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        kk = torch.cat((self.pg.kk_inac, self.pg.kk))
        target = torch.cat((self.pg.target_inac, self.pg.target), dim=1).float().contiguous()
        weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1).float().contiguous()
        net = torch.zeros(1, E, self.DIM, **self.kwargs)
        # age gate (default 0 -> every edge eligible, refresh_mask None)
        min_age = int(self.cfg.REMEASURE_MIN_AGE)
        refresh_mask = age_mask(ii, jj, min_age) if min_age > 0 else None
        n_eligible = int(refresh_mask.sum().item()) if refresh_mask is not None else E
        # edges the refresh actually touches (diagnostics only)
        refreshed_sel = self._resident_mask(ii, jj)
        if refresh_mask is not None:
            refreshed_sel = refreshed_sel & refresh_mask

        def _stats(t_new, t_old, w_before, w_after):
            d = (t_new[:, :n_inac] - t_old[:, :n_inac]).norm(dim=-1).float().reshape(-1)
            d = torch.nan_to_num(d, nan=1e3, posinf=1e3)
            if d.numel() == 0:
                return {"median_px": None, "p90_px": None, "stale_fraction_gt3px": None,
                        "mean_weight_before": None, "mean_weight_after": None}
            return {
                "median_px": float(d.median().item()),
                "p90_px": float(torch.quantile(d.clamp(max=1e3), 0.9).item()),
                "stale_fraction_gt3px": float((d > 3.0).float().mean().item()),
                "mean_weight_before": float(w_before[:, :n_inac].mean().item()),
                "mean_weight_after": float(w_after[:, :n_inac].mean().item()),
            }

        def _w_inac(w):
            return float(w[:, :n_inac].mean().item()) if n_inac else None

        for pass_id, iters in enumerate(schedule):
            t_pass = time.time()
            if pass_id == 0 or not carry_state:
                net.zero_()
            warm_rows = 0
            if pass_id == 0 and warm_active:
                # pg.net rows are in pg.ii order = the active slice of cat(ii_inac, ii)
                warm_rows = warm_start_active(net, self.pg.net, n_inac)
            target_pass0 = target.clone()
            weight_pass0 = weight.clone()
            # burn-in: refresh without BA so the GRU state settles on the converged poses
            burn = None
            for b in range(burnin):
                t_b = target.clone()
                n_ref_b, n_miss_b, _ = self.refresh_factors(ii, jj, kk, target, weight, net, refresh_mask=refresh_mask)
                if b == burnin - 1:
                    med_b, p90_b = move_stats(target, t_b, n_inac)
                    med_s, p90_s = move_stats(target, target_pass0, n_inac)
                    burn = {"steps": int(burnin), "last_step_median_px": med_b, "last_step_p90_px": p90_b,
                            "vs_stored_median_px": med_s, "vs_stored_p90_px": p90_s,
                            "mean_weight_inactive_after": _w_inac(weight), "edges_refreshed": int(n_ref_b),
                            "edges_missing_fmap": int(n_miss_b)}
            weight_by_age_start = mean_weight_by_age_class(weight, ii, jj, refreshed_sel)
            iter_median_px, iter_p90_px, iter_mean_weight_inactive = [], [], []
            first = None
            missing_pass = 0
            oob_pass = 0
            refreshed_pass = 0
            for it in range(iters):
                t_prev = target.clone()
                n_ref, n_miss, n_oob = self.refresh_factors(ii, jj, kk, target, weight, net, refresh_mask=refresh_mask)
                missing_pass += n_miss
                oob_pass += n_oob
                refreshed_pass += n_ref
                med_it, p90_it = move_stats(target, t_prev, n_inac)
                iter_median_px.append(med_it)
                iter_p90_px.append(p90_it)
                iter_mean_weight_inactive.append(_w_inac(weight))
                if it == 0:
                    # vs the STORED targets/weights at pass start (measured after the burn-in refreshes when on)
                    first = _stats(target, target_pass0, weight_pass0, weight)
                    first.update({"edges_refreshed": int(n_ref), "edges_missing_fmap": int(n_miss)})
                # write the refreshed measurements back so __run_global_BA sees them
                self.pg.target_inac = target[:, :n_inac]
                self.pg.weight_inac = weight[:, :n_inac]
                self.pg.target = target[:, n_inac:]
                self.pg.weight = weight[:, n_inac:]
                self.__run_global_BA(t0_override=1)
            torch.cuda.synchronize()
            rec = {
                "pass": int(pass_id),
                "iterations": int(iters),
                "edges_total": int(E),
                "edges_inactive": int(n_inac),
                "edges_active": int(n_act),
                "first_refresh_vs_stored": first,
                "whole_pass": _stats(target, target_pass0, weight_pass0, weight),
                "edge_refreshes_missing_fmap": int(missing_pass),
                "seconds": float(time.time() - t_pass),
                # age-gate diagnostics
                "age_gate": int(min_age),
                "edges_eligible": int(n_eligible),
                "edges_remeasured_fraction": float(n_eligible) / float(E),
                # cold-start diagnostics
                "burnin": burn,
                "warm_start_rows": int(warm_rows),
                "carried_state": bool(pass_id > 0 and carry_state),
                "iter_median_px": iter_median_px,
                "iter_p90_px": iter_p90_px,
                "iter_mean_weight_inactive": iter_mean_weight_inactive,
                "weight_by_age_pass_start": weight_by_age_start,
                "weight_by_age_pass_end": mean_weight_by_age_class(weight, ii, jj, refreshed_sel),
                "oob_centre_count": int(oob_pass),
                "oob_centre_fraction": (float(oob_pass) / float(refreshed_pass)) if refreshed_pass else None,
            }
            self.remeasure_edges_missing_fmap += int(missing_pass)
            self.remeasure_passes.append(rec)
        if bool(self.cfg.LOOP_REVERSE_EDGES):
            # final refreshed weight per tag / age class (end of the last pass)
            tag_all = torch.cat((self.pg.tag_inac, self.pg.tag))
            self.remeasure_final_weight_by_tag = weight_mean_by_tag_and_age(weight, tag_all, ii, jj)
        # leave the graph holding the final measurements (contiguous copies)
        self.pg.target_inac = target[:, :n_inac].contiguous()
        self.pg.weight_inac = weight[:, :n_inac].contiguous()
        self.pg.target = target[:, n_inac:].contiguous()
        self.pg.weight = weight[:, n_inac:].contiguous()
        self.remeasure_seconds += time.time() - t_start

    def append_factors(self, ii, jj, tag=0):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)
        self.pg.tag = torch.cat([self.pg.tag, torch.full((len(ii),), int(tag), dtype=torch.long, device="cuda")])

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
            self.pg.tag_inac = torch.cat((self.pg.tag_inac, self.pg.tag[m]))
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        self.pg.tag = self.pg.tag[~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        # KEYFRAME_MAX_CULL: keep the candidate regardless of the flow
        # test once >= G candidates were culled since the last kept keyframe
        # AND their summed rotation path >= theta (False when the flag is 0).
        guard = self.hole_guard.should_keep()
        if m / 2 < self.cfg.KEYFRAME_THRESH and not guard:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)
            if self.hole_guard.enabled:
                # dP is the cumulative relative pose from the last kept
                # keyframe (k-1 stays the same kept frame through a run).
                self.hole_guard.on_cull(dP.data)
            if self.cfg.CULLED_INTERP:
                # F2: also remember the cull-time relative pose to the NEXT
                # buffered frame (k+1 <= n-3 always exists), so terminate()
                # can chain a culled run to the next kept keyframe.
                dPn = SE3(self.pg.poses_[k+1]) * SE3(self.pg.poses_[k]).inv()
                self.pg.delta_next[int(t1)] = (int(self.pg.tstamps_[k+1]), dPn.data.clone())

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.metric_disps_[i] = self.pg.metric_disps_[i+1]
                self.pg.metric_weights_[i] = self.pg.metric_weights_[i+1]
                self.pg.metric_logscale_[i] = self.pg.metric_logscale_[i+1]   # scale latent (GBA_SCALE_LATENT)
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M
            self.pg.metric_disps_[self.n].zero_()
            self.pg.metric_weights_[self.n].zero_()
            self.pg.metric_logscale_[self.n].zero_()   # scale latent (GBA_SCALE_LATENT)

        elif self.hole_guard.enabled:
            k = self.n - self.cfg.KEYFRAME_INDEX
            self.hole_guard.on_keep(int(self.pg.tstamps_[k]), forced=bool(m / 2 < self.cfg.KEYFRAME_THRESH))

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        # In metric mode global BA is deliberately disabled, so retaining
        # inactive factors would only grow GPU memory and torch.cat cost over
        # the full sequence. Preserve the official behavior outside this mode.
        # METRIC_GLOBAL_BA re-enables the backend, so it stores them again
        # exactly as upstream DPV-SLAM does.  Index invariant (why the stored
        # ii/jj/kk never need the shift applied above): a frontend or reverse
        # edge is stored only once ix[kk] < n - REMOVAL_WINDOW, with
        # |jj - ii| < PATCH_LIFETIME, and a forward loop edge only once
        # jj <= n - OPTIMIZATION_WINDOW, so every stored index is below
        # k = n - KEYFRAME_INDEX (n - 9 / n - 10 vs n - 4 with the presets),
        # the only frame keyframe() culls; k never decreases between calls.
        self.remove_factors(
            to_remove,
            store=(not self.cfg.METRIC_DEPTH) or bool(self.cfg.METRIC_GLOBAL_BA),
        )
        if self.cfg.METRIC_GLOBAL_BA:
            self.inactive_factors_peak = max(
                self.inactive_factors_peak, int(self.pg.ii_inac.numel())
            )

    def __run_global_BA(self, t0_override=None):
        """ Global bundle adjustment
         Includes both active and inactive edges.
         t0_override (M1 final re-measured passes only) fixes t0 (=1). """
        if self.cfg.METRIC_DEPTH and not self.cfg.METRIC_GLOBAL_BA:
            raise RuntimeError(
                "Metric-depth mode is a fixed-lag local-BA experiment and cannot "
                "use DPVO's scale-normalizing global BA"
            )
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        if self.cfg.METRIC_GLOBAL_BA:
            # Metric-scale global BA: NO pg.normalize() (it divides every
            # patch inverse depth by their mean and rescales the translations,
            # which would destroy the MoGe-2 metric scale). The same robust
            # log-inverse-depth unary as the local BA is applied to every
            # patch the backend touches; the CUDA kernel adds it independently
            # of eff_impl (fastba/ba_cuda.cu, metric_depth_residuals_and_hessian
            # launch guarded only by depth_prior_lambda > 0).
            lmbda = torch.as_tensor([1e-4], device="cuda")
            if t0_override is not None:
                t0 = int(t0_override)
            elif self.cfg.METRIC_GLOBAL_BA_T0 == "all":
                t0 = 1
            else:  # 'active_min': DPV-SLAM faithful
                t0 = self.pg.ii.min().item()
            prior_lambda = (
                float(self.cfg.METRIC_DEPTH_PRIOR_WEIGHT)
                if self.metric_prior_regularization_enabled
                else 0.0
            )
            depth_prior = self.pg.metric_disps_.reshape(-1)
            if prior_lambda > 0.0 and bool(self.cfg.GBA_SCALE_LATENT):
                # Scale latent: refit the zero-mean per-keyframe log-scale latent
                # from the current (globally adjusted) patch depths vs the RAW
                # MoGe prior and hand the global BA the rescaled prior; the
                # weight rows are untouched.
                self._update_scale_latent()
                depth_prior = (
                    self.pg.metric_disps_ * torch.exp(self.pg.metric_logscale_)[:, None]
                ).reshape(-1)
            fastba.BA(self.poses, self.patches, self.intrinsics,
                full_target, full_weight, lmbda, full_ii, full_jj, full_kk,
                t0, self.n, M=self.M, iterations=2, eff_impl=True,
                depth_prior=depth_prior,
                depth_prior_weight=self.pg.metric_weights_.reshape(-1),
                depth_prior_lambda=prior_lambda,
                depth_prior_huber=float(self.cfg.METRIC_DEPTH_PRIOR_HUBER))
            # Counters are incremented only after the CUDA call returned.
            self.global_ba_calls += 1
            self.global_ba_t0_history.append(int(t0))
            self.global_ba_edge_history.append(int(full_ii.numel()))
            if prior_lambda > 0.0:
                self.global_ba_calls_with_metric_prior += 1
                self.metric_prior_ba_calls += 1
            self.ran_global_ba[self.n] = True
            return

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    def _gba_scheduled_now(self):
        """Scheduled global BA every GBA_SCHEDULE_K accepted keyframes,
        independent of proximity loop edges, when stored factors exist."""
        K = int(self.cfg.GBA_SCHEDULE_K)
        if K <= 0 or not self.cfg.METRIC_GLOBAL_BA:
            return False
        if self.n % K != 0 or self.n == self._gba_sched_last_n:
            return False
        if self.pg.ii_inac.numel() == 0:
            return False
        self._gba_sched_last_n = self.n
        return True

    def _parallax_rows(self):
        """parallax_t for keyframes 1..n-1 (shape [n-1], px): mean translation-only
        flow of frame t's patch centres into its predecessor keyframe t-1 at the
        current estimate; shared by the scale latent's parallax gate and the
        keyframe_prior sidecar. Requires n >= 2."""
        n, M = self.n, self.M
        t = torch.arange(1, n, device="cuda")
        kk = (t[:, None] * M + torch.arange(M, device="cuda")[None, :]).reshape(-1)
        ii = self.ix[kk]          # owning frame t of each patch
        jj = ii - 1               # predecessor keyframe
        poses = SE3(self.poses)
        c0 = pops.transform(poses, self.patches, self.intrinsics, ii, ii, kk)
        c1 = pops.transform(poses, self.patches, self.intrinsics, ii, jj, kk, tonly=True)
        flow = (c1 - c0)[:, :, self.P // 2, self.P // 2, :].norm(dim=-1)   # (1, (n-1)*M)
        flow = torch.nan_to_num(flow, nan=1e3, posinf=1e3).clamp(max=1e3)
        return flow.view(n - 1, M).mean(dim=1)                             # frames 1..n-1

    def _update_scale_latent(self):
        """GBA_SCALE_LATENT: solve the per-keyframe log-scale latent s_t (dpvo/scale_latent.py)
        from the current patch inverse depths vs the RAW MoGe prior and store it in
        pg.metric_logscale_[:n]. Called from __run_global_BA only (prior on)."""
        import time as _time
        t_start = _time.time()
        n, M = self.n, self.M
        c = self.P // 2
        d = self.pg.patches_[:n, :, 2, c, c]
        dp = self.pg.metric_disps_[:n]
        valid = (
            (self.pg.metric_weights_[:n] > 0) & (d > 0) & (dp > 0)
            & torch.isfinite(d) & torch.isfinite(dp)
        )
        log_ratio = torch.log(d.clamp_min(1e-6)) - torch.log(dp.clamp_min(1e-6))
        m, count, w = robust_log_scale_offsets(
            log_ratio, valid, huber=0.2,
            min_patches=int(self.cfg.GBA_SCALE_MIN_PATCHES), patches_per_frame=M,
        )
        if n >= 2:
            par = self._parallax_rows()
            parallax = torch.cat([par[:1], par])        # frame 0 shares frame 1
        else:
            parallax = torch.zeros(n, device=m.device)
        s, stats = gba_scale_latent(
            m, w, parallax,
            clamp=float(self.cfg.GBA_SCALE_CLAMP),
            min_parallax_px=float(self.cfg.GBA_SCALE_MIN_PARALLAX_PX),
            lambda_smooth=float(self.cfg.GBA_SCALE_SMOOTH), lambda_anchor=1e-3,
        )
        self.pg.metric_logscale_[:n] = s
        # diagnostics (counters only after the solve returned)
        self.scale_latent_calls += 1
        self.scale_latent_std_hist.append(round(stats["std_s"], 5))
        self.scale_latent_maxabs_hist.append(round(stats["max_abs_s"], 5))
        T = max(int(stats["n_keyframes"]), 1)
        self.scale_latent_gated_hist.append([
            round(stats["n_gated_parallax"] / T, 4), round(stats["n_gated_min_patches"] / T, 4),
            round(stats["n_gated_clamp"] / T, 4)])
        self.scale_latent_wmean_abs_max = max(self.scale_latent_wmean_abs_max, abs(float(stats["weighted_mean_s"])))
        self.scale_latent_clamped_total += int(stats["n_clamped"])
        self.scale_latent_last = {
            "tstamps": [int(t) for t in self.pg.tstamps_[:n]],
            "s": [round(float(x), 5) for x in s.tolist()],
            "m": [round(float(x), 5) for x in m.tolist()],
            "count": [int(x) for x in count.tolist()],
            "parallax_px": [round(float(x), 4) for x in parallax.tolist()],
            "used": [bool(x) for x in stats["used"].tolist()],
            "n_keyframes": int(stats["n_keyframes"]), "n_with_data": int(stats["n_with_data"]),
            "n_used": int(stats["n_used"]), "n_gated_parallax": int(stats["n_gated_parallax"]),
            "n_gated_min_patches": int(stats["n_gated_min_patches"]), "n_gated_clamp": int(stats["n_gated_clamp"]),
            "n_clamped": int(stats["n_clamped"]), "weighted_mean_s": float(stats["weighted_mean_s"]),
            "std_s": float(stats["std_s"]), "max_abs_s": float(stats["max_abs_s"]),
            "global_ba_call_index": int(self.global_ba_calls),
        }
        self.scale_latent_solve_seconds += _time.time() - t_start

    @torch.no_grad()
    def _keyframe_prior_diagnostics(self):
        """Terminate-time sidecar, read-only: per kept keyframe t the Huber
        location m_t of log(BA patch inverse depth) - log(MoGe inverse depth)
        over the frame's patches with a prior (huber 0.2, >= 16 patches), and
        parallax_t. Emitted under metric_prior_diagnostics['keyframe_prior']."""
        n, M = self.n, self.M
        if n < 1:
            return None
        disps = self.pg.patches_[:n, :, 2, self.P // 2, self.P // 2].float()
        prior = self.pg.metric_disps_[:n].float()
        log_ratio = torch.log(disps.clamp_min(1e-6)) - torch.log(prior.clamp_min(1e-6))
        valid = (self.pg.metric_weights_[:n] > 0) & (disps > 0) & (prior > 0)
        m, count, _ = robust_log_scale_offsets(log_ratio, valid, huber=0.2, min_patches=16, patches_per_frame=M)
        parallax = self._parallax_rows() if n >= 2 else None
        return keyframe_scale_summary(m, count, parallax, self.pg.tstamps_[:n], min_patches=16,
                                      warmup_frames=int(self.cfg.METRIC_DEPTH_PRIOR_WARMUP) or 12)

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=self.cfg.MIXED_PRECISION):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight
        if bool(self.cfg.LOOP_REVERSE_EDGES):
            self._tag_weight_stats(weight)   # diagnostics only

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                elif self._gba_scheduled_now():
                    # GBA_SCHEDULE_K: scheduled global BA (replaces this update's
                    # local BA; the global solve includes the active edges).
                    self.__run_global_BA()
                    self.global_ba_scheduled_calls += 1
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    prior_lambda = (
                        float(self.cfg.METRIC_DEPTH_PRIOR_WEIGHT)
                        if self.cfg.METRIC_DEPTH
                        and self.metric_prior_regularization_enabled
                        else 0.0
                    )
                    fastba.BA(self.poses, self.patches, self.intrinsics,
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk,
                        t0, self.n, M=self.M, iterations=2, eff_impl=False,
                        depth_prior=self.pg.metric_disps_.reshape(-1),
                        depth_prior_weight=self.pg.metric_weights_.reshape(-1),
                        depth_prior_lambda=prior_lambda,
                        depth_prior_huber=float(self.cfg.METRIC_DEPTH_PRIOR_HUBER))
                    if prior_lambda > 0.0:
                        # Increment only after the CUDA BA call returns without
                        # raising, so diagnostics prove the unary was executed.
                        self.metric_prior_ba_calls += 1
            except Exception:
                # A silent fallback would make the metric experiment look as
                # though it used the prior while actually running no BA.
                if self.cfg.METRIC_DEPTH:
                    raise
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics, metric_depth=None, metric_mask=None):
        """ track new frame """


        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        image_height, image_width = image.shape[-2:]
        metric_disps = None
        metric_valid = None
        metric_diagnostics = None
        metric_prior = None
        metric_weight = None
        if self.cfg.METRIC_DEPTH:
            if metric_depth is None:
                raise ValueError("METRIC_DEPTH=True requires metric_depth for every frame")
            metric_disps, metric_valid, metric_diagnostics = prepare_metric_disparity(
                metric_depth,
                metric_mask,
                image_hw=(image_height, image_width),
                min_depth=float(self.cfg.METRIC_DEPTH_MIN),
                max_depth=float(self.cfg.METRIC_DEPTH_MAX),
            )
            metric_disps = metric_disps.to(device=image.device)
            metric_valid = metric_valid.to(device=image.device)
            if not metric_diagnostics["has_valid_depth"]:
                raise ValueError(
                    "metric depth has no valid pixels; rejecting the frame "
                    "instead of injecting a 1 m initialization"
                )

        image = 2 * (image[None,None] / 255.0) - 0.5
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    disps=metric_disps,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)

        # Validate the sampled priors before mutating timestamps, poses, graph
        # indices, or diagnostic counters. A rejected frame therefore leaves
        # the tracker in the same state in which it entered this call.
        if self.cfg.METRIC_DEPTH:
            metric_prior, metric_weight = patch_metric_priors(patches, metric_valid)
            if not bool(metric_weight.any()):
                raise ValueError(
                    "metric depth has no valid sampled DPVO patches; rejecting "
                    "the frame instead of injecting an unobserved depth"
                )

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        if self.cfg.METRIC_DEPTH:
            patches[:, :, 2] = metric_prior[None, :, None, None]
            self.pg.metric_disps_[self.n] = metric_prior
            self.pg.metric_weights_[self.n] = metric_weight
            # Scale latent: a new keyframe inherits its predecessor's latent until the
            # next global BA refits it (never read while GBA_SCALE_LATENT is off).
            self.pg.metric_logscale_[self.n] = self.pg.metric_logscale_[max(self.n - 1, 0)]
            self.metric_prior_valid_pixel_fractions.append(
                metric_diagnostics["valid_pixel_fraction"]
            )
            self.metric_prior_valid_patches += int(metric_weight.sum().item())
            self.metric_prior_total_patches += int(metric_weight.numel())
        else:
            # Preserve the official DPVO initialization when no external prior
            # is requested.
            patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
            if self.is_initialized:
                s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
                patches[:,:,2] = s
            self.pg.metric_disps_[self.n].zero_()
            self.pg.metric_weights_[self.n].zero_()
            self.pg.metric_logscale_[self.n].zero_()   # scale latent (GBA_SCALE_LATENT)

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M
        if self.cfg.METRIC_DEPTH:
            self.metric_keyframes_seen += 1
            enable_metric_prior_after_call = (
                not self.metric_prior_regularization_enabled
                and self.metric_keyframes_seen
                >= int(self.cfg.METRIC_DEPTH_PRIOR_WARMUP)
            )
        else:
            enable_metric_prior_after_call = False

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj, tag=TAG_LOOP_FWD)
                    self._count_loop_edges(lii)
                    if bool(self.cfg.LOOP_REVERSE_EDGES):
                        self._append_reverse_loop_edges(lii, ljj)

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()


        # The first warm-up graph is optimized without the unary term. Once
        # the monotonic accepted-keyframe count reaches the threshold, latch
        # regularization on for subsequent updates even if compaction lowers n.
        if enable_metric_prior_after_call:
            self.metric_prior_regularization_enabled = True

    def metric_prior_diagnostics(self):
        total = self.metric_prior_total_patches
        fractions = self.metric_prior_valid_pixel_fractions
        result = {
            "valid_patches": int(self.metric_prior_valid_patches),
            "total_patches": int(total),
            "valid_patch_fraction": (
                float(self.metric_prior_valid_patches / total) if total else 0.0
            ),
            "mean_valid_pixel_fraction": (
                float(np.mean(fractions)) if fractions else 0.0
            ),
            "keyframes_seen": int(self.metric_keyframes_seen),
            "regularization_enabled": bool(
                self.metric_prior_regularization_enabled
            ),
            "successful_ba_calls_with_metric_prior": int(
                self.metric_prior_ba_calls
            ),
            "active_factors": int(self.pg.ii.numel()),
            "inactive_factors": int(self.pg.ii_inac.numel()),
        }
        if self.cfg.METRIC_GLOBAL_BA:
            # Only METRIC_GLOBAL_BA adds these keys; with the flag off
            # the dictionary above is byte-identical to the local-BA mode.
            result.update({
                "metric_global_ba": True,
                "metric_global_ba_t0_mode": str(self.cfg.METRIC_GLOBAL_BA_T0),
                "loop_closure_proximity_edges": bool(self.cfg.LOOP_CLOSURE),
                "classic_loop_closure": bool(self.cfg.CLASSIC_LOOP_CLOSURE),
                "backend_thresh_px": float(self.cfg.BACKEND_THRESH),
                "max_edge_age": int(self.cfg.MAX_EDGE_AGE),
                "global_opt_freq": int(self.cfg.GLOBAL_OPT_FREQ),
                "pmem": int(self.pmem),
                # kept keyframes at the end (self.n); edges_loop needs
                # n > REMOVAL_WINDOW and reduce_edges needs >= 30 kept
                # keyframes of separation, so this bounds what the backend
                # could have done on a short prefix.
                "keyframes_kept_at_terminate": int(self.n),
                "global_ba_calls": int(self.global_ba_calls),
                "global_ba_calls_with_metric_prior": int(
                    self.global_ba_calls_with_metric_prior
                ),
                "global_ba_t0_min": (
                    int(min(self.global_ba_t0_history))
                    if self.global_ba_t0_history else None
                ),
                "global_ba_t0_history": [int(t) for t in self.global_ba_t0_history],
                "global_ba_edge_history": [
                    int(e) for e in self.global_ba_edge_history
                ],
                "global_ba_edges_max": (
                    int(max(self.global_ba_edge_history))
                    if self.global_ba_edge_history else 0
                ),
                "loop_edges_added": int(self.loop_edges_added),
                "loop_edge_batches": int(self.loop_edge_batches),
                "inactive_factors_peak": int(self.inactive_factors_peak),
                "inactive_factors_at_terminate": self.inactive_factors_at_terminate,
                "active_factors_at_terminate": self.active_factors_at_terminate,
                "inactive_factors_released_at_terminate": (
                    self.inactive_factors_at_terminate is not None
                ),
                "normalize_called": False,
            })
        if int(self.cfg.GBA_SCHEDULE_K) > 0:
            # Scheduling group (block name kept for the aggregators); absent when off.
            result["backend_variants"] = {
                "gba_schedule_k": int(self.cfg.GBA_SCHEDULE_K),
                "global_ba_scheduled_calls": int(self.global_ba_scheduled_calls),
            }
        if bool(self.cfg.CULLED_INTERP):
            # Culled-frame interpolation (block name kept for the aggregators); absent when off.
            result["burst_fixes"] = {
                "culled_interp": True,
                "culled_interp_stats": self.culled_interp_stats,
            }
        if str(self.cfg.GBA_REMEASURE_FINAL) != "off":
            # Final re-measurement (block name kept for the aggregators); absent when off.
            peak_mem = None
            try:
                peak_mem = float(torch.cuda.max_memory_allocated() / 2**30)
            except Exception:  # noqa: BLE001
                pass
            result["backend_m"] = {
                "gba_remeasure_final": str(self.cfg.GBA_REMEASURE_FINAL),
                "fmap_store": int(self.cfg.FMAP_STORE),
                "fmap_ring_len": int(self.mem),
                "remeasure_passes": self.remeasure_passes,
                "remeasure_seconds": float(self.remeasure_seconds),
                "remeasure_edge_refreshes_missing_fmap": int(self.remeasure_edges_missing_fmap),
                "peak_gpu_memory_gib": peak_mem,
                # cold-start flags as run
                "remeasure_burnin": int(self.cfg.REMEASURE_BURNIN),
                "remeasure_warm_active": bool(self.cfg.REMEASURE_WARM_ACTIVE),
                "remeasure_carry_state": bool(self.cfg.REMEASURE_CARRY_STATE),
                "remeasure_schedule": list(pass_schedule(str(self.cfg.GBA_REMEASURE_FINAL))),
            }
        if bool(self.cfg.GBA_SCALE_LATENT):
            # Scale latent; absent when off.
            result["scale_latent"] = self._scale_latent_diagnostics()
        if self.cfg.METRIC_GLOBAL_BA and self.keyframe_prior_diag is not None:
            # terminate-time sidecar (computed before the factor release); diagnostics only.
            result["keyframe_prior"] = self.keyframe_prior_diag
        if bool(self.cfg.LOOP_REVERSE_EDGES) or int(self.cfg.KEYFRAME_MAX_CULL) > 0:
            # Graph-structure group; absent when both keys are default.
            tag_means = {TAG_NAMES[t]: (float(self.tag_w_sum[t] / self.tag_w_cnt[t]) if self.tag_w_cnt[t] else None)
                         for t in TAG_NAMES}
            fwd, rev = tag_means["forward_loop"], tag_means["reverse_loop"]
            result["graph_g"] = {
                "loop_reverse_edges": bool(self.cfg.LOOP_REVERSE_EDGES),
                "loop_forward_pairs": int(self.loop_forward_pairs),
                "loop_reverse_pairs": int(self.loop_reverse_pairs),
                "loop_reverse_pairs_skipped_nonresident": int(self.loop_reverse_pairs_skipped),
                "loop_reverse_edges_added": int(self.loop_reverse_edges_added),
                "loop_reverse_batches": int(self.loop_reverse_batches),
                "tag_weight_mean_update": tag_means,           # running mean over every update() call
                "tag_weight_edge_updates": {TAG_NAMES[t]: int(self.tag_w_cnt[t]) for t in TAG_NAMES},
                "reverse_over_forward_weight_ratio": (float(rev / fwd) if (fwd and rev is not None) else None),
                "remeasure_final_weight_by_tag": self.remeasure_final_weight_by_tag,   # M1 last pass, per age class
                **self.hole_guard.diagnostics(),
            }
        return result

    def _scale_latent_diagnostics(self):
        """Per-run summary of the scale latent: flags, number of GBA calls where it was
        applied, std / max|s| (last call and over calls), gate fractions, quartile means,
        the post-clamp weighted mean (0 by construction), and the per-keyframe snapshot of
        the last call (s_t, m_t, count_t, parallax_t keyed by input frame index)."""
        last = self.scale_latent_last or {}
        out = {
            "gba_scale_latent": True,
            "gba_scale_smooth": float(self.cfg.GBA_SCALE_SMOOTH),
            "gba_scale_clamp": float(self.cfg.GBA_SCALE_CLAMP),
            "gba_scale_min_patches": int(self.cfg.GBA_SCALE_MIN_PATCHES),
            "gba_scale_min_parallax_px": float(self.cfg.GBA_SCALE_MIN_PARALLAX_PX),
            "gba_scale_local_use": str(self.cfg.GBA_SCALE_LOCAL_USE),
            "gba_calls_applied": int(self.scale_latent_calls),
            "solve_seconds_total": float(self.scale_latent_solve_seconds),
            "std_s_last": last.get("std_s"),
            "std_s_median_over_calls": (float(np.median(self.scale_latent_std_hist)) if self.scale_latent_std_hist else None),
            "std_s_max_over_calls": (float(max(self.scale_latent_std_hist)) if self.scale_latent_std_hist else None),
            "max_abs_s_last": last.get("max_abs_s"),
            "max_abs_s_over_calls": (float(max(self.scale_latent_maxabs_hist)) if self.scale_latent_maxabs_hist else None),
            "min_s_last": (float(min(last["s"])) if last.get("s") else None),
            "max_s_last": (float(max(last["s"])) if last.get("s") else None),
            "gated_parallax_fraction_last": (last["n_gated_parallax"] / max(last["n_keyframes"], 1)) if last else None,
            "gated_min_patches_fraction_last": (last["n_gated_min_patches"] / max(last["n_keyframes"], 1)) if last else None,
            "gated_clamp_fraction_last": (last["n_gated_clamp"] / max(last["n_keyframes"], 1)) if last else None,
            "used_fraction_last": (last["n_used"] / max(last["n_keyframes"], 1)) if last else None,
            "gated_fractions_per_call": self.scale_latent_gated_hist,      # [parallax, min_patches, clamp] per GBA call
            "clamped_frames_total": int(self.scale_latent_clamped_total),
            "exp_weighted_mean_s_last": (float(np.exp(last["weighted_mean_s"])) if last else None),
            "weighted_mean_s_abs_max_over_calls": float(self.scale_latent_wmean_abs_max),
            "std_s_history": self.scale_latent_std_hist,
        }
        if last.get("s"):
            s = np.asarray(last["s"], dtype=np.float64)
            m = np.asarray(last["m"], dtype=np.float64)
            par = np.asarray(last["parallax_px"], dtype=np.float64)
            used = np.asarray(last["used"], dtype=bool)
            q = max(len(s) // 4, 1)
            m_used = m[used] if used.any() else m
            out.update({
                "mean_s_first_quartile": float(s[:q].mean()),
                "mean_s_last_quartile": float(s[-q:].mean()),
                "m_std_used": float(m_used.std()) if m_used.size > 1 else 0.0,
                "m_median_used": float(np.median(m_used)) if m_used.size else None,
                "m_mean_first_quartile": float(m[:q].mean()),
                "m_mean_last_quartile": float(m[-q:].mean()),
                "frac_parallax_lt_2px": float((par < 2.0).mean()),
                "parallax_median_px": float(np.median(par)),
                # warm-up excursion: mean m_t over the first 12 keyframes minus the median m_t
                "warmup_excursion_log": float(m[:12].mean() - np.median(m)) if len(m) >= 12 else None,
                "per_keyframe": {k: last[k] for k in ("tstamps", "s", "m", "count", "parallax_px", "used")},
            })
        return out
