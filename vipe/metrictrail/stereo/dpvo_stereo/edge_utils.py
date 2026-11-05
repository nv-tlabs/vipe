# SPDX-FileCopyrightText: Copyright (c) 2022 Princeton Vision & Learning Lab
# SPDX-License-Identifier: MIT
#
# Source: https://github.com/princeton-vl/DPVO
# Edge-selection helpers retained unchanged from the MIT DPVO code.
# See the inherited DPVO LICENSE and THIRD_PARTY_LICENSES.md.

"""Select and suppress nearby patch-graph edges."""

from vipe.metrictrail.dpvo_deps import numba as nb  # [vipe] was: import numba as nb
import numpy as np


@nb.njit(cache=True)
def _format(es):
    return np.asarray(es, dtype=np.int64).reshape((-1, 2))[1:]


@nb.njit(cache=True)
def reduce_edges(flow_mag, ii, jj, max_num_edges, nms):
    es = [(-1, -1)]

    if ii.size == 0:
        return _format(es)

    Ni, Nj = (ii.max()+1), (jj.max()+1)
    ignore_lookup = np.zeros((Ni, Nj), dtype=nb.bool_)

    idxs = np.argsort(flow_mag)
    for idx in idxs: # edge index

        if len(es) > max_num_edges:
            break

        i = ii[idx]
        j = jj[idx]
        mag = flow_mag[idx]

        if ((j - i) < 30):
            continue

        if mag >= 1000: # i.e., inf
            continue

        if ignore_lookup[i, j]:
            continue

        es.append((i, j))

        for di in range(-nms, nms+1):
            i1 = i + di

            if 0 <= i1 < Ni:
                ignore_lookup[i1, j] = True

    return _format(es)
