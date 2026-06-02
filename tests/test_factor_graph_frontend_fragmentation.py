from __future__ import annotations

import torch

from vipe.slam.components.factor_graph import FactorGraph


class _FakeBuffer:
    def __init__(self, n_frames: int = 5, buffer_size: int = 8) -> None:
        self.device = torch.device("cpu")
        self.height = 8
        self.width = 8
        self.n_views = 1
        self.n_frames = n_frames
        self.poses = torch.zeros(buffer_size, 7)
        self.nets = torch.zeros(buffer_size, 1, 128, 1, 1)
        self.masks = torch.zeros(buffer_size, 1, 1, 1, dtype=torch.bool)

    @property
    def flattened_disps(self) -> torch.Tensor:
        return torch.ones(self.poses.shape[0], 1, 1)

    def expand_edge_multiview(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        cross: bool = True,
        view_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del cross, view_offset
        view = torch.zeros_like(ii)
        return ii, view, ii, jj, view, jj

    def reproject_dense_disp(self, ii: torch.Tensor, jj: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.zeros(ii.shape[0], 1, 1, 2), None

    def frame_distance_dense_disp(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        beta: float = 0.3,
        bidirectional: bool = True,
        view_offset: int = 0,
    ) -> torch.Tensor:
        del beta, bidirectional, view_offset
        distance = (ii - jj).abs().float() + 0.05 * ii.float() + 0.01 * jj.float()
        return distance.view(-1, 1)


def _make_graph(n_frames: int = 5) -> FactorGraph:
    return FactorGraph(
        net=None,  # type: ignore[arg-type]
        buffer=_FakeBuffer(n_frames=n_frames),
        device=torch.device("cpu"),
        max_factors=48,
        incremental=False,
        cross_view=False,
    )


def _reference_proximity_edges(
    *,
    n_frames: int,
    t0: int,
    t1: int,
    rad: int,
    nms: int,
    thresh: float,
    max_factors: int,
    active_edges: list[tuple[int, int]],
    inactive_edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(t0, n_frames) for j in range(t1, n_frames)]
    d = torch.tensor([abs(i - j) + 0.05 * i + 0.01 * j for i, j in pairs], dtype=torch.float)
    width = n_frames - t1

    def suppress(i: int, j: int) -> None:
        if (t0 <= i < n_frames) and (t1 <= j < n_frames):
            d[(i - t0) * width + (j - t1)] = torch.inf

    def suppress_nms(i: int, j: int) -> None:
        radius = max(min(abs(i - j) - 2, nms), 0)
        for di in range(-nms, nms + 1):
            for dj in range(-nms, nms + 1):
                if abs(di) + abs(dj) <= radius:
                    suppress(i + di, j + dj)

    for edge in active_edges + inactive_edges:
        suppress_nms(*edge)

    for idx, (i, j) in enumerate(pairs):
        if i - rad < j or d[idx] > thresh:
            d[idx] = torch.inf

    selected_edges: list[tuple[int, int]] = []
    for i in range(t0, n_frames):
        for j in range(max(i - rad - 1, 0), i):
            selected_edges.append((i, j))
            selected_edges.append((j, i))
            suppress(i, j)

    for idx in torch.argsort(d):
        if d[idx].item() > thresh:
            continue
        if len(selected_edges) > max_factors:
            break
        edge = pairs[int(idx.item())]
        selected_edges.append(edge)
        selected_edges.append((edge[1], edge[0]))
        suppress_nms(*edge)

    existing_edges = set(active_edges + inactive_edges)
    return [edge for edge in selected_edges if edge not in existing_edges]


def test_filter_repeated_edges_removes_active_and_inactive_edges_but_preserves_candidate_duplicates() -> None:
    graph = _make_graph(n_frames=6)
    graph.ii = torch.tensor([2, 4])
    graph.jj = torch.tensor([0, 1])
    graph.ii_inac = torch.tensor([3, 1])
    graph.jj_inac = torch.tensor([0, 0])

    ii = torch.tensor([2, 3, 5, 5, 4])
    jj = torch.tensor([0, 0, 2, 2, 1])

    filtered_ii, filtered_jj = graph._FactorGraph__filter_repeated_edges(ii, jj)

    assert list(zip(filtered_ii.tolist(), filtered_jj.tolist())) == [(5, 2), (5, 2)]


def test_add_proximity_factors_matches_reference_selection_with_existing_edges() -> None:
    graph = _make_graph(n_frames=5)
    active_edges = [(4, 1), (1, 0)]
    inactive_edges = [(3, 0)]
    graph.ii = torch.tensor([edge[0] for edge in active_edges])
    graph.jj = torch.tensor([edge[1] for edge in active_edges])
    graph.age = torch.zeros(len(active_edges), dtype=torch.long)
    graph.ii_inac = torch.tensor([edge[0] for edge in inactive_edges])
    graph.jj_inac = torch.tensor([edge[1] for edge in inactive_edges])

    graph.add_proximity_factors(t0=2, t1=0, rad=1, nms=1, beta=0.3, thresh=6.0, remove=False)

    expected_new_edges = _reference_proximity_edges(
        n_frames=5,
        t0=2,
        t1=0,
        rad=1,
        nms=1,
        thresh=6.0,
        max_factors=48,
        active_edges=active_edges,
        inactive_edges=inactive_edges,
    )
    expected_edges = active_edges + expected_new_edges

    assert list(zip(graph.ii.tolist(), graph.jj.tolist())) == expected_edges
