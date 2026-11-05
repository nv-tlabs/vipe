"""Trajectory (and point-cloud) evaluation for MetricTrail predictions.

One module per dataset (``kitti``, ``vbr``, ``oxford``, ``lamaria``, ``tum``,
``sevenscenes``, ``wild_slam``) reproduces the exact protocol (frame matching,
Sim(3) alignment, units, averaging) that produced the historical numbers;
``sim3_ate`` is the shared hash-locked Sim(3) trajectory evaluator and
``python -m metrictrail_eval.score`` is the command-line entry point.
``vipe_output`` reads ViPE output directories (``--pred-format vipe``) into the
same prediction files.
"""

__version__ = "0.3.0"
