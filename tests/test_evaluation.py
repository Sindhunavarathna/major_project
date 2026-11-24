"""Unit tests for Phase 4: Evaluation module and benchmark exports."""

import os
import pytest

from src.evaluation import evaluate_baseline_vs_our_model


def test_evaluation_runs_and_exports(tmp_path):
    out_dir = str(tmp_path / "eval_out")
    res = evaluate_baseline_vs_our_model(output_dir=out_dir)

    assert isinstance(res, dict)
    assert "Kannada" in res or "Tamil" in res or "Telugu" in res

    # Ensure output files are created
    assert os.path.exists(os.path.join(out_dir, "evaluation_results.csv"))
    assert os.path.exists(os.path.join(out_dir, "evaluation_results.md"))
    assert os.path.exists(os.path.join(out_dir, "evaluation_benchmark_chart.png"))
