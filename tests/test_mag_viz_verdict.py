import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mag.viz_verdict import load_rows, margin_stats, accuracy, plot_margin, plot_accuracy


def _write_toy_csv(tmp_path):
    rows = [("direction", "tau", "statement", "label", "p_yes", "p_no", "margin")]
    for d in ("sup_mean_diff", "random_unit"):
        for tau in (-1.0, 0.0, 1.0):
            # true statements: margin follows tau; false statements: margin fixed positive
            rows.append((d, tau, "s_true", 1, 0.8, 0.1, 0.5 * tau))
            rows.append((d, tau, "s_false", 0, 0.7, 0.2, 0.4))
    with open(tmp_path / "mag_verdict_logits_toy.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_margin_stats_and_accuracy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_toy_csv(tmp_path)
    rows = load_rows("toy")
    ms = margin_stats(rows)
    assert ms[("sup_mean_diff", -1.0, 1)][0] == -0.5      # mean margin, true class
    assert ms[("sup_mean_diff", -1.0, 0)][0] == 0.4
    acc = accuracy(rows)
    # tau=-1: true stmt margin<0 -> predicted 0 (wrong); false stmt margin>0 -> predicted 1 (wrong)
    assert acc[("sup_mean_diff", -1.0)] == 0.0
    # tau=+1: true stmt correct, false stmt wrong
    assert acc[("sup_mean_diff", 1.0)] == 0.5


def test_plots_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_toy_csv(tmp_path)
    plot_margin("toy")
    plot_accuracy("toy")
    assert os.path.exists("plot_mag_verdict_margin_toy.png")
    assert os.path.exists("plot_mag_verdict_acc_toy.png")
