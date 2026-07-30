import csv
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _fixture(tmp_path):
    rows = [("vector", "layer", "rank", "feature", "coef", "cumulative_explained")]
    for name in ("jtw_full_matched_mean", "jtw_stem_mean"):
        for r in range(5):
            rows.append((name, 11, r, 100 + r, "0.5", f"{0.2 * (r + 1):.3f}"))
    with open(tmp_path / "sae_features_ds.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    json.dump({"k": 5, "width": "16k", "layers": {"src": 11, "tgt": 20},
               "supports": {"jtw_full_matched_mean": [100, 101],
                            "jtw_stem_mean": [101, 102]},
               "jaccard": {"jtw_full_matched_mean|jtw_stem_mean": 0.333}},
              open(tmp_path / "sae_overlap_ds.json", "w"))


def test_figures_are_written(tmp_path, monkeypatch):
    import viz_sae
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)
    viz_sae.fig_explained("ds")
    viz_sae.fig_overlap("ds")
    assert (tmp_path / "plot_sae_explained_ds.png").exists()
    assert (tmp_path / "plot_sae_overlap_ds.png").exists()


def test_figures_skip_when_inputs_missing(tmp_path, monkeypatch, capsys):
    import viz_sae
    monkeypatch.chdir(tmp_path)
    viz_sae.fig_explained("missing")
    viz_sae.fig_overlap("missing")
    out = capsys.readouterr().out
    assert "skip fig_explained: sae_features_missing.csv missing" in out
    assert "skip fig_overlap: sae_overlap_missing.json missing" in out
    assert not (tmp_path / "plot_sae_explained_missing.png").exists()
    assert not (tmp_path / "plot_sae_overlap_missing.png").exists()
