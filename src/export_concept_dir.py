"""Export supervised concept directions (mean-diff + gradient) for any labeled contrast.

Generalizes export_truth_dir.py. Writes truth_dir_<concept>.npz (schema unchanged so
steer_supervised.py works as-is). 'concept' is a dataset name in got_datasets/.

    .venv/bin/python export_concept_dir.py --dataset refusal
"""
import argparse
import numpy as np
import funnel_utils as fu


def concept_directions(X, y):
    """Return (mean_diff, grad) unit vectors for a binary-labeled activation matrix."""
    return fu.mean_diff_dir(X, y), fu.grad_dir(X, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--layer", type=int, default=None)
    args = p.parse_args()
    ds = args.dataset
    layer = fu.resolve_layer(ds, args.layer)
    X, y = fu.load_acts(ds, layer)
    mean_diff, grad = concept_directions(X, y)
    out = f"truth_dir_{ds}.npz"
    np.savez(out, mean_diff=mean_diff.astype(np.float32),
             grad=grad.astype(np.float32), layer=np.array(layer))
    print(f"Saved {out}: mean_diff & grad at layer {layer} (d={X.shape[1]}); "
          f"cos={float(mean_diff @ grad):+.3f}")


if __name__ == "__main__":
    main()
