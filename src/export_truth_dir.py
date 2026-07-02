"""
export_truth_dir.py — save the supervised truth directions to a small file for cluster use.

steer_supervised.py runs on the GH200 (generation) but the supervised directions need the
activations (big, laptop-only) + scikit-learn. So compute them HERE (laptop) and save a tiny
2304-float-per-direction file the cluster can load.

    .venv/bin/python export_truth_dir.py --dataset cities
Produces: truth_dir_<ds>.npz  (mean_diff, grad, layer)  — rsync this to the cluster.
"""

import argparse
import numpy as np
import funnel_utils as fu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--layer", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    layer = fu.resolve_layer(ds, args.layer)
    X, y = fu.load_acts(ds, layer)
    mean_diff = fu.mean_diff_dir(X, y)
    grad = fu.grad_dir(X, y)
    out = f"truth_dir_{ds}.npz"
    np.savez(out, mean_diff=mean_diff.astype(np.float32),
             grad=grad.astype(np.float32), layer=np.array(layer))
    print(f"Saved {out}: mean_diff & grad unit vectors at layer {layer} (d={X.shape[1]})")
    print(f"  cos(mean_diff, grad) = {float(mean_diff @ grad):+.3f}")
    print(f"rsync this to the cluster, then run steer_supervised.py --dataset {ds}")


if __name__ == "__main__":
    main()
