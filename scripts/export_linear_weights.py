"""
Export trained pipeline (StandardScaler + LogisticRegression) weights
into a lightweight text format the C++ engine can load.

Handles both:
  - Pipeline(scaler, logreg)
  - Raw LogisticRegression

For multi-class models (classes = {-1, 0, +1}), exports the class-+1
row so the C++ sigmoid produces an approximate P(up) signal.

Output:  engine/src/onnx/linear.txt
  Line 1: bias_combined
  Line 2: n_features
  Lines 3+: weight_combined[i]  (one per line)

When a scaler is present the combined weight is:
    w_combined[i] = w[i] / scale[i]
    b_combined    = b - sum(w[i] * mean[i] / scale[i])
so the C++ engine can do a simple dot product with raw features.
"""
from joblib import load
import numpy as np, os
from sklearn.pipeline import Pipeline

os.makedirs("engine/src/onnx", exist_ok=True)
clf = load("outputs/logreg.pkl")

if isinstance(clf, Pipeline):
    scaler = clf.named_steps.get("scaler", None)
    logreg = clf.named_steps.get("logreg", clf[-1])
else:
    scaler = None
    logreg = clf

# Handle multi-class: pick the row for class +1
classes = list(logreg.classes_)
if logreg.coef_.ndim == 2 and logreg.coef_.shape[0] > 1:
    # Multi-class: use the +1 class row
    up_idx = classes.index(1) if 1 in classes else len(classes) - 1
    w = logreg.coef_[up_idx]
    b = float(logreg.intercept_[up_idx])
    print(f"Multi-class model ({classes}) — exporting class +1 (idx={up_idx}) weights")
else:
    w = logreg.coef_.ravel()
    b = float(logreg.intercept_.ravel()[0])

# Bake scaler into weights
if scaler is not None and hasattr(scaler, "scale_"):
    mean = scaler.mean_
    scale = scaler.scale_
    w_comb = w / scale
    b_comb = b - float(np.sum(w * mean / scale))
    print(f"Baked StandardScaler into weights (mean/scale folded in)")
else:
    w_comb, b_comb = w, b

with open("engine/src/onnx/linear.txt", "w") as f:
    f.write(f"{b_comb}\n{w_comb.size}\n")
    f.write("\n".join(str(float(x)) for x in w_comb))
print(f"Wrote engine/src/onnx/linear.txt  ({w_comb.size} weights)")
