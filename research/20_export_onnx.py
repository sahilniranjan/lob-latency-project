"""
Export trained model (Pipeline or raw estimator) to ONNX format.

Requires: pip install skl2onnx  (optional dependency)

Usage:
    python research/20_export_onnx.py --model_path outputs/logreg.pkl \
                                       --onnx engine/src/onnx/model.onnx
"""
import argparse
from joblib import load
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--onnx', required=True)
args = parser.parse_args()

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print("skl2onnx not installed — run: pip install skl2onnx")
    raise SystemExit(1)

clf = load(args.model_path)

# Determine input dimension from the model (Pipeline or raw estimator)
if isinstance(clf, Pipeline):
    logreg = clf.named_steps.get("logreg", clf[-1])
    input_dim = int(logreg.coef_.shape[1])
else:
    input_dim = int(clf.coef_.shape[1])

onnx_model = convert_sklearn(
    clf, initial_types=[('input', FloatTensorType([None, input_dim]))]
)
with open(args.onnx, 'wb') as f:
    f.write(onnx_model.SerializeToString())
print(f'[*] Exported ONNX to {args.onnx}  (input_dim={input_dim})')
