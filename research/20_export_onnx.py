import argparse
from joblib import load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--onnx', required=True)
args = parser.parse_args()

clf = load(args.model_path)
input_dim = int(clf.coef_.shape[1])
onnx_model = convert_sklearn(clf, initial_types=[('input', FloatTensorType([None, input_dim]))])
with open(args.onnx, 'wb') as f:
    f.write(onnx_model.SerializeToString())
print('[*] Exported ONNX to', args.onnx)
