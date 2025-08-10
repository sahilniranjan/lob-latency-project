from joblib import load
import numpy as np, os

os.makedirs("engine/src/onnx", exist_ok=True)
clf = load("outputs/logreg.pkl")
w = clf.coef_.ravel(); b = float(clf.intercept_.ravel()[0])
with open("engine/src/onnx/linear.txt",'w') as f:
    f.write(f"{b}\n{w.size}\n")
    f.write("\n".join(str(float(x)) for x in w))
print("Wrote engine/src/onnx/linear.txt")
