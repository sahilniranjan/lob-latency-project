import argparse, yaml
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
from joblib import dump
from utils_data import make_features

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
parser.add_argument('--data', default='data/processed/lob.parquet')
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
df = pd.read_parquet(args.data)
X, y = make_features(df, cfg['levels'], cfg['rolling_windows_ms'], cfg['horizon_ms'])

n = len(X); cut = int(n * (1 - cfg['val_split_time']))
Xtr, Xva = X[:cut], X[cut:]
ytr, yva = y[:cut], y[cut:]

clf = LogisticRegression(C=cfg['C'], class_weight=cfg['class_weight'], max_iter=200, n_jobs=1)
clf.fit(Xtr, ytr)

pred = clf.predict(Xva)
print('F1(macro):', f1_score(yva, pred, average='macro'))
print('MCC:', matthews_corrcoef(yva, pred))
print(classification_report(yva, pred))
Path('outputs').mkdir(exist_ok=True)
dump(clf, cfg['model_pkl'])
print('[*] Saved model to', cfg['model_pkl'])
