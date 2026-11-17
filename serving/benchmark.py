import time, statistics, json
import pandas as pd, joblib
from src.data_pipeline import load_data, split_data
bundle=joblib.load("artifacts/churn_pipeline.joblib")
X,y=load_data(); Xtr,Xv,Xte,ytr,yv,yte=split_data(X,y)
sample=Xte.iloc[[0]]
lat=[]
for _ in range(300):
    t=time.perf_counter()
    bundle["pipeline"].predict_proba(sample)
    lat.append((time.perf_counter()-t)*1000)
lat.sort()
def pct(p): return lat[min(int(p*len(lat)),len(lat)-1)]
print(json.dumps({"requests":len(lat),"p50_ms":statistics.median(lat),"p95_ms":pct(.95),"p99_ms":pct(.99)},indent=2))
