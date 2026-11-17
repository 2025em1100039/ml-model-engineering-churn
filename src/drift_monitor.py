import numpy as np
import pandas as pd

def psi(expected, actual, bins=10):
    # Numeric PSI using training quantile bins; small epsilon prevents log(0).
    edges=np.unique(np.quantile(expected.dropna(),np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    e=np.histogram(expected.dropna(),bins=edges)[0]/max(expected.notna().sum(),1)
    a=np.histogram(actual.dropna(),bins=edges)[0]/max(actual.notna().sum(),1)
    e=np.clip(e,1e-6,None); a=np.clip(a,1e-6,None)
    return float(np.sum((a-e)*np.log(a/e)))

if __name__=="__main__":
    print("Import psi(expected_training_series, recent_production_series) for numeric feature drift.")
    print("Illustrative alert policy: PSI >= 0.20 investigate; PSI >= 0.25 trigger review/retraining assessment.")
