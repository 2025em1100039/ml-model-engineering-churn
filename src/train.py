from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score, classification_report)
from src.data_pipeline import load_data, split_data

ART=Path("artifacts"); OUT=Path("outputs")
ART.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

X,y=load_data()
Xtr,Xv,Xte,ytr,yv,yte=split_data(X,y)
num_cols=Xtr.select_dtypes(include=["number"]).columns.tolist()
cat_cols=Xtr.select_dtypes(exclude=["number"]).columns.tolist()
pre=ColumnTransformer([
 ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num_cols),
 ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat_cols)
])
baseline=Pipeline([("prep",pre),("model",DummyClassifier(strategy="prior"))])
model=Pipeline([("prep",pre),("model",LogisticRegression(max_iter=2000,class_weight="balanced",solver="liblinear",random_state=42))])
baseline.fit(Xtr,ytr); model.fit(Xtr,ytr)
# Choose operating threshold using validation F1; test set is untouched until final evaluation.
val_prob=model.predict_proba(Xv)[:,1]
prec,rec,thr=precision_recall_curve(yv,val_prob)
f1s=2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
best_idx=int(np.nanargmax(f1s)); threshold=float(thr[best_idx])
test_prob=model.predict_proba(Xte)[:,1]
test_pred=(test_prob>=threshold).astype(int)
base_prob=baseline.predict_proba(Xte)[:,1]
base_pred=(base_prob>=0.5).astype(int)

def metrics(ytrue, prob, pred):
 return {"accuracy":float(accuracy_score(ytrue,pred)),
 "precision":float(precision_score(ytrue,pred,zero_division=0)),
 "recall":float(recall_score(ytrue,pred,zero_division=0)),
 "f1":float(f1_score(ytrue,pred,zero_division=0)),
 "roc_auc":float(roc_auc_score(ytrue,prob)),
 "pr_auc":float(average_precision_score(ytrue,prob)),
 "confusion_matrix":confusion_matrix(ytrue,pred).tolist()}
results={"dataset_rows_after_cleaning":int(len(X)),"train_rows":len(Xtr),"validation_rows":len(Xv),"test_rows":len(Xte),
 "churn_rate":float(y.mean()),"threshold_selected_on_validation":threshold,
 "baseline":metrics(yte,base_prob,base_pred),"logistic_regression":metrics(yte,test_prob,test_pred)}
joblib.dump({"pipeline":model,"threshold":threshold,"features":X.columns.tolist()},ART/"churn_pipeline.joblib")
pd.DataFrame({"actual":yte.values,"probability":test_prob,"prediction":test_pred}).to_csv(OUT/"test_predictions.csv",index=False)
(OUT/"metrics.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
