import pandas as pd
import numpy as np
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, classification_report

df=pd.read_csv("student_dataset_10000_rows.csv").drop_duplicates()

for c in ["Unnamed: 0","id","Id","student_id","Student_ID"]:
    if c in df.columns: df=df.drop(columns=[c])

target="placement_status"
X=df.drop(columns=[target])
y=df[target]

# Numeric target with many unique values = regression; otherwise classification
is_regression=pd.api.types.is_numeric_dtype(y) and y.nunique()>10
problem_type="Regression" if is_regression else "Classification"

num=X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat=X.select_dtypes(include=["object","category","bool"]).columns.tolist()

pre=ColumnTransformer([
 ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y if not is_regression else None)

if is_regression:
    models={
      "Linear Regression":LinearRegression(),
      "Random Forest Regressor":RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1)
    }
else:
    models={
      "Logistic Regression":LogisticRegression(max_iter=1000),
      "Random Forest Classifier":RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
    }

best=None; best_name=None; best_metric=-np.inf
results={}

for name,m in models.items():
    pipe=Pipeline([("preprocessor",pre),("model",m)])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    if is_regression:
        metric=r2_score(yte,pred)
        results[name]={"R2":metric,"MAE":mean_absolute_error(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred))}
    else:
        metric=accuracy_score(yte,pred)
        results[name]={"Accuracy":metric}
        print("\n",name,"Accuracy:",round(metric*100,2),"%")
        print(classification_report(yte,pred))
    if metric>best_metric:
        best_metric=metric; best=pipe; best_name=name

os.makedirs("models",exist_ok=True)
joblib.dump(best,"models/student_prediction_model.pkl")
joblib.dump({"target":target,"problem_type":problem_type,"feature_columns":X.columns.tolist(),"numeric_features":num,"categorical_features":cat,"best_model":best_name,"results":results},"models/metadata.pkl")

print("\nPROJECT COMPLETED!")
print("Target:",target)
print("Problem type:",problem_type)
print("Best model:",best_name)
print("Saved: models/student_prediction_model.pkl")
