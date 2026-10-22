print("Developed by: Nikunj Tangle")
print("Submitted to: Vaishali Ingale Mam")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


columns=["Vehicle Cost","Maintainence Cost","No. of doors", "No. of vehicles", "Lug Boot","Safety","Decision"]
df=pd.read_csv("car_evaluation.csv",header=None, names=columns)

print("No. of rows:", df.shape[0])
print("No. of columns:", df.shape[1])
df.head()


print(df["Decision"].value_counts())

encoder=LabelEncoder()
df["Decision"]=encoder.fit_transform(df["Decision"])


x=pd.get_dummies(df.drop("Decision", axis=1), dtype=int)
y=df["Decision"]


X_train,X_test,Y_train,Y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


def evaluate(name,model,X_train,X_test,Y_train,Y_test):
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    
    cm=confusion_matrix(Y_test,Y_pred)
    ConfusionMatrixDisplay(cm,display_labels=model.classes_).plot()
    
    result={
        "Model":name,
        "Accuracy": accuracy_score(Y_test,Y_pred),
        "Precision": precision_score(Y_test,Y_pred,average="weighted",zero_division=0),
        "Recall":recall_score(Y_test,Y_pred,average="weighted",zero_division=0),
        "F1_score": f1_score(Y_test,Y_pred,average="weighted",zero_division=0)
    }
    
    print(name)
    for metric,value in list(result.items())[1:]:
        print(f"{metric:10s}: {value:.4f}")
    return result


models={
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}
results=[]
for name,model in models.items():
    results.append(evaluate(name,model,X_train,X_test,Y_train,Y_test))


for name,model in models.items():
    score=cross_val_score(model,X_train,Y_train,cv=5)
    print(f"{name}-> Score: {score.mean()}")


param_grids={
    "SVM": {"C":[0.1,1,10],"kernel":["linear","rbf"]},
    "Random Forest": {"n_estimators":[50,100,200],"max_depth":[None,10,20]},
    "AdaBoost": {"n_estimators":[50,100,200],"learning_rate":[0.1,1.0]}
    
    
}

for name, model in models.items():
    grid=GridSearchCV(model,param_grids[name],cv=5,n_jobs=-1)
    grid.fit(X_train,Y_train)
    print(name,"Best Params",grid.best_params_,"Best Score:", grid.best_score_)
    results.append(evaluate(name+"(tuned)",grid.best_estimator_,X_train,X_test,Y_train,Y_test))


results_df=pd.DataFrame(results).set_index("Model")

print(results_df.round(4))

ax=results_df.plot(kind="bar",figsize=(12,7),rot=0)

for bars in ax.containers:
    ax.bar_label(bars,fmt="%.2f",fontsize=8)
    
plt.ylim(0,1.1)
plt.tight_layout()
plt.show()
