print("Developed by: Nikunj Tangle")
print("Submitted to: Vaishali Ingale Mam")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.naive_bayes import GaussianNB


df=pd.read_csv("Admission_Predict.csv")

print("No. of rows:", df.shape[0])
print("No. of columns:", df.shape[1])

df.head()


#missing values
print("No. of missing values:", df.isnull().sum())

print("No. of duplicated values:", df.duplicated().sum())


# Data cleaning
df.columns = df.columns.str.strip()
df = df.dropna()

threshold = df["Chance of Admit"].quantile(0.75)
df["Admit"] = df["Chance of Admit"]>threshold

print("75th percentile cutoff:", threshold)
print(df["Admit"].value_counts())

plt.hist(df["Chance of Admit"], bins=20, edgecolor="black")
plt.axvline(threshold, color="red", linestyle="--")
plt.show()


x=df.drop(columns=["Chance of Admit","Admit"])
y=df["Admit"]

print("X Features:",x.shape)


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
    plt.title("COnfusion Matrix-"+ name)
    plt.show()
    
    result={
        "Model":name,
        "Accuracy": accuracy_score(Y_test,Y_pred),
        "Precision": precision_score(Y_test,Y_pred),
        "Recall": recall_score(Y_test,Y_pred),
        "F1": f1_score(Y_test,Y_pred)
    }
    
    print(name)
    for metric,value in list(result.items())[1:]:
        print(f"{metric:10s}: {value:.4f}")
    return result


models={
    "Decision Tree(Entropy)": DecisionTreeClassifier(criterion="entropy",max_depth=4,random_state=42),
    "Decision Tree(Gini)": DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=42),
    "Naive Bayes": GaussianNB()
}

results=[]
for name, model in models.items():
    results.append(evaluate(name,model,X_train,X_test,Y_train,Y_test))


for name in ["Decision Tree(Entropy)","Decision Tree(Gini)"]:
    plt.figure(figsize=(20,8))
    plot_tree(models[name],feature_names=x.columns,class_names=["False", "True"],filled=True)
    plt.title(name)
    plt.show()
    
    
    
    
importance=pd.DataFrame({
    "Entropy": models["Decision Tree(Entropy)"].feature_importances_,
    "Gini": models["Decision Tree(Gini)"].feature_importances_
},index=x.columns)
importance.plot(kind="barh")
plt.title("Importance")
plt.show()


print("Prior Probablities:")
print(Y_train.value_counts(normalize=True).round(3))

X_cat=X_train.copy()
for col in X_cat.columns:
    if X_cat[col].nunique()>5:
        X_cat[col]=pd.qcut(X_cat[col],3,labels=["Low", "Medium", "High"])
        
        
likelihood = pd.concat(
    [pd.crosstab(X_cat[col], Y_train, normalize="columns") for col in X_cat.columns],
    keys=X_cat.columns, names=["Feature", "Value"])
likelihood.columns = ["P(x | No)", "P(x | Yes)"]
likelihood.round(3)


results_df=pd.DataFrame(results).set_index("Model")
print(results_df.round(4))

results_df.plot(kind="bar",subplots=True,layout=(2,2),figsize=(12,7),rot=0,legend=False)
plt.tight_layout()
plt.show()


applicant = pd.DataFrame([[325, 112, 4, 4.0, 4.5, 9.1, 1]], columns=x.columns)

for name,model in models.items():
    print(name,"->", model.predict(applicant)[0])
