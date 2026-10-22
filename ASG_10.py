print("Developed by: Nikunj Tangle")
print("Submitted to: Vaishali Ingale Mam")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


df=pd.read_csv("Wine.csv")
df.head()


scaler=StandardScaler()
X_scaled=scaler.fit_transform(df.drop("Customer_Segment",axis=1))


x=X_scaled
y=df["Customer_Segment"]


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


results=[]
results.append(evaluate("KNN Before PCA",KNeighborsClassifier(n_neighbors=5),X_train,X_test,Y_train,Y_test))


pca_full=PCA().fit(X_scaled)
cumulative=np.cumsum(pca_full.explained_variance_ratio_)

plt.bar(range(1,14),pca_full.explained_variance_ratio_,label="Individual")
plt.step(range(1,14),cumulative,label="Cumulative")

plt.legend()
plt.show()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Features before PCA:", X_scaled.shape[1])
print("Features after PCA :", X_pca.shape[1])
print("Variance kept by PC1, PC2:", pca.explained_variance_ratio_.round(3))
print("Total variance kept:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

plt.scatter(X_pca[:,0],X_pca[:,1],c=y)
plt.show()


X=X_pca
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
results.append(evaluate("KNN After PCA",KNeighborsClassifier(n_neighbors=5),X_train,X_test,Y_train,Y_test))


results_df=pd.DataFrame(results).set_index("Model")

print(results_df.round(4))

ax=results_df.plot(kind="bar",rot=0,legend=True)

for bars in ax.containers:
    ax.bar_label(bars,fmt="%.2f",fontsize="10")
plt.tight_layout()
plt.show()
