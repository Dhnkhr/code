print("Developed by: Nikunj Tangle")
print("Submitted to: Vaishali Ingale Mam")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.spatial.distance import cdist


df= pd.read_csv("Iris.csv")
df.head()


species=df["Species"]
X=df.drop(columns=["Id", "Species"])

scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)


X_train,X_test=train_test_split(
    X_scaled,
    test_size=0.2,
    random_state=42
)


def wcss(X,labels):
    total=0
    for c in np.unique(labels):
        pts=X[labels==c]
        ctr=pts.mean(axis=0)
        total+=np.sum(pts-ctr)**2
    return total

def gini(labels,species):
    res=0
    for c in np.unique(labels):
        p=X[species==c].value_counts(normalize=True)
        g=1-np.sum(p**2)
        res+=g*np.sum(labels==0)/len(labels)
        
    return res

def dunn(X,labels):
    clusters=[X[labels==c] for c in np.unique(labels)]
    gap=min(cdist(a,b).min() for i,a in enumerate(clusters) for b in clusters[i+1:])
    size=max(cdist(a,a) for a in clusters)
    return gap/size


wcss_values=[]
for k in range(1,11):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    km.fit(X_train)
    wcss_values.append(km.inertia_)
    
    
plt.plot(range(1,11), wcss_values,marker="o")

plt.legend()
plt.show()


kmeans=KMeans(n_clusters=3,random_state=42, n_init=10)
kmeans.fit(X_train)

train_labels=kmeans.predict(X_train)
test_labels=kmeans.predict(X_test)

print("Test points assigned to clusters:", test_labels)


plt.scatter(X_train[:,2],X_train[:,3],c=train_labels,label="Train")
plt.scatter(X_test[:,2],X_test[:,3], c=test_labels,marker="x",label="Test")
plt.scatter(kmeans.cluster_centers_[:,2], kmeans.cluster_centers_[:,3],c="red", marker="*", s=300, label="Centres")


km_labels=kmeans.predict(X_scaled)

print(pd.crosstab(species,km_labels,colnames=["K means cluster"]))


Z=linkage(X_scaled,method="ward")
plt.figure(figsize=(10,5))

dendrogram(Z,truncate_mode="lastp",p=20)
plt.axhline(y=10,color="red",linestyle="--")


hc=AgglomerativeClustering(n_clusters=3,linkage="ward")
hc_labels=hc.fit_predict(X_scaled)

plt.scatter(X_scaled[:,2],X_scaled[:,3],c=hc_labels)


results = []
for name, labels in [("K-Means", km_labels), ("Hierarchical", hc_labels)]:
    results.append({
        "Model": name,
        "WCSS": wcss(X_scaled, labels),
        "Gini": gini(labels, species),
        "Dunn": dunn(X_scaled, labels),
    })

results_df = pd.DataFrame(results).set_index("Model")
print(results_df.round(4))

ax = results_df.plot(kind="bar", subplots=True, layout=(1, 3), figsize=(12, 4), rot=0, legend=False)
for a in ax.flatten():
    for bars in a.containers:
        a.bar_label(bars, fmt="%.3f")
plt.tight_layout()
plt.show()
