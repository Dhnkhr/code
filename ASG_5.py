student_name='\u0906\u092F\u0941\u0937\u0020\u0915\u0941\u092E\u093E\u0930'
print("STUDENT NAME: ",student_name)


submited_to='\u0935\u0948\u0936\u093E\u0932\u0940\u0020\u0907\u0902\u0917\u0933\u0947'
print("SUBMITED TO: ",submited_to)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df=pd.read_csv("Titanic-Dataset.csv")


## Printing number of samples and features in our dataset.
print("Number of samples: ",df.shape[0])
print("Number of features: ",df.shape[1])


## PRINTING TOP 10 SAMPLE OF OUR DATASET.
df.head(10)


## Printing feature name alongwith its total values present and its data type
for col in df.columns:
    print(f'feature Name = "{col}",' , f'Total Values Present = {df[col].count()},',  f'Data Type = {df[col].dtype}')


## there is no use of "Name" and "PassengerId". therefore dropping these columns.
df.drop(columns=["Name","PassengerId"],inplace=True)


df.describe()


for col in df.columns:
    print(f"Column Name      : {col}")
    print(f"Missing Values   : {df[col].isnull().sum()}")
    print("-" * 35)


# df.isnull().sum()


# 'Cabin' has too many missing values, so it is removed
df.drop(columns=["Cabin"],inplace=True)


# Fill missing ages with the median since it is less affected by outliers
df["Age"] = df["Age"].fillna(df["Age"].median())


# Fill missing values in 'Embarked' with the most frequent category (mode)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


encoder = LabelEncoder()


df["Sex"] = encoder.fit_transform(df["Sex"])
df.head()


## CHECKING HOW MANY UNIQUE VALUES ARE PRESENT IN "EMBARKED" COLUMN
df["Embarked"].unique()


# One-Hot Encode the 'Embarked' column
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)


df.head()


sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex')
plt.title('Survival Rate by Passenger Class & Sex')
plt.xlabel('Passenger Class (1st, 2nd, 3rd)')
plt.ylabel('Survival Rate')
plt.show()


sns.boxplot(x=df["Age"])


sns.boxplot(x=df["Fare"])


plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Number of peoples")


scaler = MinMaxScaler()


cols_to_scale = ["Age", "Fare"]
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


df.head(10)


X = df.drop(columns=['Survived'])

y = df['Survived']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


## Printing number of samples and features in our TRAINING dataset.
print("Number of samples in training dataset: ",X_train.shape[0])
print("Number of features in training dataset: ",X_train.shape[1])
print("Number of samples in testing dataset: ",X_test.shape[0])
print("Number of features in testing dataset: ",X_test.shape[1])


X_train.head()
