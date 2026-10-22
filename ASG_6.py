print("Developed by: Nikunj Tangle")
print("Submitted to: Vaishali Ingale Mam")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# model_selection se train test split
# sklearn.impute se standardimputer
# linear model se linear regression
# preprocessing se standard scaler polynomial features
# metrics se r2 score

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


df=pd.read_csv("housing.csv")

print("No. of rows:", df.shape[0])
print("No. of columns:", df.shape[1])

df.head()


#mising values
print("Number of missing entries:", df.isnull().sum(),"\n")

#duplicated values
print("Number of duplicated entries: ", df.duplicated().sum())


target="median_house_value"

cor=df.corr(numeric_only=True)
cor=cor[target]
cor=cor.drop(target)
cor=cor.abs()
cor=cor.sort_values(ascending=False)

selected_values=list(cor.index[:2])

print(cor,"\n")
print(selected_values)


#evaluation metrics
def calculate_mse(y_true,y_pred):
    y_true= np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    return np.mean((y_true-y_pred)**2)
    
    
def calculate_mae(y_true,y_pred):
    y_true= np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    return np.mean(np.abs(y_true-y_pred))
    
def calculate_r2_score(y_true,y_pred):
    y_true= np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-np.mean(y_true))**2)
    
    return 1- (ss_res/ss_tot)


split_results=[]

#loading the model
linear_model=LinearRegression()

for train_size in np.round(np.arange(0.1,1.0,0.1),1):
    X_train,X_test,Y_train,Y_test=train_test_split(
        
        df[[selected_values[0]]],
        df["median_house_value"],
        train_size=train_size,
        random_state=42
    )
    
    linear_model.fit(X_train,Y_train)
    
    train_pred=linear_model.predict(X_train)
    test_pred=linear_model.predict(X_test)
    
    split_results.append(
        {
            "Training_Size": train_size,
            "Training_Percentage": int(train_size*100),
            "Testing_Size": round(1-train_size, 1),
            "Testing_Percentage": int(round((1-train_size)*100)),
            "Train_MSE": calculate_mse(Y_train,train_pred),
            "Test_MSE": calculate_mse(Y_test,test_pred),
            "Train_MAE": calculate_mae(Y_train,train_pred),
            "Test_MAE": calculate_mae(Y_test,test_pred),
            "Train_r2": calculate_r2_score(Y_train,train_pred),
            "Test_r2": calculate_r2_score(Y_test,test_pred)
        }
    )
    
split_results_df=pd.DataFrame(split_results)
    
display(
    split_results_df[[
        "Training_Percentage",
        "Testing_Percentage",
        "Train_MSE",
        "Test_MSE"
    ]]
)


selected_split=split_results_df.loc[split_results_df["Test_MSE"].idxmin()]
print("selected_split",
      f"{int(selected_split["Training_Percentage"])}% Training",
      f"{int(selected_split["Testing_Percentage"])}% Testing")


plt.plot(split_results_df["Training_Percentage"],split_results_df["Train_MSE"],marker="o",label="Train MSE")
plt.plot(split_results_df["Training_Percentage"],split_results_df["Test_MSE"],marker="o",label="Test MSE")


best_test_size=selected_split["Testing_Size"]

X_train,X_test,Y_train,Y_test=train_test_split(
    df[[selected_values[0]]],
    df["median_house_value"],
    test_size=best_test_size,
    random_state=42
)

linear_model.fit(X_train,Y_train)

linear_train_pred=linear_model.predict(X_train)
linear_test_pred=linear_model.predict(X_test)

#equation
print("Median_House_Value=", f"{linear_model.coef_[0]:.4f}*{selected_values[0]}+{linear_model.intercept_:.4f}")

print("Train MSE:", calculate_mse(Y_train,linear_train_pred))
print("Test MSE:", calculate_mse(Y_test,linear_test_pred))


plt.scatter(X_train[selected_values[0]],Y_train,s=5,label="Actual Train")
plt.scatter(X_test[selected_values[0]],Y_test,s=5,label="Actual test")
plt.plot(X_test[selected_values[0]],linear_test_pred,color="red",label="Prediction")

plt.legend()
plt.show()


polynomial_model=Pipeline([
    ("poly",PolynomialFeatures(degree=2,include_bias=False)),
    ("linear", LinearRegression())
    
])

x_poly_train=df.loc[X_train.index,selected_values]
x_poly_test=df.loc[X_test.index,selected_values]

y_poly_train=df.loc[X_train.index,"median_house_value"]
y_poly_test=df.loc[X_test.index,"median_house_value"]

polynomial_model.fit(x_poly_train,y_poly_train)

poly_train_mse=polynomial_model.predict(x_poly_train)
poly_train_mae=polynomial_model.predict(x_poly_test)


plane_model=LinearRegression()
plane_model.fit(x_poly_train,y_poly_train)

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(x_poly_train[selected_values[0]], x_poly_train[selected_values[1]], y_poly_train, s=3)
ax.scatter(x_poly_test[selected_values[0]], x_poly_test[selected_values[1]], y_poly_test, s=3)

x1 = np.linspace(df[selected_values[0]].min(), df[selected_values[0]].max(), 20)
x2 = np.linspace(df[selected_values[1]].min(), df[selected_values[1]].max(), 20)
X1, X2 = np.meshgrid(x1, x2)

grid = pd.DataFrame({selected_values[0]: X1.ravel(), selected_values[1]: X2.ravel()})
Z = plane_model.predict(grid).reshape(X1.shape)

ax.plot_surface(X1, X2, Z)

plt.show()
