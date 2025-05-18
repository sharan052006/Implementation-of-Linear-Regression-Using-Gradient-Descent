# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 
## Program:

Program to implement the linear regression using gradient descent.

Developed by: Sharan.I

RegisterNumber: 212224040308
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)

print(X1_Scaled)

theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```
## Output:
Head:

![{7DADA87B-4D08-4163-8830-BC4A976477BE}](https://github.com/user-attachments/assets/f61785a9-9726-401b-945b-7096120e2269)

Value of X:

![{CD89220A-C4BF-49D6-A44C-30FA1D7FC22A}](https://github.com/user-attachments/assets/e5a26bff-c5c2-4882-b209-adadd138db90)

Value of X1_Scaled:

![{D5942F72-BE92-42B9-957C-E7C95E042AD2}](https://github.com/user-attachments/assets/2da135d7-dfbd-4866-a3b6-68422982f8f2)

Predicted Value:

![{2048CCBE-5423-4F59-9D28-5FA262BCC668}](https://github.com/user-attachments/assets/1a4bca30-e9b2-494a-8bda-4da78b8f38db)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
