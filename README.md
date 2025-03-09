# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. .Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given
datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Harisankar.S 
RegisterNumber:212224240051  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Datatest:
![rt](https://github.com/user-attachments/assets/fe3e8628-d0ea-4256-b229-4e3c9e97fbe8)


Headvalues:
![rty](https://github.com/user-attachments/assets/b162d2a2-35a1-44e7-9de8-126876f24eb3)


Tailvalues:
![po](https://github.com/user-attachments/assets/9bcb6aab-263a-411c-a61c-dff4fafc6650)



X and Y values:
![poi](https://github.com/user-attachments/assets/6d6b179a-6636-4840-bf4f-ff7baa3577cf)


pridication value of x and y:
![p](https://github.com/user-attachments/assets/76f2f496-2181-4309-8f17-a603743413f1)


MSE,MAE and RMSE:
![o](https://github.com/user-attachments/assets/e6a7b06c-128f-4677-98d0-c7d0bdf35470)


Training set:
![op](https://github.com/user-attachments/assets/00b8e526-d409-4d2f-afed-3dd352053b8e)


Texting set:
![jpg](https://github.com/user-attachments/assets/42b61370-33a9-4ed4-b162-2a81301d7d4d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
