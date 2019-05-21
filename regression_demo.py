from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

#######
# Regression Example
#######
#
# Problem Statement:
#   Given a historic dataset of cold-drink sales in your canteen,
#   create an ML model to predict sales for a given day (in the future).
#
#   Historic Dataset has the following:
#      Average Temperature of the day, Sales in number of bottles sold
#       Holiday(0--> no holiday/1-->holiday)
#       Eve day(0-->normal day/1-->fest day/2-->sports day)
#   Evaluation / Prediction Dataset will have the following:
#      Average Temperature of the day
#

#historic_x = [12, 15, 20, 18, 25, 30, 32, 31, 30, 29, 28, 33] #avg temperature
#historic_y = [ 2,  4,  5,  4,  7,  9,  9, 10,  8,  6,  6, 12] #avg sales

df=[[12,2,1,1],[15,4,0,0],[20,5,0,0],[18,4,1,1],[25,7,1,1],[30,9,0,0],[32,9,0,0],[31,10,1,1],[30,8,0,0],[29,6,0,0],[28,6,0,0],[33,12,1,1],[35,13,0,0],[40,18,1,2],[38,15,0,0],[41,18,0,0],[43,20,1,1],[43,25,0,0],[40,20,1,1],[42,20,1,1]]
new_data=pd.DataFrame(df,columns=['Temp','Sales','Holiday','Event Day'])

X=new_data[['Temp','Holiday','Event Day']]
y=new_data[['Sales']]

X = np.asarray(X).reshape(20, 3)
y = np.asarray(y).reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#historic_x = np.asarray(historic_x).reshape(-1, 1)
#X_train, X_test, Y_train, Y_test = train_test_split(historic_x, historic_y, test_size=0.2, random_state=0)

X_pred = [10, 13, 19, 21, 35]
X_pred = np.asarray(X_pred).reshape(-1, 1)

model_lr = LinearRegression().fit(X_train, Y_train)


eval_output = [int(round(i)) for i in model_lr.predict(X_test)]
pred_output = [int(round(i)) for i in model_lr.predict(X_pred)]

model_mae = round(mean_absolute_error(Y_test, eval_output), 2)

print('\nOutput for evaluation dataset (split from training dataset), MAE = ', model_mae)
eval_xy = pd.DataFrame(X_test, columns=['temp'])
eval_xy['sales'] = eval_output
print(eval_xy.to_string())

print('\nOutput for prediction dataset')
pred_xy = pd.DataFrame(X_pred, columns=['temp'])
pred_xy['sales'] = pred_output
print(pred_xy.to_string())
