# by Syed Saif Ali Alvi, February 17th, 2019
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the CSV file
houses = pd.read_csv('housingdata.csv')

# Importing the method to split testing and training data under Supervised ML
from sklearn.model_selection import train_test_split

# Importing the Scikit-Learn Linear Regression Model
from sklearn.linear_model import LinearRegression

# Preprocessing data into Feeding and Prediction Data Frames
X = houses[['RM', 'INDUS', 'RAD','CHAS','DIS','STAND','ZN']]
y = houses['MEDV']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.289, random_state = 115)

# Training the Model
lm = LinearRegression()
lm.fit(X_train, y_train)
coeff = lm.coef_
coeff_df = pd.DataFrame()

# Gathering Predictions and Residue
predictions = lm.predict(X_test)
residue = y_test - predictions
acc = lm.score(X_test,y_test)
percentage = (acc*100)
percent= float("{0:.2f}".format(percentage))
print(percent,'%')
label = [' ','Model 1',' ',' ',' ',' ',' ']
percents=[0,percent,0,0,0,0,0]
index = np.arange(len(label))
plt.bar(index,percents, label="Model 1",color='coral')
plt.legend()
plt.xlabel('Number of Models', fontsize=11)
plt.ylabel('Percentage Accuracy', fontsize=11)
plt.xticks(index,label,fontsize=7, rotation=30)
plt.title('Predictor Model 1')
plt.axis([0,7,0,100])
lebels = [' ',percent,' ',' ',' ',' ',' ']
ran=range(7)
for i in ran:
    plt.text(x=index[i],y=percents[i],s=lebels[i],size=11,color='black')
plt.show()

# Application
print('Hello there, to find out the approximate value of the house you are looking for, please answer the following questions to the best of your knowledge.')
print('What are the number of rooms?  1--10')
rooms = input()
print('What is the Plot area of the House? 1--10 in 1000sq.feet ')
plot = input()
print('Near Locomotive Stand distance from the house ? 1--4')
dist = input()
print('What Should be the house facing 0 for East and 1 for North?')
facing = input()
print('What Residential Area of the house ? Population by size locality  1--100 in 100 sq.acres')
zin = input()
print('Nearest Market Place ? 1--10 Kms')
ind = input()
print('Nearest highway away in Kms ?   1--5 Kms')
high = input()

data = np.array([rooms, plot, high,facing,ind,dist,zin])
user_df = pd.DataFrame(data, ['RM', 'INDUS', 'RAD','CHAS','DIS','STAND','ZN']).transpose()

# Prediciting off User given Data
pred = lm.predict(user_df)
res = pred[0] * 700000
print('This Real Estate is going to cost you around Rs',int(res))