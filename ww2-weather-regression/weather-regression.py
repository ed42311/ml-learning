import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv(
  'Weather.csv',
  dtype={"MeanTemp": float, "Snowfall": str, "PoorWeather": str, "SNF": str, "TSHDSBRSGF": str},
  skip_blank_lines=True
)

# view data in dataframe
# print(dataset.shape)
# print(dataset.describe())

# plot data on simple x and y matrix
# dataset.plot(x='MinTemp', y='MaxTemp', style='o')  
# plt.title('MinTemp vs MaxTemp')  
# plt.xlabel('MinTemp')  
# plt.ylabel('MaxTemp')  
# plt.show()

# plt.figure(figsize=(15,10))
# plt.tight_layout()
# seabornInstance.distplot(dataset['MaxTemp'])
# plt.show()

X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

# To retrieve the intercept:
# print(regressor.intercept_)
# For retrieving the slope:
# print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
# print(df)

# diff actaul versus predicted
# df1 = df.head(25)
# df1.plot(kind='bar',figsize=(16,10))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()

# scatter plot with regression line on top
# plt.scatter(X_test, y_test,  color='gray')
# plt.plot(X_test, y_pred, color='red', linewidth=2)
# plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
