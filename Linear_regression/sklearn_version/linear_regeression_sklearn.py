import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
# read data
data = np.loadtxt("G:\github\MachineLearning_Code_Practice\Linear_regression\sklearn_version\\train_data.txt")
x_values = data[:,:-1]
y_values = data[:,-1]

# train model on data
linear_reg = linear_model.LinearRegression()
linear_reg.fit(x_values, y_values)

# results
result = linear_reg.predict(x_values)

for index in range(len(result)):
    plt.scatter(index, y_values[index],c='r')
    plt.scatter(index, result[index],c='b')
plt.show()