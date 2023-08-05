# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the Iris dataset
df = pd.read_csv('iris.csv')
print(df)

# Separate features and target
X = df.iloc[:, :-1].values
X = np.array(X)
y = df.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Apply feature scaling on the training and test sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

# Print the scaled training and test sets
print(X_train)
print(X_test)
print(y_train)
print(y_test)