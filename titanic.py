# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# loading in the dataset
dataset = pd.read_csv('titanic.csv')

# identify categorical features
cd_list = ['Sex', 'Embarked', 'Pclass']
print(cd_list)

# apply OneHotEncoder to the list of categorical features
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), cd_list)], remainder = 'passthrough' )

# use the fit transform method for column transformer
X = ct.fit_transform(dataset.drop('Survived', axis=1))

X = np.array(X)

le = LabelEncoder()
y = le.fit_transform(dataset['Survived'])

decoded_y = le.inverse_transform(y)

print(X)
print(y)