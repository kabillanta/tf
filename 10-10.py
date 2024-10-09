# form __futrue__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') 
# print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# print(y_train.head())
# print(y_eval.head())

# #age histogram
# # dftrain.age.hist(bins=20)   

# #sex histogram
# dftrain.sex.value_counts().plot(kind='bar')

# #class histogram
# dftrain["class"].value_counts().plot(kind='bar')

# plt.show()


from tensorflow import feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck','embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []
for fearure_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[fearure_name].unique()
    feature_columns.append(feature_column.categorical_column_with_vocabulary_list(fearure_name, vocabulary))

for fearure_name in NUMERIC_COLUMNS:
    feature_columns.append(feature_column.numeric_column(fearure_name, dtype=tf.float32))


print(feature_columns)

