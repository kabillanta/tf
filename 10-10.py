# form __futrue__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output


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

#training is done in batches 
#epoch is the number of times the model will see the same data

#Input function 
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

#linera classifier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)


#train the model 
linear_est.train(make_input_fn(dftrain, y_train))
result = linear_est.evaluate(make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False))
clear_output()
print(result['accuracy'])