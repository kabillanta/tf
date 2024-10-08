#importing tensorflow library
import tensorflow as tf 
checking the version of tensorflow
print(tf.version)

#Tensor - A tensor is the fundamental data structure used to represent data. It is a generalization of matrices that can be used to represent multi-dimensional arrays
#Each tensor has a data type and a shape. 
#Datatypes - tf.float32, tf.int32, tf.string, tf.bool
#Shape - A tensor's shape is the number of elements in each dimension.


#Creating a tensor
string = tf.Variable("This is a String", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

#rank/dimension of a tensor -  The rank of a tensor is the number of dimensions it has.

# rRank 1 tensor

rank1tensor = tf.Variable(["This","is a String"], tf.string)
tf.rank(rank1tensor)

rank3tensor = tf.Variable([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]], tf.int16)
print(tf.rank(rank3tensor))

# If shape - (2,3) it contains 2 list which has 3 elements each
# (3,-1) - 3 list and -1 indicates that it tf will automatically calculate the second dimension automatically  

#Types of tensors 
"""
Variable - Mutable tensor
Constant - Immutable tensor
Placeholder - Empty tensor
SparseTensor - Tensor that contains mostly zero values
"""


#Evaluating a tensor for tf 1.0
with tf.Session() as sess:
    rank1tensor.eval()


t = tf.zeros([5,5,5,5])
print(t)
print(tf.reshape(t,[125,-1]))