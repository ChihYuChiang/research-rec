import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

x = np.array([[[2., 3., 1.], [4., 5., 1.]], [[0., 1., 1.], [6., 8., 1.]]])
m_a = np.array([[[1]], [[2]]])
x.shape
m_a.shape

m = tf.reduce_sum(x ** tf.tile(m_a, [1, 2, 3]), axis=0)
tf.tile(tf.reshape([1, 2, 3], [3, 1]), [1, 5])
tf.tile(tf.reshape([1, 2, 3], [1, 3]), [5, 1])
print(m)

