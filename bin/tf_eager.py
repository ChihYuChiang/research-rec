import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

x = np.array([[[2., 3., 1.], [4., 5., 1.]], [[0., 1., 1.], [6., 8., 1.]]])
x = tf.stack([x, x])
m_a = np.array([[[[1]], [[2]]]])
print(x)
x.shape
m_a.shape

m = tf.reduce_sum(x ** tf.tile(m_a, [1, 2, 3]), axis=0)
tf.tile(tf.reshape([1, 2, 3], [3, 1]), [1, 5])
tf.tile(tf.reshape([1, 2, 3], [1, 3]), [5, 1])
print(m)

out = tf.tile(m_a, [3, 1, 5, 5])
out = tf.reduce_prod(out, axis=1)
out1 = out ** 2
out2 = out ** 3
out1 = tf.reshape(out1[:, 1, :], [3, 5, 1])
out2 = tf.reshape(out2[:, 1, :], [3, 1, 5])
h = out1 @ out2
mask = np.broadcast_to(np.reshape(np.eye(5), (1, 5, 5)), (3, 5, 5))
h_masked = tf.reshape(tf.boolean_mask(h, mask), [3, -1])

_, refIdx = tf.nn.top_k(h_masked, k=2, sorted=True)
refIdx = refIdx[:, :1]

tf.gather(h_masked, refIdx, axis=1)
tf.reduce_sum(h_masked, axis=1) / tf.reduce_sum(h_masked, axis=1) + 3
print(h)

a = np.array([[0, 1], [1, 2]])
ax = np.broadcast_to(np.arange(2).reshape((2, 1)), (2, 2))
ax = np.array([[0, 0], [1, 1]])
a_s = tf.stack([ax, a], axis=1)
b = np.array([[1, 2, 3], [4, 5, 6]])
tf.gather_nd(b, a_s)