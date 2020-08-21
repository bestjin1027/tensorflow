import tensorflow as tf
import numpy as np

X_data = [[1.,0.,3.,0.,5.],[0.,2.,0.,4.,0.]]
XT_data = [[1.,0.],[0.,2.],[3.,0.],[0.,4.],[5.,0.]]
Y_data = [1.,2.,3.,4.,5.]
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = tf.mae(W , b)
cost = tf.reduce_mean(tf.square(hypothesis))
optimizer = tf.train.GradientDescentOptimizer(0.1)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

grad_W = tf.matmul((hypothesis),XT)
grad_b = tf.reduce_sum((hypothesis - Y)*2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    print(step, sess.run([grad_W,W,grad_b,b,gvs]))
