import tensorflow as tf
import numpy as np

X = [[1.,0.,3.,0.,5.],[0.,2.,0.,4.,0.]]
XT = [[1.,0.],[0.,2.],[3.,0.],[0.,4.],[5.,0.]]
Y = [1.,2.,3.,4.,5.]
W = tf.Variable([[0.,0.]])
b = tf.Variable([1.])

hypothesis = tf.matmul(W , X) + b -Y
cost = tf.reduce_mean(tf.square(hypothesis))
optimizer = tf.train.GradientDescentOptimizer(0.1)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

grad_W = tf.matmul((hypothesis),XT)/2
grad_b = tf.reduce_sum(tf.reduce_mean((hypothesis)*2))

sess = tf.Session()
sess.run(tf.global_variables_initializer())



for step in range(2001):
    print(step, sess.run([grad_W,W,grad_b,b]))
    print(step, sess.run(gvs))
    sess.run(apply_gradients)
