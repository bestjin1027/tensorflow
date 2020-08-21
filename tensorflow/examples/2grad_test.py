import tensorflow as tf
import numpy as np

X_data = [[1.,0.,3.,0.,5.],[0.,2.,0.,4.,0.]]
XT_data = [[1.,0.],[0.,2.],[3.,0.],[0.,4.],[5.,0.]]
Y_data = [1.,2.,3.,4.,5.]
z_data = [[1.,1.],[1.,1.]]

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
XT=tf.placeholder(tf.float32)

hypothesis = tf.matmul(W , X) + b -Y

grad_W = tf.matmul((hypothesis),XT)/100
grad_b = tf.reduce_sum((hypothesis)*2)

cost = tf.reduce_mean(tf.square(hypothesis))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

descent_w = W-0.1 * grad_W
descent_b = b-0.1 * grad_b
train_w = W.assign(descent_w)
train_b = b.assign(descent_b)

for step in range(101):
    sess.run(train_w,feed_dict={X: X_data, Y: Y_data, XT: XT_data})
    sess.run(train_b,feed_dict={X: X_data, Y: Y_data, XT: XT_data})
    print(step, sess.run(cost, feed_dict={X: X_data, Y: Y_data, XT: XT_data}),sess.run(W),sess.run(b))
