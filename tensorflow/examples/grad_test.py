import tensorflow as tf

X = [[1.,2.,3.],[1.,2.,3.]]
Y = [1,2,3]

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
hypothesis = tf.matmul(W , X)
gradient = [28*(tf.reduce_sum(W)-1),28*(tf.reduce_sum(W)-1)]

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(0.01)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step,sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
