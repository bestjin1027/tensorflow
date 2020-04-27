import tensorflow as tf
import ctypes
import numpy as np

def list_shape(a):
	row=len(a)
	col=len(a[0])
	res = "[%d,%d]" % (row,col)
	return res

def list_content(a):
	res="["
	for i in range(len(a)):
		for j in range(len(a[i])):
			res+="%d,"%(a[i][j])
	res=res[0:len(res)-1] + "]"
	return res

def tensor_content(a):
	print("in tensor_content", a)
	res="["
	a=a.replace('[','').replace(']',' ').split()
	print(b)
	for i in range(len(a)):
		res+="%s,"%(a[i])
	res=res[0:len(res)-1] + "]"
	return res

def matmul(W,x):
    
    opkernel = "matmul"
    tensor1_shape =  "%s" % (W.shape)
    tensor1_shape = tensor1_shape.replace("(","[").replace(")","]")
    tensor1 = "%s" % (W.numpy())
    tensor1 = tensor_content(tensor1)
    
    tensor2_shape = list_shape(x)
    tensor2 = list_content(x)
    
    arg = [opkernel, tensor1_shape, tensor1, tensor2_shape, tensor2]

    matmul_lib=ctypes.CDLL(so_files)
  
    matmul_lib.matmul.argtypes = (ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p))  #set argtyps and return type
    matmul_lib.matmul.restype = ctypes.c_char_p
    
    matmul_arg_type = ctypes.c_char_p * len(arg)#set matmul_arg_type and instansiate matmul_arg and assign it with arg
    matmul_arg = matmul_arg_type()   
    for x in range(len(arg)):
        matmul_arg[x]=arg[x].encode('UTF-8')
    
    res = matmul_lib.matmul(5, matmul_arg).decode("UTF-8")#callin matmul function from libgrpc_matmul.so
    print("after in python file")
    res=res.replace('result: ','')
    res=res.splitlines()
    
    res=[float(i) for i in res]
    print(res)
    res=res
    print(type(res))
    return tf.contrib.eager.Variable(np.array(res,dtype=np.float32))

tf.enable_eager_execution()
so_files="./libgrpc_matmul.so"
#sess=tf.Session()
#sess.run(tf.initialize_all_variables())

# 앞의 코드에서는 정수였는데, 실수로 바뀌었다. 정수로 처리하면 에러.
# eval('sparray_type(%s,None)' % ','.join(["'%s'" % x for x in matmul_arg]))
x_data = [[1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]   # W의 값은 [[-0.49763036  0.79181528]]
y_data  = [1, 2, 3, 4, 5]



W = tf.contrib.eager.Variable(tf.random_uniform([1, 2], -1.0, 1.0))   # [2, 1]은 안 된다. 행렬 곱셈이니까.
b = tf.contrib.eager.Variable(tf.random_uniform([1], -1.0, 1.0))
out = tf.contrib.eager.Variable(tf.random_uniform([1, 5], -1.0, 1.0))

print(W.numpy())
print(b.numpy())
learning_rate = 0.001

for i in range(2001):
    t=tf.GradientTape(persistent=True)
    t.__enter__()
    t.watch(W)
    t.watch(b)
    hypothesis = matmul(W,x_data) + b
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    w_grad = t.gradient(cost,W)
    b_grad = t.gradient(cost,b)
    #print("YEO14:",w_grad, b_grad)
    W.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i%100==0:
        print("{:5} | {:10.4f}".format(i,cost.numpy()))
'''
sess=tf.Session()
sess.run(tf.initialize_all_variables())

# W와 곱해야 하기 때문에 x_data를 실수로 변경

                # (1x2) * (2x5) = (1x5)
with tf.GradientTape(persistent=True) as tape2 :
	with tf.GradientTape(persistent=True) as tape1:
		tape2.watch(b)
		tape1.watch(W)
		hypothesis = tf.matmul(W, x_data) + b
		cost = tf.reduce_mean(tf.square(hypothesis - y_data))
  	
grad1 = tape1.gradient(cost,W)
grad2 = tape2.gradient(cost,b)

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
	sess.run(train)

	if step%100 == 0:
		print(step, sess.run(cost), sess.run(W), sess.run(b))

sess.close()
'''