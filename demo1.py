# -- coding: utf-8 --
import tensorflow as tf

def loss_fun(logits, labels):
	# 二次代价函数
	loss_mean = tf.losses.mean_squared_error(logits, labels)
	tf.add_to_collection('losses', loss_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def get_params():
	a = [[1, 2, 3], [2, 3, 4], [5, 6, 7], [1, 3, 3], [7, 3, 4], [2, 1, 7], [1, 5, 3], [2, 9, 4], [5, 1, 7], [1, 1, 3]]
	b = [[7], [10], [19], [8], [15], [11], [10], [16], [13], [6]]
	weight = tf.Variable(tf.constant([[0], [0], [0]], dtype=tf.float32))
	bias = tf.Variable(tf.constant(0, dtype=tf.float32))
	x = tf.placeholder(tf.float32, [10, 3])
	y = tf.placeholder(tf.float32, [10, 1])
	logits = tf.matmul(x, weight) + bias

	# 计算代价函数
	loss = loss_fun(logits, y)
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # 应用Adam优化算法

	# 建立会话
	sess = tf.Session()
	sess.run(tf.global_variables_initializer()) # 初始化变量
	w1, b1 = sess.run([weight, bias])
	for i in range(2000):
		_, loss_value = sess.run([train_op, loss], feed_dict={x: a, y: b})
		if i % 200 == 0:
			print(i, loss_value)
	w2, b2 = sess.run([weight, bias])
	sess.close()
	return weight, bias, w1, b1, w2, b2

def get_result(w_tensor, b_tensor):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	w, b = sess.run([w_tensor, b_tensor])
	sess.close()
	return w, b

'''
运行命令如下：
import demo1
w_tensor, b_tensor, w1, b1, w2, b2 = demo1.get_params()
w, b = demo1.get_result(w_tensor, b_tensor)

输出如下：
w_tensor: <tf.Variable 'Variable:0' shape=(3, 1) dtype=float32_ref>
b_tensor: <tf.Variable 'Variable_1:0' shape=() dtype=float32_ref>
w1: [[0], [0], [0]]
b1: 0
w2: [[0.9631803], [0.98583895], [0.96481216]]
b2: 0.9731251
w: [[0], [0], [0]]
b: 0

一个关于tensorflow的简单小例子，拟合一条回归直线，主要是出于对tensorflow优化代价函数时，w和b参数变化的疑惑。
实践证明，优化过程中w和b会不断变化，但是直接输出得到的只是初始化的tensor，需要run才能把真正的参数输出：即输出中，w和b的值并不等于w2和b2，而是等于w1和b1
'''