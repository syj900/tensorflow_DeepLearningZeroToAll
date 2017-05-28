"""
신윤중
2017-5-28
lab-03-2-minimizing_cost_gradient_update
"""

import tensorflow as tf
tf.set_random_seed(777)

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1     # 학습률
gradient = tf.reduce_mean((W * X - Y) * X)  # 미분값
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train = optimizer.minimize(cost) 와 같은 동작
descent = W - learning_rate * gradient  # 미분값을 뺌으로써 기울기 0에 더 가까워진다 (cost 최소)
update = W.assign(descent)  # descent operation 의 수행 값을 W에 assign 한다.
# 이 때 수행되는 operation 이 update 이다
# update operation 실행시 이하 node 모두 실행됨

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


pass    # for debug
