"""
신윤중
2017-5-28
lab03-1-minimizing_cost_show_graph
"""

import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
W_history = []
cost_history = []
# # graph 를 그리기 위해 plot 값이 저장될 리스틀르 만든다

for i in range(-30, 50):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict={W: curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)
# W의 변화에 따라 cost 가 어떻게 달라지는지 plot 시킨다
# W가 -3에서 부터 5까지 0.1간격으로 변화함

# Show the cost function
plt.plot(W_history, cost_history)
# 리스트형태의 좌표를 순서대로 plot 시켜줌
plt.show()