"""
신윤중
2017-5-20
lab01 - basics
"""

import tensorflow as tf

# Create a constant op
# This op is added as a note to the default graph
hello = tf.constant("Hello tensorflow!")

# start a TF session
sess = tf.Session()
"""ssesion 이란?"""

# run the op and get result
print(sess.run(hello))

# Tensor
3   # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.]    # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]]    # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [7., 8., 9.]]  # a rank 3 tensor with shape [2, 1, 3]

# Computational Graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2)    # node3는 node1과 node2를 필요로 하는 node 다

print("node1: ", node1, "node2: ", node2)   # tensor 인 node 의 형태, 속성을 출력함
print("node3: ", node3)     # node3 의 속성을 출력함(기능, shape, data type)

sess = tf.Session()
print("sess.run(node1, node2) :", sess.run([node1, node2]))     # [] 형태의 필요성 ; 각개의 node 를 실행시킴
print("sess.run(node3): ", sess.run(node3))     # 위와 달리 node3 는 node1과 node2를 필요로 하므로 node3만 실행해도 저절로 실행됨

a = tf.placeholder(tf.float32)  # 변수의 type(float32) 을 지정해 주고 데이터를 입력할 수 있게 설정
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a,b) like that adder_node = tf.add(a,b)

print(sess.run(adder_node, feed_dict={a: 3, b: 2}))
print(sess.run(adder_node, feed_dict={a: [3, 2], b: [5, 4]}))

adder_and_triple = adder_node * 3
print(sess.run(adder_and_triple, feed_dict={a: 3, b: 4.5}))
