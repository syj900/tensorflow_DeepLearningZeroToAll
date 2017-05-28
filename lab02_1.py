"""
신윤중
2017-5-28
lab02_1 - linear_regression
"""

import tensorflow as tf
tf.set_random_seed(777)

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Try to find values for W and b to compute y_data = x_data * W + b
# 위 데이터를 입력한다면 W = 1, b = 0일 것이다.
# tensorflow 를 이용하여 알아보자
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# W와 b에 크기 [1]의 초기 랜덤값이 저장된다

# hypothesis XW+b
hypothesis = x_train * W + b
# 회귀 방정식 가중치 W와 편차 b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_mean : tensor 데이터의 rank 를 1로 감소 시킨다. 행, 열에 따른 처리도 가능하다.
# square : 제곱식
# 추정값과 실제값(y_train)간 차이의 제곱을 평균내어 cost 에 저장한다.

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 학습률 0.01의 gradient Descent 함수를 optimizer 에 저장
train = optimizer.minimize(cost)
# cost 를 최소화시킨다. train 에 저장
# 이 과정에서 W와 b의 최적화가 이루어진다.

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
# tensorflow variable 을 사용할 때 반드시 우선 사용되야하는 초기화 함수

# Fit the line
for step in range(2001):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
# Learns best fit W:[1.], b:[1.]
