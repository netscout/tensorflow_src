import tensorflow as tf

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

#추후에 실제 값이 입력됨ㅇ
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

#X와 W를 행렬곱해서 Y가 나올 것이므로, W는 X의 입력(2), Y출력(1)을 통해 2X1의 행렬이 된다.
W = tf.Variable(tf.random_normal([2,1]), name="weight")
#bias는 항상 나가는 값(Y)의 개수와 같다.
b = tf.Variable(tf.random_normal([1]), name="bias")

h = tf.sigmoid(tf.matmul(X, W) + b)

loss = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#0.5보다 크다면 True, 아니면 False
predicted = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())

    for step in range(10001):
        loss_val, _ = ss.run([loss, train], feed_dict={X:x_data, Y: y_data})
        if step % 200 == 0:
            print(step, loss_val)
    #h => 0.0~1.0사이의 예측값, p => 0 or 1(True or False), a => 정확도
    _h, c, a = ss.run([h, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})
    print("\nh : ", _h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
