import tensorflow as tf

#tf.data를 이용하여 파일에서 데이터 읽어오기
iterator = tf.data.TextLineDataset("data-01-test-score.csv")\
           .skip(1)\   #첫번째 줄은 제외하고(헤더)
           .repeat()\  #파일의 끝에 도달하더라도 처음부터 무한 반복
           .batch(10)\ #한 번에 10개씩 묶어서 사용
           .make_initializable_iterator();

#반복자가 다음 데이터를 읽어오도록 dataset 노드에 명령을 저장
dataset = iterator.get_next()

#csv를 읽어서 데이터로 변환한다.
lines = tf.decode_csv(dataset, record_defaults=[[0.], [0.], [0.], [0.]])
#변환된 데이터의 첫번째 열부터 마지막-1번째 열까지 합쳐서 train_x_batch에 할당
train_x_batch = tf.stack(lines[0:-1], axis=1)
#마지막 열을 합쳐서 train_y_batch에 할당
train_y_batch = tf.stack(lines[-1:], axis=1)

#placeholder는 실제 값이 입력된다.
X = tf.placeholder(tf.float32, shape=[None, 3]) #x값은 Nx3의 행렬이다.
y = tf.placeholder(tf.float32, shape=[None, 1]) #y값은 Nx1의 행렬이다.

#Variable은 모델을 훈련하여 업데이트 된다.
W = tf.Variable(tf.random_normal([3,1]), name="weight") #3x1의 행렬(3행 1열)
b = tf.Variable(tf.random_normal([1]), name="bias")

#가설h
h = tf.matmul(X, W) + b

#loss(cost) 함수
loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

ss = tf.Session()
ss.run(tf.global_variables_initializer())

for step in range(2001):
    #반복자 초기화
    ss.run(iterator.initializer)
    #반복자를 통해 할당된 값을 읽어오기
    x_batch, y_batch = ss.run([train_x_batch, train_y_batch])
    #모델 훈련을 진행하고 loss및 h값 가져오기
    loss_val, h_val, _ = ss.run([loss, h, train],
                                     feed_dict={X: x_batch, y: y_batch})
    if step % 10 == 0:
        print("loss : {0}".format(loss_val))

print("score..", ss.run(h, feed_dict={X:[[100,70,101]]}))
