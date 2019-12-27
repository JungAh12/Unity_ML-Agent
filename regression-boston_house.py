'''
Classification & Regeression

==regression==
- Dataset : Boston Housing
- features
    * 범죄율
    * 비소매상업지역의 면적 비율
    * 일산화질소 농도
    * 주택당 평균 방 개수
    * 인구 중 하위 계층의 비율
    * 인구 중 흑인 비율
    * 학생/교사의 비율
    * 25,000 평방피트를 초과하는 거주 지역의 비율
    * 찰스강의 경계에 위치했는지 여부(0 또는 1)
    * 1940년 이전 건축된 주택 비율
    * 방사형 고속도로까지의 거리
    * 직업센터까지의 거리
    * 재산세율
'''

# 1-1. 필요 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import datetime

# 1-2. 파라미터 세팅
algorithm = 'ANN'

data_size = 13          # feature 수

load_model = False      # 미리 저장된 모델 정보를 불러올 것인가

batch_size = 32         # 배치사이즈(한 번 학습 시 데이터 셋에서 가져올 데이터 수)

learning_rate = 1e-3    # 학습률

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/" + date_time + "_" + algorithm
load_path = "./saved_models/" + date_time + "_" + algorithm + "/model/model "

# 2. Boston Housing 데이터 불러오기
boston_housing = tf.keras.datasets.boston_housing.load_data(path='boston_housing.npz')

# 학습 데이터 가져오기
x_train = boston_housing[0][0]
y_train = np.reshape(boston_housing[0][1],(-1,1))   # target 값 저장

# 8:2 비율로 train set, validation set 나눔
x_train, x_valid = x_train[:len(x_train)*8//10], x_train[len(x_train)*8//10:]
y_train, y_valid = y_train[:len(y_train)*8//10], y_train[len(y_train)*8//10:]

# 테스트 셋 가져오기
x_test = boston_housing[1][0]
y_test = np.reshape(boston_housing[1][1],(-1,1))

# 3. Model 클래스 생성
class Model():
    def __init__(self):
      
        self.dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.dnn.compile(loss='mse',optimizer='Adam',metrics=['mse'])

class ANN():
    def __init__(self):
        self.model = Model()

        if load_model:
            self.model = tf.keras.models.load_model(load_path+'reg.h5')

    def train_model(self, data_x, data_y, batch_idx):
        len_data = data_x.shape[0]

        # batch_idx + batch_size가 data길이보다 길지 않으면
        if batch_idx + batch_size < len_data:
            batch_x = data_x[batch_idx : batch_idx+batch_size, :]
            batch_y = data_y[batch_idx : batch_idx+batch_size, :]

        else:
            batch_x = data_x[batch_idx : len_data, :]
            batch_y = data_y[batch_idx : len_data, :]

        history = self.model.dnn.fit(batch_x,batch_y)
        
        return history.history['loss']

    def test_model(self, data_x, data_y):
        loss, acc = self.model.dnn.evaluate(data_x, data_y)

        return loss

    #모델 저장
    def save_model(self):
        self.model.dnn.save(save_path+'reg.h5')

    #TensorBoard에 loss값 시각화
    def Write_Summary(self, train_loss, val_loss, batch):
        train_log_dir = save_path + '/train'
        test_log_dir = save_path + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)

if __name__=='__main__':
    ann = ANN()
    data_train = np.zeros([x_train.shape[0], data_size+1])
    data_train[:, :data_size] = x_train
    data_train[:, data_size:] = y_train
    total_train_loss = []
    total_val_loss = []
    # 학습 수행
    for epoch in range(num_epoch):
        train_loss_list = []
        val_loss_list = []

        #데이터를 섞어서 입력과 실제값 분리
        np.random.shuffle(data_train)
        train_x = data_train[:, :data_size]
        train_y = data_train[:, data_size:]

        #학습 수행, 손실함수 값 계산 및 텐서보드에 값 저장
        for batch_idx in range(0, x_train.shape[0], batch_size):
            train_loss = ann.train_model(train_x, train_y, batch_idx)
            val_loss = ann.test_model(x_valid, y_valid)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
        total_train_loss.extend(train_loss_list)
        total_val_loss.extend(val_loss_list)
        print("Epoch: {} / Train loss : {:.5f} / Val loss : {:.5f}"
            .format(epoch+1, np.mean(train_loss_list), np.mean(val_loss_list)))
        ann.Write_Summary(np.mean(train_loss_list), np.mean(val_loss_list), epoch)

    test_loss = ann.test_model(x_test, y_test)
    print('-----------------------------------------------------------')
    print('Test Loss : {:.5f}'.format(test_loss))

    #모델 저장
    ann.save_model()

    #loss matplot으로 시각화
    import matplotlib.pyplot as plt
    
    plt.plot(total_train_loss)
    plt.plot(total_val_loss)
