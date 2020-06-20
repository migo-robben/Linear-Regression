# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json

# housing data文件位置
data_file_path = "data/housing.data"

def load_data(file_path):
    # 读入数据
    data = np.fromfile(file_path, sep=' ')

    # 7084个数据，每14个为一组，前13个是影响房价的特征值，第14个为该类型房屋的平均价格
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 拆分为训练集和测试集
    # 80%为训练集，20%为测试集
    # 拆分率 Ratio
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    # 以及对数据进行归一化处理
    # max (axis=0), 就是'CRIM', 'ZN'， ... 'MEDV'各自的最大值
    maximums, minimums, avgs = training_data.max(axis=0) , training_data.min(axis=0), \
                                training_data.sum(axis=0)/training_data.shape[0]
    
    # 归一化
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
        # data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data, minimums, maximums

class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(90)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    # 计算损失函数
    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

    # 计算梯度
    def gradient(self, x, y):
        z = self.forward(x)

        gradient_w = (z - y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]

        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)

        return gradient_w, gradient_b

    # 更新参数
    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    # 训练
    def train(self,
        training_data,
        num_epoches=50,
        enable_mini_batch=True,
        iterations=100,
        batch_size=10,
        eta=0.01):
        if enable_mini_batch:
            n = len(training_data)
            losses = []
            for epoch_id in range(num_epoches):
                # 在每轮迭代开始之前，将训练数据的顺序随机的打乱，
                # 然后再按每次取batch_size条数据的方式取出
                np.random.shuffle(training_data)
                # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
                mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
                for iter_id, mini_batch in enumerate(mini_batches):
                    #print(self.w.shape)
                    #print(self.b)
                    x = mini_batch[:, :-1]
                    y = mini_batch[:, -1:]
                    a = self.forward(x)
                    loss = self.loss(a, y)
                    gradient_w, gradient_b = self.gradient(x, y)
                    self.update(gradient_w, gradient_b, eta)
                    losses.append(loss)
                    print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                     format(epoch_id, iter_id, loss))
            
            return losses

        else:
            losses = []
            x = training_data[:, :-1]
            y = training_data[:, -1:]
            for i in range(iterations):
                z = self.forward(x)
                L = self.loss(z, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(L)
                if (i) % 10 == 0:
                    print('iter {}, loss {}'.format(i, L))
            return losses

    # 预测
    def prediction(self, test_data):
        prediction_y = np.dot(test_data[:, :-1], self.w) + self.b
        return prediction_y

# 获取数据
training_data, test_data, minimums, maximums = load_data(data_file_path)

# 创建网络
net = Network(13)

# 启动训练
losses = net.train(training_data,
    num_epoches=500,
    enable_mini_batch=True,
    iterations=100,
    batch_size=100,
    eta=0.1)

# 小批量batch_size是100,如果batch_size=506,也就是和全数据的梯度下降一样了
# losses = net.train(training_data,
#     num_epoches=100,
#     enable_mini_batch=True,
#     batch_size=06,
#     eta=0.1)
# 和下面的相同
# losses = net.train(training_data,
#     enable_mini_batch=False,
#     iterations=100,
#     eta=0.1)

# 画出损失函数的变化趋势
plt.figure(figsize=(12, 9))
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.subplot(211)
plt.title("Losses")
plt.plot(plot_x, plot_y)

# 预测
test_prediction_y = net.prediction(test_data)
test_prediction_y = test_prediction_y * (maximums[-1]-minimums[-1]) + minimums[-1] # normal back

true_prediction_y = test_data[:, -1:]
true_prediction_y = true_prediction_y * (maximums[-1]-minimums[-1]) + minimums[-1] # normal back

x = np.arange(0, 102, 1)
plt.subplot(212)
plt.title("Prediction")
plt.plot(x, test_prediction_y, color='r', label="test_prediction_y")
plt.plot(x, true_prediction_y, color='g', label="true_prediction_y")
plt.legend(loc='best')
plt.show()