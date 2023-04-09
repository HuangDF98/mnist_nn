import os
import struct
import numpy as np
import pickle


class nn:

    def __init__(self, D_in, H, D_out):
        # 首先对最基础的变量进行定义
        self.inodes = D_in
        self.hnodes = H
        self.onodes = D_out
        self.train_loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

        # 对一开始的w1和之后的w2创建权重矩阵
        self.w1 = np.random.randn(self.inodes, self.hnodes)
        self.w2 = np.random.randn(self.hnodes, self.onodes)

        # 如果有构建好的模型参数，直接加载，如果没有就跳过
        if os.path.exists("/Users/a123/Desktop/main/data/json_w1.txt"):
            self.w1 = np.loadtxt("/Users/a123/Desktop/main/data/json_w1.txt")
        else:
            pass

        if os.path.exists("/Users/a123/Desktop/main/data/json_w2.txt"):
            self.w2 = np.loadtxt("/Users/a123/Desktop/main/data/json_w2.txt")
        else:
            pass

    def load_data(self):
        x_test, y_test = self.load_mnist("mnist_data", 't10k')
        x_test = x_test / 255 * 0.99 + 0.01
        y_test = self.one_hot(y_test, 10)
        x_train, y_train = self.load_mnist("mnist_data", 'train')
        y_train = self.one_hot(y_train, 10)
        x_train = x_train / 255 * 0.99 + 0.01
        # 这里其实是一种，把数据化成0-1的方法，然后加上0.01是为了让得到的训练和测试数据不为0
        return x_train, y_train, x_test, y_test


    # 定义个可以通用的softmax函数
    @staticmethod
    def softmax(s):
        s_max = s - s.max(axis=1).reshape((-1, 1))
        temp_s = np.exp(s_max)
        temp = temp_s / temp_s.sum(axis = 1).reshape((-1,1))  
        return temp

    # 正向传播
    def forward_pass(self, input):
        inputs = np.array(input, ndmin=2)
        t = inputs @ self.w1
        h = 1 / (1 + np.exp(-t))
        ypp = h @ self.w2
        yp = self.softmax(ypp)
        return yp, h

    # 交叉熵的误差
    @staticmethod
    def cee(y, t):
        d = 1e-7
        # 为了防止计算错误，加上一个微小值
        dd = -np.sum(t * np.log(y + d))
        return dd


    # 损失函数 其中要加入w1和w2的正则项 lossfunction
    def lf(self, mu, yp, y):
        loss = self.cee(yp, y) + 0.5 * mu * (np.sum(self.w1 * self.w1) +  np.sum(self.w2 * self.w2))
        return loss

    # 梯度计算 通过SGD的方法
    def gradient_compute(self, size, yp, h, mu):

        sample = np.random.randint(0, self.x_train.shape[0], size)
        yp1 = yp[sample, :]
        y1 = self.y_train[sample, :]
        h1 = h[sample, :]
        x1 = self.x_train[sample, :]
        dypp = yp1 - y1
        dw2 = h1.T @ dypp + mu * np.ones(self.w2.shape)
        dh = dypp @ self.w2.T
        dw1 = x1.T @ (dh * (h1 * (1 - h1))) + mu * np.ones(self.w1.shape)
        return dw1, dw2

    # 把输入转为one-hot的函数
    @staticmethod
    def one_hot(y, C):
        return np.eye(C)[y.reshape(-1)]

    # 把one——hot的结果转为标签
    @staticmethod
    def oh_to_label(y):
        return np.argmax(y, axis=-1)


    # 导入mnist数据的方法
    @staticmethod
    def load_mnist(path, kind='train'):

        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

        with open(labels_path, 'rb') as label_path:
            magic, n = struct.unpack('>II', label_path.read(8))
            labels = np.fromfile(label_path, dtype=np.uint8)

        with open(images_path, 'rb') as img_path:
            magic, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
            images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 784)
        return images, labels


    # 定义训练函数的方法
    def train(self, train_number, mu, lr, test1, print1):
        # decrease的gamma一开始设为0.99
        dgamma = 0.99
        for t in range(train_number):
            # 正向传播
            yp, h = self.forward_pass(self.x_train)

            # 计算loss的值
            loss = self.lf(mu, yp, self.y_train)
            self.train_loss.append(loss)

            if print1 == 1:
                print("第", t+1, "次训练的损失值:", loss)
            else:
                pass

            # 反向传播
            sample_size = 128
            dw1, dw2 = self.gradient_compute(sample_size, yp, h, mu)

            # 每次都更新w1和w2
            self.w1 -= lr * (dw1 / sample_size)
            self.w2 -= lr * (dw2 / sample_size)

            # 学习率训练的方法
            if t % 100 == 0:
                lr = dgamma * lr

            if test1 == 1:
                self.test(1, mu)
            else:
                pass

    def test(self, test1, mu):
        h = np.exp(-self.x_test @ self.w1)
        testh = 1 / (1 + h)
        testypp = testh @ self.w2
        # 把（64，100）的隐藏层h转为（64，10）的输出矩阵Y
        testyp = self.softmax(testypp)  
        accuracy = np.sum(self.oh_to_label(testyp) == self.oh_to_label(self.y_test)) / self.y_test.shape[0]

        if test1 == 1:
            self.test_accuracy.append(accuracy)
            # 计算loss
            loss = self.lf(mu, testyp, self.y_test)
            self.test_loss.append(loss)
        else:
            pass

        return accuracy

    # 将权重写入到指定的文件中来
    def store_values(self):  
        filename_w1 = "/Users/a123/Desktop/main/data/json_w1.txt"

        filename_w2 = "/Users/a123/Desktop/main/data/json_w2.txt"

        with open(filename_w1, 'w') as fw1:
            np.savetxt(fw1, self.w1)

        with open(filename_w2, 'w') as fw2:
            np.savetxt(fw2, self.w2)


if __name__ == '__main__':
    # 构建神经网络结构
    input_nodes = 784
    output_nodes = 10

    # 参数查找
    hidden_nodes_list = [50, 100, 200]  # 隐藏层大小
    mu_list = [1e-3, 1e-5, 1e-7]  # 正则化强度
    learning_rate_list = [3,1,0.5,0.05]  # 学习率
    test_flag = 0  # 等于1时表示每次迭代训练都进行测试集的测试并保存结果
    train_number = 100  # 训练次数
    print_flag = 1  # 等于1时打印训练的loss
    # 最小的loss，最终选择同等迭代下最后一次loss最小的
    loss_min = 1e10

    hidden_nodes_final = 200
    mu_final = 1e-3
    learning_rate_final = 2
    # hidden_nodes = 100
    # mu = 1e-05
    # learning_rate = 3
    for hidden_nodes in hidden_nodes_list:
        for mu in mu_list:
            for learning_rate in learning_rate_list:
                model = nn(input_nodes, hidden_nodes, output_nodes)
                model.train(train_number, mu, learning_rate, test_flag, print_flag)
                if model.train_loss[-1] < loss_min:
                    loss_min = model.train_loss[-1]
                    print(model.train_loss[-1])
                    hidden_nodes_final = hidden_nodes
                    mu_final = mu
                    learning_rate_final = learning_rate
                else:
                    pass
    # model = nn(input_nodes, hidden_nodes, output_nodes)
    # print(model.train(train_number, mu, learning_rate, test_flag, print_flag))

    print("最终学习率为", learning_rate_final, "；最终隐藏层大小为", hidden_nodes_final, "；最终正则化强度为", mu_final)

    # 保存各项参数
    f = open('/Users/a123/Desktop/main/data/hidden_nodes.txt', 'wb')
    pickle.dump(hidden_nodes_final, f)
    f.close()
    f = open('/Users/a123/Desktop/main/data/mu.txt', 'wb')
    pickle.dump(mu_final, f)
    f.close()
    f = open('/Users/a123/Desktop/main/data/learning_rate.txt', 'wb')
    pickle.dump(learning_rate_final, f)
    f.close()


