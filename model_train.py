import numpy as np
import matplotlib.pyplot as plt
import pickle
from param import nn


# 加载模型
f = open('/Users/a123/Desktop/main/data/hidden_nodes.txt', 'rb')
hidden_nodes = pickle.load(f)
f.close()

input_nodes = 784
output_nodes = 10
final_model = nn(input_nodes, hidden_nodes, output_nodes)

f = open('/Users/a123/Desktop/main/data/mu.txt', 'rb')
mu = pickle.load(f)
f.close()
f = open('/Users/a123/Desktop/main/data/learning_rate.txt', 'rb')
learning_rate = pickle.load(f)
f.close()

test_flag = 1  # 等于1时表示每次迭代训练都进行测试集的测试并保存结果
train_number = 500  # 训练次数
print_flag = 1  # 等于1时打印训练的loss

# 训练模型
final_model.train(train_number, mu, learning_rate, test_flag, print_flag)
# 可视化训练和测试的loss曲线
x = np.arange(len(final_model.train_loss))
plt.plot(x, final_model.train_loss, label='train loss')
plt.plot(x, final_model.test_loss, label='test loss', linestyle='--')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('loss_image.jpg')
plt.show()

# 可视化测试的accuracy曲线
plt.plot(x, final_model.test_accuracy, label='test accuracy')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig('accuracy_image.jpg')
plt.show()
print("测试集的分类精度：", final_model.test(0, mu))  # 输出测试集上的结果

# 保存模型的权重矩阵
final_model.store_values()


