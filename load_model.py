import numpy as np
import matplotlib.pyplot as plt
from param import nn
import pickle

# 加载模型所需参数
f = open('/Users/a123/Desktop/main/data/hidden_nodes.txt', 'rb')
hidden_nodes = pickle.load(f)
f.close()
f = open('/Users/a123/Desktop/main/data/mu.txt', 'rb')
mu = pickle.load(f)
f.close()

# 构建模型
input_nodes = 784
output_nodes = 10
final_model = nn(input_nodes, hidden_nodes, output_nodes)

# 进行测试，输出分类精度
print("测试集的分类精度：", final_model.test(0, mu))

# 可视化每层的网络参数
figure = plt.figure()
axes = figure.add_subplot(221)
caxes = axes.matshow(final_model.w1[0:200, :].T, interpolation='nearest')
figure.colorbar(caxes)
axes = figure.add_subplot(222)
caxes = axes.matshow(final_model.w1[200:400, :].T, interpolation='nearest')
figure.colorbar(caxes)
axes = figure.add_subplot(223)
caxes = axes.matshow(final_model.w1[400:600, :].T, interpolation='nearest')
figure.colorbar(caxes)
axes = figure.add_subplot(224)
caxes = axes.matshow(final_model.w1[600:784, :].T, interpolation='nearest')
figure.colorbar(caxes)
plt.savefig('w1_image_break.jpg')
plt.show()

plt.matshow(final_model.w1.T)
plt.colorbar()
plt.savefig('w1_image.jpg')
plt.show()

plt.matshow(final_model.w2.T)
plt.colorbar()
plt.savefig('w2_image.jpg')
plt.show()


