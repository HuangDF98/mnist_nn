1.	最终的参数选择
这个部分是为了查找参数的，我们在对类“nn”进行代码编写后，找寻机组学习率，隐藏层大小以及正则化的强度来进行最优的选择
学习率	3	1	0.5	0.05
隐藏层大小	50	100	200
正则化强度	1e-3	1e-5	1e-7
所以经过排列组合，一共有48种组合，每次组合都进行200次训练，然后通过找寻其中loss最小的定为最终的组合。

2.	类的编写
构建的类的名称为“nn”，编写在文件param.py中，其中有几个比较重要的部分组成。
2.1 激活函数
激活函数此处选用了softmax函数。
2.2 反向传播
其中有两个比较重要的板块，一个是损失函数loss function，还有一个是计算梯度的板块gradient compute。
在损失函数的板块中，用到了交叉熵函数，并且还要加上一个mu乘以L2正则化的数据，在这里就是w1和w2的平方的一半。
在梯度下降板块，根据公式Downstream Gradient = Local Gradient * Upstream Gradient反向求解，最终求得损失函数关于权重矩阵w1和w2的骗到。
2.3 学习率下降
    采取固定迭代次数学习率衰减的策略，具体实现在train函数里，下降率一开始设为0.99，之后每隔100次运行都会衰减一次。
2.4 SGD优化器
    随机选取几个数据进行梯度的反向传播与计算，这里定一个一个大小，取名为sample_size，即为样本的规模大小，同样在计算梯度的函数中实现。
2.5 对模型的参数进行保存
需要保存的参数为训练好的w1和w2的权重矩阵，使用numpy之中的savetxt的函数对权重矩阵进行保存，在函数store_values中进行实现，并在一开始的时候，设置了如果已经存在相应的txt文件，那么就不用random来得到随机的权重矩阵。
2.6 训练模型
利用一中的得到的最终的参数，对模型进行确认，并保存最终的参数，其中进行了500次的测试，以上步骤在model_train.py文件中进行。

3.	模型测试
将2.6中得到的最终的模型进行准确率的测试，得到测试集的训练精度，并进行输出，这些步骤在load_model.py文件中进行。

运行的顺序为param.py, model_train.py, load_model.py
