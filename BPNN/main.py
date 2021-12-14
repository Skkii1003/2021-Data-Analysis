import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets

#构建有1层隐藏层的MLP2，输入层有4个节点对应4个特征，隐藏层有10个节点，输出层有3个节点对应3种鸢尾花类别

# 1.初始化参数
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    # 设置权重和偏置矩阵
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 存储参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 2.前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # 通过前向传播来计算a2
    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)            # 使用tanh作为第一层的激活函数
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # 使用sigmoid作为第二层的激活函数

    # 通过字典存储参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache


# 3.计算代价函数
def compute_cost(a2, Y):
    m = Y.shape[1]      #总的样本数

    # 采用交叉熵作为代价函数
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m

    return cost


# 4.反向传播（计算代价函数的导数）
def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    # 反向传播，计算dw1、db1、dw2、db2
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


# 5.更新参数
def update_parameters(parameters, grads, learning_rate=0.4):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 更新参数
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 建立神经网络
def bpnn(X, Y, n_hide, n_input, n_output, times=5000):
    np.random.seed(3)

    n_x = n_input           # 输入层节点数
    n_y = n_output          # 输出层节点数

    # 1.初始化参数
    parameters = initialize_parameters(n_x, n_hide, n_y)

    # 梯度下降循环
    for i in range(0, times):
        # 2.前向传播
        a2, cache = forward_propagation(X, parameters)
        # 3.计算代价函数
        cost = compute_cost(a2, Y)
        # 4.反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 5.更新参数
        parameters = update_parameters(parameters, grads)

    return parameters


# 6.模型评估
def judge(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 结果的维度
    n_rows = a2.shape[0]
    n_cols = a2.shape[1]

    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0

    # 将one-hot编码反转为标签
    output = encoder.inverse_transform(output.T)
    output = output.reshape(1, output.shape[0])
    output = output.flatten()

    print('预测结果：', output)
    print('真实结果：', y_test)

    count = 0
    for k in range(0, n_cols):
        if output[k] == y_test[k]:
            count = count + 1
        else:
            print('错误分类样本序号：', k + 1)

    accuracy = count / int(a2.shape[1]) * 100
    print('BPNN精度：%.2f%%' % accuracy)
    return output


#随机抽取80%的训练集和20%的测试集
def divideData():
    completeData = np.c_[iris.data, iris.target.T]
    np.random.shuffle(completeData)
    trainData = completeData[range(int(length * 0.8)), :]
    testData = completeData[range(int(length * 0.8), length), :]
    return [trainData, testData]


if __name__ == "__main__":

    #获取iris数据集
    iris = datasets.load_iris()

    #随机获取80%训练集和20%测试集
    length = len(iris.target)
    [trainData, testData] = divideData()
    X=trainData[:,0:4].T
    Y=trainData[:,-1].T

    # 将标签转换为one-hot编码,便于计算
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(Y.shape[0], 1))
    Y = Y.toarray().T
    Y = Y.astype('uint8')

    # 输入4个节点，隐层10个节点，输出3个节点，迭代5000次
    parameters = bpnn(X, Y, n_hide=10, n_input=4, n_output=3, times=5000)

    x_test=testData[:,0:4].T
    y_test=testData[:,-1].T

    result = judge(parameters, x_test, y_test)
