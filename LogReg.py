import numpy as np
def sigmoid(a):
    return 1 / (1 + np.exp(-a))
class LogReg:
    def __init__(self, eta = 0.01, max_iter = 100, reg =0.01, mini_batches_size = 100):
        self.eta = eta
        self.max_iter = max_iter #迭代次数
        self.mini_batches_size = mini_batches_size
        self.reg = reg

    def predict_proba(self, X):
        if self.w is None:
            print("You has not train params!")
            return None
        else:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
            p = sigmoid(np.dot(X, self.w))
            return np.hstack((1-p, p))

    def predict(self, X):
        p = self.predict_proba(X)
        return np.where(p.T[1] >= p.T[0], 1, 0)

    def get_mini_batches(self, X, y):
        random_idx = np.random.permutation(len(y))
        X_shuffled = X[random_idx, :]
        y_shuffled = y[random_idx]
        mini_batches = [(X_shuffled[i:i+self.mini_batches_size, :], y_shuffled[i:i+self.mini_batches_size]) \
                        for i in range(0, len(y_shuffled), self.mini_batches_size)]
        return mini_batches

    def fit(self, X, y):
        y = np.ravel(y)
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        #使用小批量随机梯度下降
        n,m = X.shape
        w = np.ones((m, 1))
        for i in range(0, self.max_iter):
            for samples in self.get_mini_batches(X, y):
                mini_X, mini_y = samples
                g = self.evaluate_gradient(mini_X, mini_y, w)
                # print('efficite gradient: ')
                # print(g)
                # print("numberical: ")
                # print(self.numberical_gradient(X[j], y[j], w))
                w = w - self.eta * g
        self.w = w


    def numberical_gradient(self, x, y, w):
        eta = 0.01
        e = np.eye(w.shape[0]) * (eta / 2)
        W1 = w - e
        W2 = w + e

        return (self.loss_func(x, y, W2) - self.loss_func(x, y, W1)) / eta
    def evaluate_gradient(self, x, y, w):
        """
        计算梯度
        :param x:
        :param y:
        :param w:
        :return:
        """
        y = np.reshape(y, (len(y), 1))
        return np.dot(x.T , (sigmoid(np.dot(x, w)) - y)) + self.reg * w

    def loss_func(self, x, y, w):
        """
        损失函数
        :param x:
        :param y:
        :param w:
        :return:
        """
        return -(y * np.log(sigmoid(np.dot(w.T , x))) + (1-y) * np.log((1-sigmoid(np.dot(w.T, x)))))

from matplotlib.colors import ListedColormap
def plotBestFit(classifier, X, y, resolution=0.02):
    """
    可视化决策区域
    :param classifier:
    :param X:
    :param y:
    :param resolution:
    :return:
    """
    import matplotlib.pyplot as plt
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    xcord1, ycord1 = X[np.where(y == 0)[0]].T
    xcord2, ycord2 = X[np.where(y == 1)[0]].T
    fig, ax = plt.subplots()
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 网格染色,画出决策区域
    x1_max, x1_min = X[:, 0].max() + 1, X[:, 0].min() - 1
    x2_max, x2_min = X[:, 1].max() + 1, X[:, 1].min() - 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), \
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def loadDataSet():
    import pandas as pd
    data = pd.read_csv('data/testSet.txt', header=None, delimiter='\t')
    X = data[[0, 1]].values
    y = data[[2]].values
    return X, y
if __name__ == '__main__':
    lr = LogReg(max_iter=100, reg = 0.01)
    # print(lr.predict(X))
    X, y = loadDataSet()
    print(y.shape)
    exit(0)
    lr.fit(X, y)
    y_pred = lr.predict_proba(X)
    from sklearn.metrics import log_loss
    from sklearn.metrics import accuracy_score
    print("my model acc: %.2f" % log_loss(y, y_pred))
    plotBestFit(lr, X, y)
    from sklearn.linear_model.logistic import LogisticRegression
    sk_lr = LogisticRegression(C=10)
    sk_lr.fit(X, y)
    sk_y_pred = sk_lr.predict_proba(X)

    print("sk model acc: %.2f" % log_loss(y, sk_y_pred))
    plotBestFit(sk_lr, X, y)
