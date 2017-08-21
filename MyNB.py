class myNb:
    def __init__(self):
        self.cProb = {}
        self.cCondProb = {}
    def fit(self, X_train, y_train):
        import numpy as np

        # 生还与否的概率
        t = len(y_train)
        # 生还和死者总数
        cNumber = {}
        cValues = np.unique(y_train)
        #类概率
        for cv in cValues:
            cNumber[cv] = (y_train == cv).sum()
            self.cProb[cv] = cNumber[cv] / t
        #求类条件概率
        for col in X_train:
            if col not in self.cCondProb:
                self.cCondProb[col] = {}
            for cv in self.cProb:
                if cv not in self.cCondProb[col]:
                    self.cCondProb[col][cv] = {}
                for fv in np.unique(X_train[col]):
                    self.cCondProb[col][cv][fv] = ((X_train[col] == fv) & (y_train == cv)).sum() / cNumber[cv]
                    if self.cCondProb[col][cv][fv] == 0:
                        self.cCondProb[col][cv][fv] = 1e-10

    def predict(self, X_test):
        y_pred = []
        P = {}
        Plog = {}
        for cv in self.cProb:
            Plog[cv] = np.log(self.cProb[cv])
        for i in range(0, len(X_test)):
            for cv in self.cProb:
                P[cv] = Plog[cv]
                for col in X_test:
                    if self.cCondProb[col][cv][X_test.loc[i][col]] == 0:
                        print(i, col)
                    P[cv] += np.log(self.cCondProb[col][cv][X_test.loc[i][col]])
            y_pred.append(max(P, key=P.get))
        return y_pred

    # def predict_proba(self, X_test):