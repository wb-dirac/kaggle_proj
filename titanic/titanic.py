import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
import time
from scipy.stats import mode


def draw(df):
    print(df.tail())
    # marker = np.where(df['Sex'] == 'male', 'x', 'o')
    male = df[df['Sex'] == 'male']
    c = np.where(male['Survived'] == 1, 'green', 'red')
    plt.scatter(male['Age'], male['Fare'], c=c, marker='x')
    female = df[df['Sex'] == 'female']
    c = np.where(female['Survived'] == 1, 'green', 'red')
    plt.scatter(female['Age'], female['Fare'], c=c, marker='o')
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.show()

def process(df, isHot=True):
    def mapAge(x):
        age = 6
        if x > age:
            AgeC = 0
        elif x <= age:
            AgeC = 1
        else:
            AgeC = 2
        return AgeC
    df['AgeC'] = df['Age'].map(mapAge)
    def isNanMap(x):
        if x!=x:
            return 1
        else:
            return 0

    df['CabinIsNan'] = df['Cabin'].map(isNanMap)
    mode_embarked = mode(df[df['Embarked'] >= 'A']['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    df['family_size'] = df['SibSp'] + df['Parch']
    # def mapFare(x):
    #     if x > 50
    X=df[['Pclass', 'Sex', 'AgeC', 'CabinIsNan','Embarked','family_size', ]]
    class_le = LabelEncoder()
    X.loc[:, ['Sex']] = class_le.fit_transform(X['Sex'].values)
    #处理缺失数据
    # X = Imputer(missing_values='NaN', strategy='most_frequent', axis=0).fit_transform(X)
    if isHot:
        ohe = OneHotEncoder(categorical_features=[0, 2, 3])
        X=ohe.fit_transform(X).toarray()

    return X


def useLR(train_df, test_df):
    X = process(train_df)
    y=train_df['Survived']
    lr = LogisticRegression(C=10.0, random_state=0)
    # lr.predict_proba(X_test[0,:])
    fitAndTest(lr, X, y)
    X_test = process(test_df)
    predict(lr, X_test, "LR")


def fitAndTest(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    # lr.predict_proba(X_test[0,:])

    y_train_pred = model.predict(X_train)
    print('Train data Accuracy: %.2f' % metrics.accuracy_score(y_train, y_train_pred))
    y_pred = model.predict(X_test)
    print('Test Accuracy: %.2f' % metrics.accuracy_score(y_test, y_pred))
    cv_pred = sklearn.model_selection.cross_val_predict(model, X, y, cv=7)
    print('Cross validate Accuracy: %.2f' % metrics.accuracy_score(y, cv_pred))
    #用全部数据训练一次
    model.fit(X, y)
    y_pred = model.predict(X)

    print('All Accuracy: %.2f' % metrics.accuracy_score(y, y_pred))
    return y_pred

def useRF(train_df, test_df):

    X = process(train_df, isHot=False)
    y=train_df['Survived']
    # forest = RandomForestClassifier(criterion='entropy',n_estimators=100,random_state=10,n_jobs=2, oob_score=True)
    # fitAndTest(forest, X, y)
    # print(forest.oob_score_)
    # y_predprob = forest.predict_proba(X)[:, 1]
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
    X_test = process(test_df, isHot=False)
    # predict(forest, X_test, "RF")

    def fit_n_estimators():

        param_test1 = {'n_estimators': list(range(10, 71, 10))}
        gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                             min_samples_leaf=20, max_depth=8,
                                                             random_state=10),
                            param_grid=param_test1, scoring='roc_auc', cv=5)
        gsearch1.fit(X, y)
        print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    def fit_others():
        param_test2 = {'max_depth': list(range(3, 14, 2)), 'min_samples_split': list(range(50, 201, 20))}
        gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,
                                                                 min_samples_leaf=20, oob_score=True,
                                                                 random_state=10),
                                param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
        gsearch2.fit(X, y)
        print(gsearch2.grid_scores_)
        print(gsearch2.best_params_)
        print(gsearch2.best_score_)

    # fit_others()

    rf1 = RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_split=50,
                                 min_samples_leaf=20, oob_score=True, random_state=10)
    fitAndTest(rf1, X, y)
    print("oob score: %f" % rf1.oob_score_)
    y_predprob = rf1.predict_proba(X)[:, 1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
    predict(rf1, X_test, "RF")

def predict(model, X_test, model_name):
    results = model.predict(X_test)
    id_list = test_df["PassengerId"]
    output = pd.DataFrame({'PassengerId': id_list, "Survived": results})
    output.to_csv('./titanic/results/'+model_name + '_prediction_'+str(time.time())+'.csv', index=False)

class myNb:
    def __init__(self):
        self.cProb = {}
        self.cCondProb = {}
    def fit(self, X_train, y_train):
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


def useNb(train_df, test_df):
    nb = myNb()
    X = process(train_df, isHot=False)
    y = train_df['Survived']
    nb.fit(X, y)
    y_pred = nb.predict(X)
    print('Accuracy: %.2f' % metrics.accuracy_score(y, y_pred))
    X_test = process(test_df, False)
    predict(nb, X_test, 'MyNB')


def useMultinomialNB(X, y):
    del X['Fare']
    clf = MultinomialNB()
    fitAndTest(clf, X, y)


# nb(df)
# exit(0)
# draw(df)

# useMultinomialNB(X, y)
# model = useLR(X, y)
test_df = pd.read_csv('./titanic/data/test.csv', header=0)
train_df  = pd.read_csv('./titanic/data/train.csv', header=0)
# useRF(train_df, df_test)
# useLR(train_df, df_test)
# draw(train_df)

useNb(train_df, test_df)