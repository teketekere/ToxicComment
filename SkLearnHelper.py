# class for extending sk-learn classification
class SklearnClassifierHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def score(self, x, y):
        return self.clf.score(x, y)

    def fit(self, x, y, dummyx=None, dummyy=None):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


# class for extending sk-learn regression
class SklearnRegressorHelper(object):
    def __init__(self, clf, seed=0, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def score(self, x, y):
        return self.clf.score(x, y)

    def fit(self, x, y, dummyx=None, dummyy=None):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

