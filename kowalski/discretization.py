from sklearn.tree import DecisionTreeClassifier



class DecisionTreeDiscretizer:


    def __init__(
        self,
        dt=DecisionTreeClassifier(max_depth=2, min_samples_leaf=10)
    ): 
        self.dt = dt

    def fit(self, X, y):
        """
        y is mandatorily discrete.
        """
        self.dt.fit(X, y)

    