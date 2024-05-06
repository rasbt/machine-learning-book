import numpy as np

class Perceptron:
    """Perceptron classifier.
    
    Parameters
    --------
    eta:float
        Learning rate (between 0.0 and 1.0)
    n_iter:int
        Passes over the training dataset
    random_state:int
        Random number generator seed for random weight
        initialization.
    
    Attributes
    --------
    w_:1d-array
        Weights after fitting
    b_:Scalar
        Bias unit after fitting
    errors_:list
        Number of missclassifications (updates) in each epoch
            
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        """Fit training data.

        Parameters
        --------
        X:{array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y:array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------
        self:object
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors) 
            # For each iteration, in n_iter, you'd ideally want to 
            # see the number of errors in self.errors_ decline and
            # apporach zero, as our weights become more refined
            # with each iteration
        return self
    
    def net_input(self, X):
        """Calculate net input

        Parameters
        --------
        X:{array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
            
        """
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Return class label after unit step

        Parameters
        --------
        X:{array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    