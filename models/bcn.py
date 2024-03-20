import BCN as bcn 
from models.basemodel import BaseModel

class BCNEstimator(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)            
        if args.objective == "regression":
            self.model = bcn.BCNRegressor(**params,
                                 activation= "tanh",
                                 type_optim="nlminb",
                                 show_progress = True)
        elif args.objective == "classification":
            self.model = bcn.BCNClassifier(**params,
                                 activation= "tanh",
                                 type_optim="nlminb",
                                 show_progress = True)
        
    def fit(self, X, y, X_val=None, y_val=None):        
        self.model.fit(X, y)
        return [], []
    
    def predict_proba(self, X):
        return super().predict_proba(X)
    
    def predict(self, X):
        return super().predict(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        # B = int(x[0]),
        # nu = 10**x[1],
        # lam = 10**x[2],
        # r = 1 - 10**x[3],
        # tol = 10**x[4],
        # col_sample = np.ceil(x[5]),
        # n_clusters = np.ceil(x[6]))
        # lower_bound = np.array([   3,    -6, -10, -10,   -6, 0.8, 1]),
        # upper_bound = np.array([ 100,  -0.1,  10,  -1, -0.1,   1, 4]),                    
        params = {
            "B": trial.suggest_int("B", 3, 10, log=True),
            "nu": trial.suggest_float("nu", 1e-6, 10**-0.1, log=True),
            "lam": trial.suggest_float("lam", 1e-10, 1e10, log=True),
            "r": trial.suggest_float("r", 1 - 1e-1, 1 - 1e-10, log=True),
            "tol": trial.suggest_float("tol", 1e-6, 10**-0.1, log=True),
            "col_sample": trial.suggest_float("col_sample", 0.8, 1, log=True),
            "n_clusters": trial.suggest_int("n_clusters", 2, 4)
        }
        return params

