import mlsauce as ms 
from models.basemodel import BaseModel

class LSBoost(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)
        if args.objective == "regression":
            self.model = ms.LSBoostRegressor()
        elif args.objective == "classification":
            self.model = ms.LSBoostClassifier()

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params
