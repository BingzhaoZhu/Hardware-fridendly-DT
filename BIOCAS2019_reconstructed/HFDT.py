'''Hardware-friendly decision trees:
    ----with cost-efficient gradient boosting
    and ---- leaf value quantization'''
import numpy as np
import lightgbm as lgb
import model_cost as cost
class HFDT:
    def __init__(self, num_trees, max_depth, CEGB_tradeoff=0, learning_rate=0.1, num_bits_leaf=0):
        self.num_trees=num_trees
        self.max_depth = max_depth
        self.CEGB_tradeoff = CEGB_tradeoff
        self.learning_rate = learning_rate
        self.num_bits_leaf = num_bits_leaf

    def _get_params(self,feature_cost):
        self.feature_cost = feature_cost
        params = {
            'max_depth': self.max_depth,
            'verbose': -1,
            'cegb_penalty_split': 0,
            'cegb_penalty_feature_lazy': feature_cost,
            'cegb_tradeoff': self.CEGB_tradeoff,
            'learning_rate': self.learning_rate,
            'objective': 'binary'
        }
        return params

    def _quantization_retrain(self,num_bits, params, lgb_train):
        step = cost.quantization(num_bits)
        gbm = lgb.train(params,
                        lgb_train,
                        init_model='quan_model.txt',
                        num_boost_round=1
                        )
        return gbm, step

    def fit(self,tr_X,tr_Y,feature_cost):
        n,f=tr_X.shape
        if not f==len(feature_cost):
            raise ValueError('feature cost should have the same length with the number of features')
        lgb_train = lgb.Dataset(tr_X, tr_Y, free_raw_data=False)
        params = self._get_params(feature_cost)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1
                        )
        gbm.save_model('model.txt')
        for n_round in range(self.num_trees-1):
            gbm, step = self._quantization_retrain(self.num_bits_leaf, params, lgb_train)
            lgb_train.init_score = None
            gbm.save_model('model.txt', num_iteration=gbm.num_trees())
        self.step = cost.quantization(self.num_bits_leaf)
        self.model = lgb.Booster(model_file='quan_model.txt')

    def predict(self,te_X,threshold=0.5):
        y_pred=self.model.predict(te_X)
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        return y_pred

    def get_f1_score(self,y_pred,te_Y):
        from sklearn.metrics import f1_score
        return f1_score(te_Y,y_pred)

    def get_cost(self,X):
        try:
            return cost.cost(X, self.feature_cost , 'model.txt', 100)
        except:
            raise ValueError('Have you trained the model?')