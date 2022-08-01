import argparse
import os 
from utils.common_utils import (read_params, create_dirs, clean_prev_dirs_if_exists,
                                 get_data, save_models)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
# import optuna
# from optuna.integration.mlflow import MLflowCallback
import json

class Train():
    def __init__(self, config_path):
        self.config = read_params(config_path)

        self.split_data_dir = self.config['artifacts']['04_split_data']['split_data_dir']

        self.train_dir = self.config['artifacts']['05_train']['train_dir']
        clean_prev_dirs_if_exists([self.train_dir])
        create_dirs([self.train_dir])
                    
        self.random_state = self.config['base']['random_state']

         

    def __get_best_balanced_random_forest_classifier(self, x_train, y_train):
        try:
            print("balanced_random_forest_classifier")
            print()
            # Random Forest GridSearchCV params
            cv = self.config['hyperparams']['balanced_random_forest']['cv']
            verbose = self.config['hyperparams']['balanced_random_forest']['verbose']
            n_estimators = self.config['hyperparams']['balanced_random_forest']['n_estimators']
            criterion = self.config['hyperparams']['balanced_random_forest']['criterion']
            max_depth = self.config['hyperparams']['balanced_random_forest']['max_depth']
            max_features = self.config['hyperparams']['balanced_random_forest']['max_features']

            clf = BalancedRandomForestClassifier()
            param_grid = {  "random_state" : self.random_state,
                            "n_estimators": n_estimators,
                            "criterion": criterion,
                            "max_depth": range(max_depth[0], max_depth[1], max_depth[2]),
                            "max_features": max_features
                        }

            grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv,  verbose=verbose)
            #finding the best parameters
            grid.fit(x_train, y_train)

            #extracting the best parameters
            params = dict()
            best_criterion = grid.best_params_['criterion']
            best_max_depth = grid.best_params_['max_depth']
            best_max_features = grid.best_params_['max_features']
            best_n_estimators = grid.best_params_['n_estimators']

            params = { "best_criterion":best_criterion,
                         "best_max_depth":best_max_depth,
                         "best_max_features":best_max_features, 
                         "best_n_estimators":best_n_estimators
                         }

            #creating a new model with the best parameters
            brfc = BalancedRandomForestClassifier(n_estimators=best_n_estimators,
                                                criterion=best_criterion,
                                                max_depth=best_max_depth,
                                                max_features=best_max_features)
            # training the mew model
            brfc.fit(x_train, y_train)
            return brfc, params
        except Exception as e:
           raise e
    
    def __get_best_balanced_bagging_classifier(self, x_train, y_train):
        try:
            print("balanced_bagging_classifier")
            print()
            
            # Random Forest GridSearchCV params
            cv = self.config['hyperparams']['balanced_bagging_classifer']['cv']
            verbose = self.config['hyperparams']['balanced_bagging_classifer']['verbose']
            n_estimators = self.config['hyperparams']['balanced_bagging_classifer']['n_estimators']
            sampling_strategy = self.config['hyperparams']['balanced_bagging_classifer']['sampling_strategy']
            replacement = self.config['hyperparams']['balanced_bagging_classifer']['replacement']
            
            clf = BalancedBaggingClassifier()
            param_grid = {  
                            "base_estimator" : [DecisionTreeClassifier()],
                            "random_state" : self.random_state,
                            "n_estimators": n_estimators,
                            "sampling_strategy" : sampling_strategy,
                            "replacement" : replacement
                        }

            grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv,  verbose=verbose)
            #finding the best parameters
            grid.fit(x_train, y_train)

            #extracting the best parameters
            best_replacement = grid.best_params_['replacement']
            best_sampling_strategy = grid.best_params_['sampling_strategy']
            best_n_estimators = grid.best_params_['n_estimators']

            params = {   "best_replacement":best_replacement,
                         "best_sampling_strategy":best_sampling_strategy, 
                         "best_n_estimators":best_n_estimators 
                         }
            #creating a new model with the best parameters
            bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                            n_estimators=best_n_estimators,
                                            sampling_strategy=best_sampling_strategy,
                                            replacement=best_replacement,
                                           )
            bbc.fit(x_train, y_train) 

            return bbc, params
        except Exception as e:
           raise e

    def __get_best_EasyEnsembleClassifier(self, x_train, y_train):
        try:
            print("EasyEnsembleClassifier")
            print()
            # Random Forest GridSearchCV params
            cv = self.config['hyperparams']['easy_ensemble_classifier']['cv']
            verbose = self.config['hyperparams']['easy_ensemble_classifier']['verbose']
            n_estimators = self.config['hyperparams']['easy_ensemble_classifier']['n_estimators']
            sampling_strategy = self.config['hyperparams']['easy_ensemble_classifier']['sampling_strategy']
            replacement = self.config['hyperparams']['easy_ensemble_classifier']['replacement']
            
            clf = EasyEnsembleClassifier()
            param_grid = {  
                            "random_state" : self.random_state,
                            "n_estimators": n_estimators,
                            "sampling_strategy" : sampling_strategy,
                            "replacement" : replacement
                        }

            grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv,  verbose=verbose)
            #finding the best parameters
            grid.fit(x_train, y_train)

            #extracting the best parameters
            best_replacement = grid.best_params_['replacement']
            best_sampling_strategy = grid.best_params_['sampling_strategy']
            best_n_estimators = grid.best_params_['n_estimators']

            
            params = {   "best_replacement":best_replacement,
                         "best_sampling_strategy":best_sampling_strategy, 
                         "best_n_estimators":best_n_estimators 
                         }
            #creating a new model with the best parameters
            eec = EasyEnsembleClassifier(          # base_estimator --> default : AdaBoost
                                            n_estimators=best_n_estimators,
                                            sampling_strategy=best_sampling_strategy,
                                            replacement=best_replacement,
                                           )
            eec.fit(x_train, y_train) 

            return eec, params
        except Exception as e:
           raise e

    def train_models(self):
        try:  
            self.X_train = get_data(os.path.join(self.split_data_dir, "X_train.csv"))
            # self.X_test = get_data(os.path.join(self.split_data_dir, "X_test.csv"))
            self.Y_train = get_data(os.path.join(self.split_data_dir, "Y_train.csv")).values.ravel()
            # self.Y_test = get_data(os.path.join(self.split_data_dir, "Y_test.csv")).values.ravel()

            # Easy Ensemble Classifier
            eec_clf, eec_best_params = self.__get_best_EasyEnsembleClassifier(self.X_train, self.Y_train)
            eec_model_name = self.config['hyperparams']['easy_ensemble_classifier']['model_name']

            # Balanced Random Forest Classfier
            brf_clf, brf_best_params = self.__get_best_balanced_random_forest_classifier(self.X_train, self.Y_train)
            brf_model_name = self.config['hyperparams']['balanced_random_forest']['model_name']

            # Balanced Bagging Classifier
            bbc_clf, bbc_best_params = self.__get_best_balanced_bagging_classifier(self.X_train, self.Y_train)
            bbc_model_name = self.config['hyperparams']['balanced_bagging_classifer']['model_name']

            

            save_models([[brf_clf, os.path.join(self.train_dir, brf_model_name)],
                         [bbc_clf, os.path.join(self.train_dir, bbc_model_name)],
                         [eec_clf, os.path.join(self.train_dir, eec_model_name)]])

            print("###"*40)
            print("==="*15, " BalancedRandomForest Best  Params ", "==="*15)
            print(brf_best_params, "type", type(brf_best_params))
            print("==="*15, " BalancedBaggingClassifer Best Params ", "==="*15)
            print(bbc_best_params, "type", type(bbc_best_params))
            print("==="*15, " EasyEnsembleClassifier Best Params", "==="*15)
            print(eec_best_params, "type", type(eec_best_params))
            print("###"*40)
            
            
            best_params = dict()
            best_params.update({"eec_best_params" : eec_best_params})
            best_params.update({"brf_best_params" : brf_best_params})
            best_params.update({"bbc_best_params" : bbc_best_params})

            with open(os.path.join(self.train_dir, "best_params.json"), 'w') as f:
                json.dump(best_params, f, indent=4)

           

        except Exception as e:
           raise e

if __name__ == '__main__':
    print("\n", 10*"===", " 05 Training Stage ", 10*"===", "\n")
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()

    try:      
        obj = Train(config_path=parsed_args.config)
        obj.train_models()
    except Exception as e:
        raise e