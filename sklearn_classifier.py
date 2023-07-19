import io
# import os.path
from os import path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split 
from types import NoneType
import pickle

import load_data_set as lds
import utils as utl



# def print_model_performance(model, y, prediction, reg_method_id):
    # perf = {}
    # perf.update({'coeffecient:': model.coef_})
    # # print ('intercept: ', model.intercept_)

    # perf.update({'Mean squared Error (MSE)': mean_squared_error(y, prediction)})
    # print (f'Mean absolute Error (MAE): {mean_absolute_error(y, prediction):.2f}')
    # print (f'Coeffecint of determination (R2): {r2_score(y, prediction):.2f}')
    # plt.figure()
    # p= sns.regplot(x=y, y=prediction, marker = '+')
    # p= p.set_title(reg_method[reg_method_id])


# Apply standard scalling to get optimized results
def std_scalling(X_train, X_test): 
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


def use_random_forest(X_train, X_test, y_train, y_test):

    # # Apply standard scalling to get optimized results
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    X_train, X_test = std_scalling(X_train, X_test)
    
    rfc = RandomForestClassifier(n_estimators=200)
    try:
        rfc.fit(X_train, y_train)
    except Exception as err:
        print ('Error: from use_forest_classifier: ', err)
        return None

    # # pred_rfc = rfc.predict(X_test)
    # # return pred_rfc
    return rfc


def use_SVC(X_train, X_test, y_train, y_test):

    X_train, X_test = std_scalling(X_train, X_test)

    rfc = SVC()
    try:
        rfc.fit(X_train, y_train)
    except Exception as err:
        print ('Error: from SVC_classifier: ', err)
        return None

    # pred_rfc = rfc.predict(X_test)
    return rfc


def use_nueral_networks(X_train, X_test, y_train, y_test):
    mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=500)
    mlpc.fit(X_train, y_train)
    # pred = mlpc.predict(X_test)

    # classification_performance(y_test, pred)
    return mlpc




def out_string(*args, **kwargs):
    with io.StringIO() as output:
        print(*args, file=output, **kwargs)
        contents = output.getvalue()+ '\n'
    return contents



def best_model(X_test, y_test):

    perf = {}
    scores = []

    for reg_model_name, _ in CFCN_MODELS.items():
        # print (model_id,  reg_model_id, model_id and reg_model_id != model_id)
         with open(f'{path.join(DATA_FOLDER, reg_model_name)}.pkl', 'rb') as fid:
            cfn_model = pickle.load(fid) 

            # out_str += out_string(20*'*', cfcn_method[reg_model_id],20*'*')
            # cls_model = reg_fn(X_train, X_test, y_train, y_test)
            pred = cfn_model.predict(X_test)
            perf.update({f"{reg_model_name}-Classification Report": classification_report(y_test, pred)})
            perf.update({f"{reg_model_name}-Confusion Matrix     ": confusion_matrix(y_test, pred)})
            perf.update({f"{reg_model_name}-Accuracy Score       ": accuracy_score(y_test, pred)})
            scores.append(accuracy_score(y_test, pred))

    best_accuracy = max(scores)
    best_model = CLS_MODELS[scores.index(best_accuracy)]
    # out_str += out_string(20*'*', " Summary " , 20*'*')
    # out_str += out_string(f"Best model: {best_model} with accuracy: {best_accuracy:.2f}") 
    return perf, best_model, best_accuracy


# def use_model(cfn_model, X_test):

#     # with open(f'{os.path.join(DATA_FOLDER, model_name)}.pkl', 'rb') as fid:
#     #         cfn_model = pickle.load(fid) 
#     sc = StandardScaler()
#     # X_test = sc.transform(X_test)
#     pred = cfn_model.predict(X_test)
#     return pred

CLS_MODELS= {'Random Forest Calssification': use_random_forest, 
                  'SVC Calssification': use_SVC, 
                  'Nueral Network Calssification': use_nueral_networks}

if __name__ == '__main__':
    dataset_path = r"C:\Yahia\python\ML\data\WineQT.csv"
    X_train, X_test, y_train, y_test = load_data_set(dataset_path)

    # run_classification(X_train, X_test, y_train, y_test, model_id=None)
    pred_rfc = rfc.predict(X_test)
    # return pred_rfc