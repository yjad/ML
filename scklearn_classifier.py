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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from types import NoneType
import pickle

DATA_FOLDER = "./data/"


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

def load_model_from_file(reg_model_name):
    with open(f'{path.join(DATA_FOLDER, reg_model_name)}.pkl', 'rb') as fid:
            cfn_model = pickle.load(fid) 
    return cfn_model

def plot_all():
    all_reg = pd.DataFrame()
    _, X_test, _, _ = load_data_set_wine()

    # for reg_model_id, reg_fn in enumerate(cfcn_method):
    for reg_model_name in CFCN_MODELS.keys():

        # print (20*'*', cfcn_method[reg_model_id],20*'*' )
        cfcn_model = load_model_from_file(reg_model_name)
        # pred = cfcn_model(X_train, X_test, y_train, y_test)
        pred = use_model(cfcn_model, X_test)
        if type(pred) == NoneType: 
            print ('Model is not suitable for data')
            continue   # model is not suitable
        x = pd.DataFrame(pred)
        x['predicted'] = pred
        x['Method'] = reg_model_name
        all_reg = pd.concat([all_reg, x])
        # print_model_performance(None, val_y, pred, reg_model_id)

    all_reg.to_csv(r'.\\out\\all_reg.csv')
    g = sns.FacetGrid(data=all_reg, col= 'Method', col_wrap=2)
    fig = g.map(sns.regplot, all_reg.columns[0], 'predicted',  marker = '+')
    return fig

def load_data_set_wine():
    #Loading dataset
    dataset_path = r".\data\wine-quality-white-and-red.zip"
    # dataset_path = r"C:\Yahia\Python\ML\data\wine-quality-white-and-red.csv"
    dataset = pd.read_csv(dataset_path)
    wine  = pd.read_csv(dataset_path)
    wine =wine.query("type == 'red'").drop(columns='type', axis=1)
    

    # Pre processing
    # wine = dataset.copy()
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    wine.quality = pd.cut(wine.quality, bins=bins, labels = group_names)
    # wine.quality.unique()
    label_quality = LabelEncoder()
    wine.quality = label_quality.fit_transform(wine.quality)

    #sns.countplot(x = wine.quality)

    X = wine.drop('quality', axis='columns')
    y = wine.quality
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42) 

    # Apply standard scalling to get optimized results
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test



def save_models(X_train, X_test, y_train, y_test, model_name=None):
    
    # X_train, X_test, y_train, y_test = load_data_set(ds)
    for reg_model_name, reg_model_fn in CFCN_MODELS.items():
        # print (model_id,  reg_model_id, model_id and reg_model_id != model_id)
        # if model_id is not None and (reg_model_id != model_id):
        #     continue

        cls_model = reg_model_fn(X_train, X_test, y_train, y_test)
        with open(f'{path.join(DATA_FOLDER, reg_model_name)}.pkl', 'wb') as fid:
            pickle.dump(cls_model, fid) 


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
    best_model = CFCN_MODELS[scores.index(best_accuracy)]
    # out_str += out_string(20*'*', " Summary " , 20*'*')
    # out_str += out_string(f"Best model: {best_model} with accuracy: {best_accuracy:.2f}") 
    return perf, best_model, best_accuracy


def use_model(cfn_model, X_test):

    # with open(f'{os.path.join(DATA_FOLDER, model_name)}.pkl', 'rb') as fid:
    #         cfn_model = pickle.load(fid) 
    sc = StandardScaler()
    # X_test = sc.transform(X_test)
    pred = cfn_model.predict(X_test)
    return pred

CFCN_MODELS= {'Random Forest Calssification': use_random_forest, 
                  'SVC Calssification': use_SVC, 
                  'Nueral Network Calssification': use_nueral_networks}

if __name__ == '__main__':
    dataset_path = r"C:\Yahia\python\ML\data\WineQT.csv"
    X_train, X_test, y_train, y_test = load_data_set(dataset_path)

    # run_classification(X_train, X_test, y_train, y_test, model_id=None)
    pred_rfc = rfc.predict(X_test)
    # return pred_rfc