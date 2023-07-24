from os import path
import seaborn as sns
import pandas as pd
from types import NoneType
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error   # reg
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # classification
from sklearn.preprocessing import StandardScaler
import pickle

DATA_FOLDER = "./data/"

def load_model_from_file(model_name, dataset_name, prefix):
    model_file_name = f"{prefix}_{dataset_name}-{model_name}"
    with open(f'{path.join(DATA_FOLDER, model_file_name)}.pkl', 'rb') as fid:
            model = pickle.load(fid) 
    return model

def plot_all(ds_name, ds_load_fn, models_dict, prefix):
    all_reg = pd.DataFrame()
    metrics=[]
    # _, X_test, _, _ = lds.load_data_set_wine()
    _, X_test, _, y_test, _ = ds_load_fn()
    # for reg_model_id, reg_fn in enumerate(cfcn_method):
    for model_name in models_dict.keys():

        # print (20*'*', cfcn_method[reg_model_id],20*'*' )
        model = load_model_from_file(model_name, ds_name, prefix)
        # pred = cfcn_model(X_train, X_test, y_train, y_test)
        # if prefix == 'cls': # classification
        #     sc = StandardScaler()
        #     X_test = sc.transform(X_test)
        pred = model.predict(X_test)
        # pred = model(cfcn_model, X_test)
        if type(pred) == NoneType: 
            print ('Model is not suitable for data')
            continue   # model is not suitable
        x = pd.DataFrame(y_test)
        x['predicted'] = pred
        x['Method'] = model_name
        all_reg = pd.concat([all_reg, x])
        if prefix == 'reg':
            metrics.append([model_name,
                        mean_squared_error(y_test, pred),
                        mean_absolute_error(y_test, pred),
                        r2_score(y_test, pred)])
            metrics_columns = \
                ["Model Name", "MSE: Mean squared Error", "MAE: Mean absolute Error", "R2: Coeffecint of determination"]
            accuracy_col_idx = 2
        else:
            cm = confusion_matrix(y_test, pred)
            cls_rep = classification_report(y_test, pred)
            # cls_rep_dict = classification_report(y_test, pred, output_dict=True)
            metrics.append([model_name, 
                            cls_rep, 
                            f"[{cm[0][0]:4d}, {cm[0][1]:4d}] - [{cm[1][0]:4d}, {cm[1][1]:4d}]", 
                            accuracy_score(y_test, pred)])
            metrics_columns = \
                ["Model Name", "classification_report", "confusion_matrix", "accuracy_score"]
            accuracy_col_idx = 3
            # print (crep, type(crep))
    # all_reg.to_csv(r'.\\out\\all_reg.csv')
    g = sns.FacetGrid(data=all_reg, col= 'Method', col_wrap=2)
    fig = g.map(sns.regplot, all_reg.columns[0], 'predicted',  marker = '+')

    metrics = pd.DataFrame(
        data=metrics, columns= metrics_columns).\
        sort_values(by=metrics_columns[accuracy_col_idx], ascending=False)
    # model_name_to_use = metrics.loc[0,0]
    return fig, metrics

# used to save either regression or classification models into files
def save_models(X_train, X_test, y_train, y_test, model_dict, dataset_name, prefix):
    
    for model_name, model_fn in model_dict.items():
        model = model_fn(X_train, X_test, y_train, y_test)

        model_file_name = f"{prefix}_{dataset_name}-{model_name}"
        with open(f'{path.join(DATA_FOLDER, model_file_name)}.pkl', 'wb') as fid:
            pickle.dump(model, fid) 


def use_reg_model(reg_model, X_test):

    pred = reg_model.predict(X_test)
    return pred


def use_cls_model(cfn_model, X_test):

    # sc = StandardScaler()
    # X_test = sc.transform(X_test)
    pred = cfn_model.predict(X_test)
    return pred

def check_non_numeric(s:pd.Series):
    # print (s.dtype)
    if 'pyarrow' in str(s.dtype):
        idx = pd.to_numeric(s, errors='coerce', dtype_backend='pyarrow').isna()
    else:
        idx = pd.to_numeric(s, errors='coerce').isna()
    return idx  # return index of non numeric
