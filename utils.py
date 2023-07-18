from os import path
import seaborn as sns
import pandas as pd
from types import NoneType
import pickle

DATA_FOLDER = "./data/"

def load_model_from_file(model_name, dataset_name):
    model_file_name = f"{dataset_name}-{model_name}"
    with open(f'{path.join(DATA_FOLDER, model_file_name)}.pkl', 'rb') as fid:
            model = pickle.load(fid) 
    return model

def plot_all(ds_name, ds_load_fn, models_dict):
    all_reg = pd.DataFrame()
    # _, X_test, _, _ = lds.load_data_set_wine()
    _, X_test, _, _ = ds_load_fn()
    # for reg_model_id, reg_fn in enumerate(cfcn_method):
    for reg_model_name in models_dict.keys():

        # print (20*'*', cfcn_method[reg_model_id],20*'*' )
        model = load_model_from_file(reg_model_name, ds_name)
        # pred = cfcn_model(X_train, X_test, y_train, y_test)
        # sc = StandardScaler()
        # X_test = sc.transform(X_test)
        pred = model.predict(X_test)
        # pred = model(cfcn_model, X_test)
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


# used to save either regression or classification models into files
def save_models(X_train, X_test, y_train, y_test, model_dict, dataset_name):
    
    for model_name, model_fn in model_dict.items():
        model = model_fn(X_train, X_test, y_train, y_test)
        model_file_name = f"{dataset_name}-{model_name}"
        with open(f'{path.join(DATA_FOLDER, model_file_name)}.pkl', 'wb') as fid:
            pickle.dump(model, fid) 