from os import path
import streamlit as st
from types import NoneType
import scklearn_classifier as sc
import pandas as pd
import numpy as np
import pickle
import seaborn as sns


@st.cache_data
def load_data_set_wine():
    dataset_path = r"C:\Yahia\Home\Yahia-Dev\Python\ML\data\wine-quality-white-and-red.zip"
    st.write(dataset_path)
    
    # st.write(ds.columns)
    # return X_train, X_test, y_train, y_test,ds
    return sc.load_data_set_wine()


def plot_all():
    # all_reg = pd.DataFrame()
    # _, X_test, _, _ = sc.load_data_set_wine()

    # # for reg_model_id, reg_fn in enumerate(cfcn_method):
    # for reg_model_name in sc.CFCN_MODELS.keys():

    #     # print (20*'*', cfcn_method[reg_model_id],20*'*' )
    #     cfcn_model = sc.load_model_from_file(reg_model_name)
    #     # pred = cfcn_model(X_train, X_test, y_train, y_test)
    #     pred = sc.use_model(cfcn_model, X_test)
    #     if type(pred) == NoneType: 
    #         print ('Model is not suitable for data')
    #         continue   # model is not suitable
    #     x = pd.DataFrame(pred)
    #     x['predicted'] = pred
    #     x['Method'] = reg_model_name
    #     all_reg = pd.concat([all_reg, x])
    #     # print_model_performance(None, val_y, pred, reg_model_id)

    # all_reg.to_csv(r'.\\out\\all_reg.csv')
    # g = sns.FacetGrid(data=all_reg, col= 'Method', col_wrap=2)
    # fig = g.map(sns.regplot, all_reg.columns[0], 'predicted',  marker = '+')
    fig = sc.plot_all()
    st.pyplot(fig)
    # st.dataframe(X_test)
    

def load_dataset():
    st.write("Loadng the dataset ...")
    X_train, X_test, y_train, y_test, ds = load_data_set()
    st.write("Loading done ...")

# out = ""

# st.write(tab1, tab2)
# with tab1:
#     st.header("Build Classification Model")
   

def use_model():
    st.header("Use Model ...")
    
    reg_model_name = st.selectbox("Select Model:", sc.CFCN_MODELS.keys())
    X_train, X_test, y_train, y_test,ds = load_data_set()

    index = st.number_input("Select index from test set:",0,X_test.shape[0]-1)
    # index = 12
    test_row = X_test[index:index+1,:]
    st.write(test_row)
    ds_row = ds.iloc[index,:]
    st.write(ds_row)
    
    qlty= 'Bad' if ds_row['quality'] < 6.5 else 'Good'
    st.write("Data Set Quality: ", qlty)
  
    

    reg_model = load_model_from_file(reg_model_name)
    pred = sc.use_model(reg_model, test_row)
    # pred = sc.use_model(model_name, X_test[:1,:])
    
    # st.write("Actual Value   : ", y_test[index:index+1])
    pred_qlty= 'Bad' if pred[0] < 6.5 else 'Good'
   
    st.write("Predicted Value: ", pred_qlty)

    if qlty != pred_qlty:
         st.error ("Quality Mismatch ....")


def build_models(load_ds_fn):
    st.header("Build Model ...")
    # st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    # dataset_path = r"C:\Yahia\Home\Yahia-Dev\Python\ML\data\wine-quality-white-and-red.csv"
    # ds  = pd.read_csv(dataset_path)
    # st.write("Dataset path: ", dataset_path)
    # # st.write(ds.columns)
    # # st.write(ds.type.value_counts())
    # ds =ds.query("type == 'red'").drop(columns='type', axis=1)
    # # st.dataframe(ds)
    
   
    select_opt = st.radio("Opions: ", ['Options', 'Generate Models', 'Best Model', 'Models Performace'], horizontal=True)
    match select_opt:
        case 'Generate Models':
            X_train, X_test, y_train, y_test,_ = load_ds_fn()
            out = sc.save_models(X_train, X_test, y_train, y_test, model_id=None)
            st.info('Done')

        case 'Best Model':
            X_train, X_test, y_train, y_test,_ = load_ds_fn()
            perf, best_model, best_model_accuracy = sc.best_model(X_test, y_test)
            st.write(f"Best Model: {best_model} with accuracy: {best_model_accuracy}")
            st.write(perf)
            

        # case 'Models Performace':
        #     st.write(out)

     

st.header ("Classification/Regression")
datasets = {"...": None, 
           "Wine dataset": load_data_set_wine, 
           "Germany Cars": None, 
           "Diabets": None, 
            }
ds_load_function = st.sidebar.selectbox("Datasets: ", datasets.keys())
if datasets[ds_load_function]:
    options = {"...": None, 
            "Save Models": None, 
            "Plot Models": plot_all, 
            "Use Model": use_model}

    opt = st.sidebar.selectbox("Options", options.keys())
    match opt:
        case '...':
            pass
        case "Save Models": 
            X_train, X_test, y_train, y_test = datasets[ds_load_function]()
            sc.save_models(X_train, X_test, y_train, y_test, model_name=None)
            st.info('Models generated and saved to files ...')
        case others:
            options[opt]()




