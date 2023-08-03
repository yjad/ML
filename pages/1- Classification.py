from os import path
import streamlit as st
from types import NoneType

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import sklearn_classifier as cls
import load_data_set as lds
import utils


    

def use_model(ds_name, ds_load_fn, models_dict):
    st.header("Use Model: " + st.session_state.cls_model_name)
    
    # reg_model_name = st.selectbox("Select Model:", cls.CFCN_MODELS.keys())
    X_train, X_test, y_train, y_test,ds = ds_load_fn()

    idx = st.slider("Select index from test set:",min_value=0,max_value=X_test.shape[0]-1)
    test_row = X_test[idx].reshape(1,-1)

    # ds_row = ds.iloc[index,:]
    # st.write(ds_row)
    
    # qlty= 'Bad' if ds_row['quality'] < 6.5 else 'Good'
    # st.write("Data Set Quality: ", qlty)
  
    
    if 'cls_model_name' not in st.session_state:
        st.warning("No model to use ...")
    else:
        cls_model = utils.load_model_from_file(st.session_state.cls_model_name, ds_name, 'cls')
    pred = utils.use_cls_model(cls_model, test_row)
    
    # # st.write("Actual Value   : ", y_test[index:index+1])
    # pred_qlty= 'Bad' if pred[0] < 6.5 else 'Good'
   
    # st.write("Predicted Value: ", pred_qlty)

    # if qlty != pred_qlty:
    #      st.error ("Quality Mismatch ....")
    # st.write(y_test.iloc[0])
    st.write("Test sample: ", test_row)
    test_qulity= 'Bad' if y_test.iloc[idx] == 0 else 'Good'
    pred_qulity= 'Bad' if pred[0] == 0 else 'Good'
    st.write(f"Test Quality: {test_qulity}")
    st.write(f"Predicted Quality: {pred_qulity}")
    if test_qulity == pred_qulity:
        st.info("Quality Matches ...")
    else:
        st.warning("Quality Mismatch ...")

def build_models(load_ds_fn):
    st.header("Build Model ...")
   
    select_opt = st.radio("Opions: ", 
                          ['Options', 
                           'Generate Models', 
                           'Best Model', 
                           'Models Performace'],
                    horizontal=True)

    match select_opt:
        case 'Generate Models':
            X_train, X_test, y_train, y_test,_ = load_ds_fn()
            out = cls.save_models(X_train, X_test, y_train, y_test, cls.CFCN_MODELS)
            st.info('Done')

        case 'Best Model':
            X_train, X_test, y_train, y_test,_ = load_ds_fn()
            perf, best_model, best_model_accuracy = cls.best_model(X_test, y_test)
            st.write(f"Best Model: {best_model} with accuracy: {best_model_accuracy}")
            st.write(perf)
            

        # case 'Models Performace':
        #     st.write(out)

     

st.header ("Classification")
datasets = {"...": None, 
           "Wine dataset": lds.load_data_set_wine, 
           "Boston House Data": None,   #lds.load_ds_boston_house, 
           "Germany Cars": None, 
           "Diabets": None, 
            }
selected_ds = st.sidebar.selectbox("Datasets: ", datasets.keys())
if datasets[selected_ds]:
    options = {"...": None, 
            "Save Models": None, 
            "Plot Models": None, 
            "Use Model": use_model}

    opt = st.sidebar.selectbox("Options", options.keys())
    match opt:
        case '...':
            pass
        case "Save Models": 
            X_train, X_test, y_train, y_test, _ = datasets[selected_ds]()
            utils.save_models(X_train, X_test, y_train, y_test, model_dict= cls.CLS_MODELS, 
                              dataset_name= selected_ds,
                              prefix='cls')
            st.info('Models generated and saved to files ...')
        
        case "Plot Models":
            fig, metrics = utils.plot_all(selected_ds, datasets[selected_ds], cls.CLS_MODELS, 'cls')
            if 'cls_model_name' not in st.session_state:
                st.session_state.cls_model_name = ''
            st.session_state.cls_model_name= metrics.iloc[0,0]   # model name of the first 

            st.dataframe(metrics, hide_index=True)
            st.pyplot(fig)
            # case others:
            #     options[opt]()
        case "Use Model":
            if 'cls_model_name' not in st.session_state:
                st.warning("No Model to use, select option 'Plot Models first'")
            else:
                use_model(selected_ds, datasets[selected_ds], cls.CLS_MODELS)




