from os import path
import streamlit as st
from types import NoneType

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import sklearn_classifier as cls
import sklearn_reg as reg
import load_data_set as lds
import utils


def use_model(ds_name, ds_load_fn, models_dict):
    st.header("Use Model: " + st.session_state.reg_model_name)
    
    
    X_train, X_test, y_train, y_test,ds = ds_load_fn()

    idx = st.slider("Select index from test set:",min_value=0,max_value=X_test.shape[0]-1)

    test_idx =  X_test.index[idx]
    # test_idx
    test_row = X_test.loc[test_idx:test_idx]

    
    if 'reg_model_name' not in st.session_state:
        st.warning("No model to use ...")
    else:
        reg_model = utils.load_model_from_file(st.session_state.reg_model_name, ds_name, 'reg')

    # X_test.iloc[index,:]
    pred = utils.use_reg_model(reg_model, test_row)
    # pred = utils.use_reg_model(reg_model, X_test[:1,:])
    
    # # st.write("Actual Value   : ", y_test[index:index+1])
    # pred_qlty= 'Bad' if pred[0] < 6.5 else 'Good'
   
    # st.write("Predicted Value: ", pred_qlty)

    # if qlty != pred_qlty:
    #      st.error ("Quality Mismatch ....")
    st.write("Test factores: ", test_row)
    st.write(f"Test Value: {y_test.loc[test_idx]:.2f}")
    st.write(f"Predicted Value: {pred[0]:.2f}") 
    


st.header ("Regression")
datasets = {"...": None, 
           "Boston Housing Data": lds.load_ds_boston_housing, 
           "Germany Cars": None, 
           "Diabets": None, 
            }
selected_ds = st.sidebar.selectbox("Datasets: ", datasets.keys())
if datasets[selected_ds]:
    options = {"...": None, 
            "Save Models": None, 
            "Plot Models": None, 
            "Use Model": None}

    opt = st.sidebar.selectbox("Options", options.keys())
    match opt:
        case '...':
            pass
        case "Save Models": 
            try:
                X_train, X_test, y_train, y_test, _ = datasets[selected_ds]()
                utils.save_models(X_train, X_test, y_train, y_test, reg.REG_MODELS, selected_ds, 'reg')
                st.info('Models generated and saved to files ...')
            except Exception as err:
                st.error("Error loadding data set: "+ str(err))
            
        case "Plot Models":
            fig, metrics = utils.plot_all(selected_ds, datasets[selected_ds], reg.REG_MODELS, 'reg')
            if 'reg_model_name' not in st.session_state:
                st.session_state.reg_model_name = ''
            st.session_state.reg_model_name= metrics.sort_values(by='Model Name', ascending= False).iloc[0,0]   # model name of the first 

            # st.write("reg Model Name: ", st.session_state.reg_model_name)
            st.dataframe(metrics, hide_index=True)
            st.pyplot(fig)
            
        case "Use Model":
            if 'reg_model_name' not in st.session_state:
                st.warning("No Model to use, select option 'Plot Models first'")
            else:
                use_model(selected_ds, datasets[selected_ds], reg.REG_MODELS)
        # case others:
        #     options[opt]()





