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
    
    
    # _, X_test, _, y_test, ds = ds_load_fn()

    # idx = st.slider("Select index from test set:",min_value=0,max_value=X_test.shape[0]-1)

    # test_idx =  X_test.index[idx]
    # test_row = X_test.loc[test_idx:test_idx]

    
    # if 'reg_model_name' not in st.session_state:
    #     st.warning("No model to use ...")
    # else:
    #     reg_model = utils.load_model_from_file(st.session_state.reg_model_name, ds_name, 'reg')

    # pred = utils.use_reg_model(reg_model, test_row)

    # st.write("Test factores: ", test_row)
    # st.write(f"Test Value: {y_test.loc[test_idx]:.2f}")
    # st.write(f"Predicted Value: {pred[0]:.2f}") 
    
@st.cache_data
def load_dataset(selected_ds):
    global datasets
    # _, X_test, _, y_test, ds = datasets[selected_ds]()
    # _, X_test, _, y_test, ds = datasets[selected_ds]()  #load_ds_function()
    _, X_test, _, y_test, ds = lds.load_ds_germany_used_cars()  #load_ds_function()
    # st.write('b4 loading model from file ...')
    reg_model = utils.load_model_from_file(st.session_state.reg_model_name, selected_ds, 'reg')
    # st.write('after loading model from file ...')


    return X_test, y_test, ds, reg_model



st.header ("Regression")
datasets = {"...": None, 
           "Boston Housing Data": lds.load_ds_boston_housing, 
           "Germany Cars": lds.load_ds_germany_used_cars, 
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
            metrics =  metrics.sort_values(by="R2: Coeffecint of determination", ascending= False)
            st.session_state.reg_model_name=metrics.iloc[0,0]   # model name of the first 

            # st.write("reg Model Name: ", st.session_state.reg_model_name)
            st.dataframe(metrics, hide_index=True)
            st.pyplot(fig)
            
        case "Use Model":
            if 'reg_model_name' not in st.session_state:
                st.warning("No Model to use, select option 'Plot Models first'")
            else:
                st.header("Use Model: " + st.session_state.reg_model_name)
                # X_test, y_test, ds, reg_model = load_dataset(datasets[selected_ds], selected_ds)
                X_test, y_test, ds, reg_model = load_dataset(selected_ds)
                    
                    # reg_model = utils.load_model_from_file(st.session_state.reg_model_name, selected_ds, 'reg')

                    # use_model(selected_ds, datasets[selected_ds], reg.REG_MODELS)
                    # use_model(X_test, y_test, reg_model)
                idx = st.slider("Select index from test set:",min_value=0,max_value=X_test.shape[0]-1)

                test_idx =  X_test.index[idx]
                test_row = X_test.loc[test_idx:test_idx]
                # test_row = ds.loc[test_idx:test_idx]

                # st.write('B4 using reg model')
                pred = utils.use_reg_model(reg_model, test_row)
                # st.write('After using reg model')

                    # st.write("Test factores: ", test_row)
                st.write("Test factores: ", ds.loc[test_idx:test_idx])
                st.write(f"Test Value: {y_test.loc[test_idx]:.2f}")
                st.write(f"Predicted Value: {pred[0]:.2f}") 

        # case others:
        #     options[opt]()





