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


# def plot_all():
#     fig = cls.plot_all()
#     st.pyplot(fig)



def use_model():
    pass


st.header ("Regression")
datasets = {"...": None, 
           "Boston House Data": lds.load_ds_boston_housing, 
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
            X_train, X_test, y_train, y_test = datasets[selected_ds]()
            cls.save_models(X_train, X_test, y_train, y_test, reg.REG_MODELS, selected_ds)
            st.info('Models generated and saved to files ...')
        case "Plot Models":
            fig = utils.plot_all(selected_ds, datasets[selected_ds], reg.REG_MODELS)
            st.pyplot(fig)
        case others:
            options[opt]()





