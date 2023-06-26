from os import path
import streamlit as st
import scklearn_classifier as sc
import pandas as pd
import numpy as np
import pickle

@st.cache_data
def load_data_set():
    st.header("Build Classification Model")
    # st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    dataset_path = r"C:\Yahia\Home\Yahia-Dev\Python\ML\data\wine-quality-white-and-red.csv"
    ds  = pd.read_csv(dataset_path)
    st.write("Dataset path: ", dataset_path)
    # st.write(ds.columns)
    # st.write(ds.type.value_counts())
    ds =ds.query("type == 'red'").drop(columns='type', axis=1)
    # st.dataframe(ds)
    X_train, X_test, y_train, y_test = sc.load_clsfn_data_set(ds)

    # st.write(ds.columns)
    return X_train, X_test, y_train, y_test,ds

@st.cache_data
def load_model_from_file(reg_model_name):
    with open(f'{path.join(sc.DATA_FOLDER, reg_model_name)}.pkl', 'rb') as fid:
            cfn_model = pickle.load(fid) 
    return cfn_model


st.title ("Classification")
tab1, tab2 = st.tabs(["Build Model", "Use Model"])
st.write("Loadng the dataset ...")
X_train, X_test, y_train, y_test, ds = load_data_set()
st.write("Loading done ...")
out = ""

# st.write(tab1, tab2)
with tab1:
    st.header("Build Classification Model")
    # st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    # dataset_path = r"C:\Yahia\Home\Yahia-Dev\Python\ML\data\wine-quality-white-and-red.csv"
    # ds  = pd.read_csv(dataset_path)
    # st.write("Dataset path: ", dataset_path)
    # # st.write(ds.columns)
    # # st.write(ds.type.value_counts())
    # ds =ds.query("type == 'red'").drop(columns='type', axis=1)
    # # st.dataframe(ds)
    # X_train, X_test, y_train, y_test = sc.load_data_set(ds)
   

    # out = sc.run_classification(X_train, X_test, y_train, y_test, model_id=None)
    # select_opt = st.radio("Opions: ", ['Options', 'Generate Models', 'Best Model', 'Models Performace'], horizontal=True)
    # match select_opt:
    #     case 'Generate Models':
    #         sc.genetate_models(ds)
    #         st.info('Done')

    #     case 'Best Model':
    #         out, best_model, best_model_accuracy = sc.best_model(X_test, y_test)
    #         st.write(f"Best Model: {best_model} with accuracy: {best_model_accuracy}")
            

    #     case 'Models Performace':
    #         st.write(out)

with tab2:
    st.header("Use Model")
    cfcn_method= ['Random Forest Calssification', 
                  'SVC Calssification', 
                  'Nueral Network Calssification']
    reg_model_name = st.selectbox("Select Model:", cfcn_method)


    index = st.number_input("Select index from test set:",0,X_test.shape[0]-1)
    # index = 12
    # st.write(X_test[index:index+1,:])
    # st.write(X_test[index:index+1,:])
    row = ds.iloc[index,:]
    st.write(row)
    
    qlty= 'Bad' if row['quality'] < 6.5 else 'Good'
    st.write("Quality: ", qlty)
  
    

    reg_model = load_model_from_file(reg_model_name)
    pred = sc.use_model(reg_model, X_test[index:index+1,:])
    # pred = sc.use_model(model_name, X_test[:1,:])
    
    # st.write("Actual Value   : ", y_test[index:index+1])
    pred_qlty= 'Bad' if pred[0] < 6.5 else 'Good'
   
    st.write("Predicted Value: ", pred_qlty)

    if qlty != pred_qlty:
         st.error ("Quality Mismatch ....")