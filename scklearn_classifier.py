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
#     # print ('coeffecient:', model.coef_)
#     # print ('intercept: ', model.intercept_)

#     print (f'Mean squared Error (MSE): {mean_squared_error(y, prediction):.2f}')
#     print (f'Mean absolute Error (MAE): {mean_absolute_error(y, prediction):.2f}')
#     print (f'Coeffecint of determination (R2): {r2_score(y, prediction):.2f}')
#     # plt.figure()
#     # p= sns.regplot(x=y, y=prediction, marker = '+')
#     # p= p.set_title(reg_method[reg_method_id])

def classification_performance(y_test, pred):
    # model performance
    cr = classification_report(y_test, pred)
    print ('classification_report: \n', cr)
    cm = confusion_matrix(y_test, pred)
    print ('confusion_matrix: for each classification, how may correct, how many incorrect:\n', cm )

    print ('Accuracy score:', accuracy_score(y_test, pred))


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

    # pred_rfc = rfc.predict(X_test)
    # return pred_rfc
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

def run_classification(X_train, X_test, y_train, y_test, model_id=None):
    cfcn_method= ['Random Forest Calssification', 
                  'SVC Calssification', 
                  'Nueral Network Calssification']

    cfcn_functions = [use_random_forest, 
                    use_SVC,
                    use_nueral_networks] 
    
    scores = []
    for reg_model_id, reg_fn in enumerate(cfcn_functions):
        # print (model_id,  reg_model_id, model_id and reg_model_id != model_id)
        if model_id is not None and (reg_model_id != model_id):
            continue
        print (20*'*', cfcn_method[reg_model_id],20*'*' )
        cls_model = reg_fn(X_train, X_test, y_train, y_test)
        pred = cls_model.predict(X_test)
        classification_performance(y_test, pred)
        # --- save model
        with open(f'{path.join(DATA_FOLDER, cfcn_method[reg_model_id])}.pkl', 'wb') as fid:
            pickle.dump(cls_model, fid) 
    # random_forest(X_train, X_test, y_train, y_test)
    # run_SVC(X_train, X_test, y_train, y_test)
    # run_nueral_networks(X_train, X_test, y_train, y_test)
        scores.append(accuracy_score(y_test, pred))
    best_accuracy = max(scores)
    best_model = cfcn_method[scores.index(best_accuracy)]
    print("******************* Summary ****************")
    print (f"Best model: {best_model} with accuracy: {best_accuracy:.2f}" )

# def plot_all():
#     all_reg = pd.DataFrame()
#     for reg_model_id, reg_fn in enumerate(reg_functions):
#         print (20*'*', reg_method[reg_model_id],20*'*' )
#         pred = reg_fn(train_X, val_X, train_y, val_y)
#         if type(pred) == NoneType: 
#             print ('Model is not suitable for data')
#             continue   # model is not suitable
#         x = pd.DataFrame(val_y)
#         x['predicted'] = pred
#         x['Method'] = reg_method[reg_model_id]
#         all_reg = pd.concat([all_reg, x])
#         print_model_performance(None, val_y, pred, reg_model_id)

#     all_reg.to_csv(r'.\\out\\all_reg.csv')
#     g = sns.FacetGrid(data=all_reg, col= 'Method', col_wrap=2)
#     g.map(sns.regplot, all_reg.columns[0], 'predicted',  marker = '+')


def load_data_set(file_path):
    #Loading dataset
    dataset = pd.read_csv(file_path)

    # Pre processing
    wine = dataset.copy()
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

    
if __name__ == '__main__':
    dataset_path = r"C:\Yahia\python\ML\data\WineQT.csv"
    X_train, X_test, y_train, y_test = load_data_set(dataset_path)

    run_classification(X_train, X_test, y_train, y_test, model_id=None)