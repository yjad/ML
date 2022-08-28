import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from types import NoneType


def print_model_performance(model, y, prediction, reg_method_id):
    # print ('coeffecient:', model.coef_)
    # print ('intercept: ', model.intercept_)

    print (f'Mean squared Error (MSE): {mean_squared_error(y, prediction):.2f}')
    print (f'Mean absolute Error (MAE): {mean_absolute_error(y, prediction):.2f}')
    print (f'Coeffecint of determination (R2): {r2_score(y, prediction):.2f}')
    # plt.figure()
    # p= sns.regplot(x=y, y=prediction, marker = '+')
    # p= p.set_title(reg_method[reg_method_id])


reg_method= ['Random Forest_classifier',
            'SVC_classfier']

reg_functions = [use_forest_classifier,
                    SVC_classifier] 

def classification_all_models(train_X, val_X, train_y, val_y):

    all_reg = pd.DataFrame()
    for reg_model_id, reg_fn in enumerate(reg_functions):
        print (20*'*', reg_method[reg_model_id],20*'*' )
        pred = reg_fn(train_X, val_X, train_y, val_y)
        if type(pred) == NoneType: 
            print ('Model is not suitable for data')
            continue   # model is not suitable
        x = pd.DataFrame(val_y)
        x['predicted'] = pred
        x['Method'] = reg_method[reg_model_id]
        all_reg = pd.concat([all_reg, x])
        print_model_performance(None, val_y, pred, reg_model_id)

    # all_reg.to_csv(r'.\\out\\all_reg.csv')
    # g = sns.FacetGrid(data=all_reg, col= 'Method', col_wrap=2)
    # g.map(sns.regplot, all_reg.columns[0], 'predicted',  marker = '+')

# Apply standard scalling to get optimized results
def std_scalling(X_train, X_test): 
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


def use_forest_classifier(X_train, X_test, y_train, y_test):

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

    pred_rfc = rfc.predict(X_test)
    return pred_rfc


def SVC_classifier(X_train, X_test, y_train, y_test):

    X_train, X_test = std_scalling(X_train, X_test)

    rfc = SVC()
    try:
        rfc.fit(X_train, y_train)
    except Exception as err:
        print ('Error: from SVC_classifier: ', err)
        return None

    pred_rfc = rfc.predict(X_test)
    return pred_rfc