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

    # # pred_rfc = rfc.predict(X_test)
    # # return pred_rfc
    return rfc
    return rfc


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