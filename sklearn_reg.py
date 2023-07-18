import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import seaborn as sns


def print_model_performance(model, y, prediction, id):
    # print ('coeffecient:', model.coef_)
    # print ('intercept: ', model.intercept_)
    print (f'Mean squared Error (MSE): {mean_squared_error(y, prediction):.2f}')
    print (f'Mean absolute Error (MAE): {mean_absolute_error(y, prediction):.2f}')
    print (f'Coeffecint of determination (R2): {r2_score(y, prediction):.2f}')
    print (f'Accuracy: {accuracy_score(y_true=y, y_pred=prediction)}')
    # sns.scatterplot(x=y, y=prediction)


def reg_decision_tree_1(X_train, X_test, y_train, ty_test):
    # Specify Model
    model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    model.fit(X_train, y_train)
    return model
    # Make validation predictions and calculate mean absolute error
    # val_predictions = iowa_model.predict(val_X)
    # val_mae = mean_absolute_error(val_predictions, val_y)
    # print("Validation MAE: {:,.0f}".format(val_mae))
    # print_model_performance(iowa_model, val_y, val_predictions, 3)

def use_linear_model(X_train, X_test, y_train, y_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    # pred = model.predict(eval_X)
    # print_model_performance(model, eval_y, pred, 4)
    # print ('Mead squared Error (MSE):', mean_squared_error(val_y, pred))
    return model


def use_forest_regression(X_train, val_X, y_train, val_y):
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)
    # melb_preds = forest_model.predict(val_X)
    # print_model_performance(forest_model, val_y, melb_preds, 3)
    return model


def reg_decision_tree_max_leaf(X_train, X_test, y_train, y_test):
    def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred_val)
        return(mae)


    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    mae=[]
    for i, max_leaf in enumerate(candidate_max_leaf_nodes):
        #print (i, max_leaf)
        X = get_mae(max_leaf, X_train, X_test, y_train, y_test)
        mae.append(X)
        print(f"Max leaf nodes: {max_leaf}  \t\t Mean Absolute Error:  {X}")
        
    # Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
    m = mae.index(min(mae))
    best_tree_size = candidate_max_leaf_nodes[m]

    final_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=0)
    # fit the final model 
    final_model.fit(X_train, y_train)
    # pred = final_model.predict(val_X)
    # print_model_performance(final_model, val_y, pred, 2)
    return final_model


REG_MODELS= {'Reg-Decision Tree one level': reg_decision_tree_1, 
             'Reg-Decision Tree multi level optimized': reg_decision_tree_max_leaf, 
             'Reg-Random Forest': use_forest_regression, 
             'Reg-Linear Model': use_linear_model
}


# def regression_home_data():
#     iowa_file_path = r"C:\Yahia\Python\ML\data\train.csv"
#     home_data = pd.read_csv(iowa_file_path)

#     y = home_data.SalePrice
#     # Create X
#     features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
#     X = home_data[features]

#     # Split into validation and training data
#     # X_train, val_X, y_train, val_y = train_test_split(X, y, random_state=1)
#     X_train, val_X, y_train, val_y = train_test_split(X, y, test_size =0.2)

#     print ('************ Decision Tree one level ***************')
#     reg_decision_tree_1(X_train, val_X, y_train, val_y)

#     print ('\n************ Decision Tree multi level optimized ***************')
#     # reg_decision_tree_max_leaf(X_train, val_X, y_train, val_y)

#     print ('\n************ User Forest model regression ***************')
#     use_forest_regression(X_train, val_X, y_train, val_y)

#     print ('\n************ Linear Model ***************')
#     use_linear_model(X_train, val_X, y_train, val_y)

# if __name__ == '__main__':
#     regression_home_data()
