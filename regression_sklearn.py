import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# import seaborn as sns


def print_model_performance(model, y, prediction):
    # print ('coeffecient:', model.coef_)
    # print ('intercept: ', model.intercept_)
    print (f'Mead squared Error (MSE): {mean_squared_error(y, prediction):.2f}')
    print (f'Mead absolute Error (MAE): {mean_absolute_error(y, prediction):.2f}')
    print (f'Coeffecint of determination (R2): {r2_score(y, prediction):.2f}')
    print (f'Accuracy: {accuracy_score(y_true=y, y_pred=prediction)}')
    # sns.scatterplot(x=y, y=prediction)


def reg_decision_tree_1(train_X, val_X, train_y, val_y):
    # Specify Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)

    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    # print("Validation MAE: {:,.0f}".format(val_mae))
    print_model_performance(iowa_model, val_y, val_predictions)


def regression_home_data():
    iowa_file_path = r"C:\Yahia\Python\ML\data\train.csv"
    home_data = pd.read_csv(iowa_file_path)

    # Create X
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]
    y = home_data.SalePrice

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size =0.2)
    reg_decision_tree_1(train_X, val_X, train_y, val_y)


if __name__ == '__main__':
    regression_home_data()
