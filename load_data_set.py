import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 



def load_data_set_wine():
    #Loading dataset
    dataset_path = r".\data\wine-quality-white-and-red.zip"
    # dataset_path = r"C:\Yahia\Python\ML\data\wine-quality-white-and-red.csv"
    # dataset = pd.read_csv(dataset_path)
    wine  = pd.read_csv(dataset_path)
    wine =wine.query("type == 'red'").drop(columns='type', axis=1)
    

    # Pre processing
    # wine = dataset.copy()
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

    return X_train, X_test, y_train, y_test, wine

def load_ds_boston_housing():
    dataset_path = r".\data\BostonHousing.zip"
    boston_ds = pd.read_csv(dataset_path)
    y= boston_ds.medv
    X = boston_ds.drop('medv', axis='columns')

    # # Split into validation and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=1)
    return X_train, X_test, y_train, y_test, boston_ds

