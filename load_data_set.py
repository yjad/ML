import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
import utils


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



def load_ds_germany_used_cars():
    #Loading dataset
    dataset_path = r".\data\GermanyUsedCars.zip"
    dfo  = pd.read_csv(dataset_path)
    
    # drop non-numeric year & price_in_euro values
    df = dfo.copy()
    df = df[~((utils.check_non_numeric(df.year)) | utils.check_non_numeric(df.price_in_euro))].\
        assign(year = lambda x: x.year.astype(int)).\
        assign (price_in_euro = lambda x: x.price_in_euro.astype(float))
    
    # drop columns of year < 1900 or > 2023
    df = df.loc[df.year.between(1900, 2023)]

    # drop desc columns
    df = df.drop(columns='offer_description')

    # set non values for columns color & fuel_consumption_l_100km
    df.color = df.color.fillna('notDefined')
    df.fuel_consumption_l_100km = df.fuel_consumption_l_100km.fillna('notDefined')
    df.power_kw = df.power_kw.fillna('999') # not defined
    df.power_ps = df.power_ps.fillna('999') # not defined
    df.mileage_in_km = df.mileage_in_km.fillna(999) # not defined 
        
    print ("number_of_dropped_rows:", dfo.shape[0]- df.shape[0])

    # convert non numeric to nueric for calculation
    # dfr = df.copy()
    for c in ['brand', 'model', 'color', 'registration_date', 'fuel_consumption_l_100km', 'transmission_type', 'fuel_type', 'fuel_consumption_g_km']:
        print (c)
        enc = LabelEncoder().fit(df[c])
        df[c] = enc.transform(df[c])

    print (df.isna().sum())
    # prepare X & Y
    y= df.price_in_euro
    X = df.drop('price_in_euro', axis='columns')

    # # Split original sd into validation and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=1)
    
    return X_train, X_test, y_train, y_test, dfo

