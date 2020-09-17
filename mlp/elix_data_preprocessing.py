import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

WEATHERCOND = ['clear','cloudy','light snow/rain','heavy snow/rain']

def load_data(path):
  if(path==None):
    raise Exception("Input filepath is empty.")
  df=pd.read_csv(path, sep=',', header=0, parse_dates=['date'])
  return df

def process_data(df, method, test=False):
  df=clean_data(df,method, test)
  df=add_new_features(df, test)

  #Use of OneHotEncoder to categorize the weather data
  onehotenc = OneHotEncoder(categories=[WEATHERCOND],sparse=False)
  weather=onehotenc.fit_transform(np.reshape(df['weather'].values,(-1,1)))
  weather=pd.DataFrame(weather, index=df['weather'].index, columns=onehotenc.categories_[0])
  weather.drop(weather.columns[-1],axis=1)
  df=pd.concat([df, weather], axis=1)

  df.drop('weather', inplace=True, axis=1)
  df.drop('date', inplace=True, axis=1)
  return df

def add_new_features(df, test=False):
  """Adds new features to the dataset such as total (for total users), year, month, dayofweek."""
  if not test:
    df['total']=df['guest-users']+df['registered-users']
  df['year']=df.date.dt.year
  df['month']=df.date.dt.month
  df['dayofweek']=df.date.dt.dayofweek
  #df = pd.get_dummies(df, prefix=None, drop_first=True)
  return df

def clean_data(df, method=None, test=False):
  """
  Processes the e-scooter data for modelling purposes.
  
  - Drop duplicates
  - Check for unintended types
  - Cleans the weather data to lower case and fix 'lear' and 'loudy' spelling mistakes
  - Data rows with NaN will be dropped at the end of the method
  - Drops the date column
  
  Usage: process_data(dataframe, method='clip')
  Returns: a cleaned dataframe
  
  dataframe: the dataframe to process
  method: How to deal with negative values in numeric columns such as hour, total users, etc.
   assert - throws an assertion error
   clip - attempts to set the negative values to zero
   zero - set those values to NaN 
  """
  method = method or 'clip'
  logging.debug("clean_data()_method:{}".format(method))
  
  df.drop_duplicates(inplace=True)

  expected_dtypes = {
  'date':'datetime64[ns]',
  'hr':'int64',
  'weather':'object',
  'temperature':'float64',
  'feels-like-temperature':'float64',
  'relative-humidity':'float64',
  'windspeed':'float64',
  'psi':'int64'
  }

  df.astype(expected_dtypes, copy=False).dtypes

  NUMERIC=['temperature', 'feels-like-temperature', 'relative-humidity', 'windspeed', 'psi']

  if not test :
    expected_dtypes['guest-users'] = 'int64'
    expected_dtypes['registered-users'] = 'int64'
    NUMERIC.extend(['guest-users', 'registered-users'])

  if method == 'raise':
    if not(np.logical_and(df['hr']<24,df['hr']>=0).all()):
      raise Exception("Data column 'hr' has out of range values.")
    if not(((df[NUMERIC]<0).sum()==0).all()):
      raise Exception("Numerical columns have negative values.")
    #assert(np.logical_and(df['hr']<24,df['hr']>=0).all())
    #assert(((df[NUMERIC]<0).sum()==0).all())
  elif method == 'clip':
    df[NUMERIC] = df[NUMERIC].clip(lower=0)
  elif method == 'nan':
    df.where(df[NUMERIC]>=0, np.NaN, inplace=True)

  df.weather=df.weather.apply(clean_weather_data)
  num_na=df.shape[0]
  df.dropna(axis=0, inplace=True)
  num_na-=df.shape[0]
  if num_na>0:
    logging.debug("{} rows are dropped as they contain NaN values.".format(num_na))
  
  return df

def clean_weather_data(x):
  """
  Returns weather data cleaned
  Input: weather column as series
  Output: weather column with cleaned data
  Usage: df.weather=df.weather.apply(clean_weather_data)
  """
  condition=x.strip().lower()
  if condition in WEATHERCOND:
    return condition
  elif condition == 'loudy':
    return 'cloudy'
  elif condition == 'lear':
    return 'clear'
  else:
    return np.NaN
  return condition
  

def split_dataset(df, test_size=None, seed=None, target=None):
  """
  Selects the feature columns for training and splits dataframe into train and test sets.
  Returns: X_train, X_test, y_train, y_test
  """
  test_size=float(test_size or 0.1)
  assert(test_size<1 and test_size>0)
  seed=int(seed or 404)
  target=target or 'total'
  logging.debug("split_dataset()_test_size:{}".format(test_size))
  logging.debug("split_dataset()_seed:{}".format(seed))
  TARGET = ['guest-users','registered-users', 'total']
  FEATURES = [c for c in df.columns if c not in TARGET]
  SELECTEDT=target
  y = df[SELECTEDT]
  X = df[FEATURES]
  logging.debug("split_dataset()_target:{}".format(SELECTEDT))
  logging.debug("split_dataset()_features:{}".format(FEATURES))
  return train_test_split(X,y, test_size=test_size, random_state=seed)
  
def prepare_results(X, X_processed, y_pred):
  y_pred=pd.DataFrame(y_pred, index=X_processed.index, columns=['Predicted_Users'])
  return pd.concat([X, y_pred], axis=1)