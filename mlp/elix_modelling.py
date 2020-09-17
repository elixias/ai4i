from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import logging
import numpy as np

def get_rmse(y_test, y_pred):
  return (mean_squared_error(y_test, y_pred))**0.5

class ModelWrapper:
  """This wrapper
	  options returns array
	  build_model(keyargs) returns model
	  report()
	  model returns the model itself"""
  model = None
  model_type = None
  opt = []
  reporter = None
  def __init__(self, model_type, reporter, opt):
    self.model_type = model_type
    self.reporter = reporter
    self.opt = opt
  def build(self, **kwargs):
    self.model = self.model_type(**kwargs)
    #review https://www.geeksforgeeks.org/extend-class-method-in-python/
  def report(self,X_train):
    return self.reporter(self,X_train)
  def model(self):
    return model

def LinearReport(self,X_train):
  model=self.model
  print('Linear Regression Report style')
  print(model.coef_[1:])
  print("========================")
  print("Coefficients:")
  print(list(zip(model.coef_[1:], X_train.columns)))
  print("\nIntercept:")
  print(model.intercept_)
  print("\n\nEND OF LINEAR REPORT")
  print("========================")

def get_exmodel(model_type):
  logging.debug("get_exmodel()_model_type:{}".format(model_type))
  if model_type=='LinearRegression':
    return ModelWrapper(LinearRegression, LinearReport, ['fit_intercept=','normalize=','copy_X=','n_jobs'])

def get_model(model, **kwargs):
  """
  Returns a model with default arguments
  get_model(model='linear')
  
  model: lasso, ridge, or linear
  """
  model = model or 'linear'

  logging.debug("get_model()_model:{}".format(model))
  estimator=None
  if model=='lasso':
    estimator=Lasso(**kwargs)
  elif model=='ridge':
    estimator=Ridge(**kwargs)
  elif model=='linear':
    estimator=LinearRegression(**kwargs)    
  logging.debug(estimator.get_params())
  return estimator

def build_pipeline(estimator, imputer=None, normalize=None, poly=None):
  """
  Returns a pipeline based on the arguments.
  imputate: True or False
  normalize: True or False
  poly: True or False
  estimator: a model with fit() method
  """
  imputer = int(imputer or 1)
  normalize = int(normalize or 1)
  poly = int(poly or 1)
  logging.debug("build_pipeline()_imputer:{}".format(imputer))
  logging.debug("build_pipeline()_normalize:{}".format(normalize))
  logging.debug("build_pipeline()_poly:{}".format(poly))
  steps=[]
  
  enc = OneHotEncoder(categories='auto')
  
  if imputer==True :
    steps.append(('imp', SimpleImputer()))
  if normalize==True :
    steps.append(('norm', Normalizer()))
  if isinstance(poly, int) and poly>0 :
    if(poly>2):
      logging.warn("Degree selection for poly features is too high.")
    steps.append(('polyfeatures', PolynomialFeatures(poly)))
  steps.append(('estimator',estimator))
  logging.debug("build_pipeline()_steps:{}".format(steps))
  return Pipeline(steps)
