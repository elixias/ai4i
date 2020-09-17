import sys, getopt
import configparser
import logging
import elix_data_preprocessing as edp
import elix_modelling as emdl

def main(argv):
  """
  python mlp.py <modelclass> [model_args,...]
  """

  config = configparser.ConfigParser()
  config.read('config.ini')
  logging.basicConfig(level=(config['SETTINGS'].get('logging') or "DEBUG"))
  
  model_type=argv[0]
  kwargs = dict(i.split("=") for i in argv[1:])
  ex_model = emdl.get_exmodel(model_type) #exmodel refers to extended model
  ex_model.build(**kwargs)
  logging.debug("main()_model_params:{}".format(ex_model.model.get_params()))

  start_ml(config,ex_model)

def start_ml(config,ex_model):
  """
	input_file=iargs.get('-i'),
	output_file=iargs.get('-o'),
	model_type=iargs.get('--model_type'),
    handle_neg=iargs.get('--handle_neg'),
    test_size=iargs.get('--test_size'),
    seed=iargs.get('--seed'),
    target=iargs.get('--target'),
    imputer=iargs.get('--imputer'),
    normalize=iargs.get('--normalize'),
    poly=iargs.get('--poly')
  """
  method = 'clip'
  test_size = 0.2
  seed = 23
  target = "total"
  #change these
  df = edp.load_data(config['SETTINGS'].get('training'))
  df = edp.process_data(df, method=method)
  X_train, X_test, y_train, y_test = edp.split_dataset(df, test_size=test_size, seed=seed, target=target)
  
  pipeline = emdl.build_pipeline(ex_model.model)
  #pipeline = emdl.build_pipeline(model, imputer=imputer, normalize=normalize, poly=poly)
  
  #pipeline build and predict
  #model_args = model_settings_from_config(model_type, config)
  #model = emdl.get_model(model_type, **model_args)
  
  pipeline=pipeline.fit(X_train, y_train)
  y_pred=pipeline.predict(X_test)

  ex_model.report(X_train)
  
  print("\nRMSE score of {} in test set.".format(emdl.get_rmse(y_test, y_pred)))
  print("====================")
  
  #perform prediction on input file only if it is specified
  #temporarily disabled
  """if(input_file and input_file!='NA'):
    logging.info("start_ml():Generating predictions for input file.")
    final = perform_prediction(input_file,handle_neg,pipeline)
    if(output_file and output_file!='NA'):
      final.to_csv(output_file)
      logging.info("Results saved to {}".format(output_file))
    else:
      print(final.head(5))
  """
def perform_prediction(input_file,handle_neg,pipeline):
  X = edp.load_data(input_file)
  X_processed = edp.process_data(X, method=handle_neg, test=True)
  y_pred=pipeline.predict(X_processed)
  return edp.prepare_results(X, X_processed, y_pred)
  
def model_settings_from_config(model_type, config):
  """Create estimator based on config file"""
  
  model_type = model_type or 'linear'
	
  if model_type == 'linear' :
    c = config['LINEAR']
    fit_intercept = int(c.get('fit_intercept') or 1)
    model_args = {'fit_intercept':fit_intercept}
  elif model_type == 'lasso' :
    c = config['LASSO']
    alpha = float(c.get('alpha') or 1)
    fit_intercept = int(c.get('fit_intercept') or 1)
    positive = int(c.get('positive') or 0)
    tol = float(c.get('tol') or 0.0001)
    model_args = {'alpha':alpha,'fit_intercept':fit_intercept,'positive':positive, 'tol':tol}
  elif model_type == 'ridge' :
    c = config['RIDGE']
    alpha = float(c.get('alpha') or 1)
    fit_intercept = int(c.get('fit_intercept') or 1)
    solver = c.get('solver') or 'auto'
    tol = float(c.get('tol') or 0.0001)
    model_args = {'alpha':alpha,'fit_intercept':fit_intercept, 'solver':solver, 'tol':tol}
  else:
    raise Exception("model_settings_from_config(): The model '{}' specified at runtime is not supported.".format(model_type))

  logging.debug("create_est_from_config()_model_type:{}".format(model_type))
  logging.debug("create_est_from_config()_model_args:{}".format(model_args))
  
  return model_args

if __name__ == "__main__":
  main(sys.argv[1:])
