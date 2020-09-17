**README**
-
1) Model Selection
2) Command Line and Options
3) Config File
4) Reflections/Future Improvements

**Model Selection**
-
python mlp.py --model_type lasso --handle_neg zero --test_size 0.1 --seed 404 --target total --imputer 1 --normalize 1 --poly 2

The above uses Lasso as the estimator. I would avoid the use of LinearRegression as predicted users will have negative values. The current Lasso settings in config file are:

[LASSO]
alpha=0.6
fit_intercept=1
positive=0
tol=0.0001

Settings fit_intercept=0 and positive=1 are also considerations but the RMSE score is higher at 1885.

**Results:**
Coefficients:
[(133093.64686126413, 'hr'), (0.0, 'temperature'), (66214.81095658946, 'feels-like-temperature'), (-45918.21359009823, 'relative-humidity'), (0.0, 'windspeed'), (-0.0, 'psi'), (-0.0, 'year'), (0.0, 'month'), (-0.0, 'dayofweek'), (-0.0, 'clear'), (0.0, 'cloudy'), (-0.0, 'light snow/rain'), (0.0, 'heavy snow/rain')]

Intercept:
-931.8589600620317

RMSE score of 1766.6014040146968 in test set.

**Command Line and Options**
-
*python mlp.py -h*
Prints help message.

*python mlp.py -i input.csv -o output.csv --model_type lasso --handle_neg zero --test_size 0.2 --seed 123 --target guest-users --imputer 0 --normalize 0 --poly 1*

**OPTIONS:**
-i
if specified, will perform prediction and produce the output on screen or to a file if -o is given.

-o
the file to output the results to, requires -i.

--model_type
the type of model to use:
 - 'linear' (default)
 - 'lasso'
 - 'ridge'

--handle_neg
how negative values in numerical columns will be handled.
 - 'clip' (default) - sets negative values to 0
 - 'raise' - raises exception and stops
 - 'nan' - sets as NaN. Rows will be dropped before prediction.

--test_size *float*
For train test split.

--seed *int*
Seed value for train test split.

--target 
 - total (default)
 - guest-users
 - registered-users

--imputer 
0 or 1. Puts a simple imputer into the pipeline.

--normalize
0 or 1. Puts a normaliser into the pipeline.

--poly *int*
Puts a PolynomialFeatures with the specified degree into pipeline. Recommended not exceed 2.

For additional configurations, change the config.ini file

A `README.md` file that sufficiently explains the pipeline design and its usage. An explanation of your choice of model(s) and an evaluation of the model(s) developed should also be included in the README.

**Config File**
-
**Configuration file is saved as config.ini. If you've chosen 'linear' as model, the parameters used will be the ones from this file under the [LINEAR] section.**

[SETTINGS]
logging=DEBUG #Any acceptable logging level
training=scooter_rental_data.csv #training data to use

[LINEAR]
fit_intercept=1

[LASSO]
alpha=1
fit_intercept=1
positive=1
tol=0.0001

[RIDGE]
alpha=1
fit_intercept=1
solver=svd
tol=0.0001

**Reflections/Future Improvements**
-
At first I was worried that the application will not be easily transferable to future projects due to hard coded names (such as specifying column names of the features). In addition, the pipeline had been made configurable with the choice to add/include additional layers but the selections are fixed. Last, the final product had no use of a script file either since you can run the file using the command *python mlp.py* and it will still produce results for the training data.

Given the amount of time I may not be able to update to address all the above so here are my thoughts on future improvements:

 1. Application not easily transferable to future projects
Luckily the hard coded variables are contained in data_preprocessing.py, which handles data cleaning and creation of new features etc. Including the option of letting the user choose which target variable to use for ingestion, all these increased the amount of customisation tailored specifically to this project. 
Data-cleaning *also* applies to both the training and prediction dataset and the excessive cleaning renders the SimpleImputer in the pipeline redundant.
Improvements: Let user specify feature and target column names in configuration file and perform minimal preprocessing to produce an output dataframe that has last column as the target variable. Any 'hard code' specific to the project should be kept within one single method for easy customisation. The best option is not having to make changes in the code itself but everything in config file. Minimal preprocessing should be done let SimpleImputer do the rest.

 2. More customizable pipeline and model selection
Configuration file should accept a list of items to go into the pipeline instead of as arguments when running the file.
Include ElasticNet as a model option.	
Create estimator just be having any model name that is available in sklearn, an let it accept **kwargs passed into the main py file.

 1. No use of script file
User should be led through a series of steps that sets the arguments for calling the file with.