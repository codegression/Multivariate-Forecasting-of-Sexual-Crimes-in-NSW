#!/usr/bin/env python
# coding: utf-8

# # Multivariate forecasting of sexual crimes in NSW
# 
# Python code to perform multivariate forecasting of sexual crimes in NSW based on data from the NSW Bureau of Crime Statistics and Research.

# # Loading libraries

# Let's load relevant Python libraries.

# In[16]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import keras
import itertools
import datetime
from dateutil.relativedelta import *
from IPython.display import display, Markdown
import pmdarima as pm
import statsmodels.api as sm
from contextlib import contextmanager
import sys, os
import warnings
warnings.filterwarnings("ignore")


# # Loading data

# The original data acquired from https://www.bocsar.nsw.gov.au/Pages/bocsar_datasets/Datasets-.aspxNSW. It contains monthly data on all criminal incidents recorded by police from 1995 to Mar 2020. It was processed and cleaned in another notebook. Now we are going to use the processed file.

# In[2]:


data = pd.read_csv('crimes_nsw.csv', index_col=0, parse_dates=True)


# In[3]:


data.head()


# # Correlation analysis

# In[4]:


corr = data.corr()


# In[392]:


corr.head()


# Lets find other crimes that are correlated with sexual crimes. For more information on correlation analysis of crimes, refer to my other notebook.

# In[137]:


threshold = 0.8
column='Sexual assault'
targetcolumn = corr[column]
filteredcolumn = targetcolumn[((targetcolumn>threshold) & (targetcolumn<1))|
                                  ((targetcolumn<-threshold) & (targetcolumn>-1))]
    

if len(filteredcolumn)==0:
    display(Markdown('None'))
else:
    print(filteredcolumn)
    print('')
    print('')
    np.random.seed(53)
    colors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(filteredcolumn)+1, replace=False)
    data[column].plot(kind = 'line', color = colors[0],label = column,linewidth=2,alpha = 1,grid = True,linestyle = '-')
    i=1
    for index, value in list(filteredcolumn.items()):            
        data[index].plot(kind = 'line', color = colors[i],label = index,linewidth=2,alpha = 1,grid = True,linestyle = '-')
        i=i+1
    plt.legend(loc='upper right')     
    plt.legend(bbox_to_anchor=(1.05, 1))   
    plt.xlabel('')              
    plt.ylabel('Number of cases')
    plt.title('Crimes correlated with '+ column.lower())            
    plt.show()
                
    for index, value in list(filteredcolumn.items()): 
        print('')    
        display(Markdown("### Relationship between '"+ column + "' and '" + index + "'" ))
        if value > 0:
            if value > threshold * 1.125:
                print('There is a strong positive correlation with a coefficient of ' + str(value) + ".")
            else:
                print('There is a somewhat a weak positive correlation with a coefficient of ' + str(value) + ".")
        else:
            if value > threshold * -1.125:                    
                print('There is a strong negative correlation with a coefficient of ' + str(-value) + '.')
            else:
                print('There is a somewhat a weak negative correlation with a coefficient of ' + str(-value) + ".")
                
        sns.jointplot(x=column, y=index, data=data, kind="kde")  
        plt.show()             
print('')   
print('')
print('')
            
   
   


# # Forecasting using machine learning

# Let's use a recurrent neural network with gated recurrent units in the first layer.

# ### Data Preprocessing and Preparation

# Since the time series are not stationary, let's take the first order difference of all the time series.

# In[6]:


data_diff = data.diff().iloc[1:] #take the first difference and then ignore the first row (NA's)


# In[7]:


data_diff.head()


# In[15]:


a = data.loc[:, columns]


# In[16]:


a.head()


# In[136]:



data['Sexual assault'].plot(kind = 'line', label='Original', color ='red',linewidth=2,alpha = 1,grid = True,linestyle = '-')
data_diff['Sexual assault'].plot(kind = 'line', label='First difference', color ='blue',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.legend(bbox_to_anchor=(1.05, 1))   
plt.show()


# Time series data need to be converted into 'supervised' data. The following function accepts a dataset in a time series form and spits out the dataset in supervised form.

# In[12]:


def BuildSupervisedData(dataset, columns, dim, targetcolumn):
    """
    The following function accepts a dataset in a time series form and spits out the dataset in supervised form.
    
    Parameters
    ----------
    dataset : Pandas DataFrame 
        Multivariate time series where rows are time stamps and columns are variables
    
    columns : List of str
        List of columns that we are interested input
    
    dim : List of int
        Corresponding dimensions for the columns. The size of the list must match columns.
        The dimension of a variable means how many past values are to be used to predict the next time step.abs
        
    targetcolumn : str
        The target column for Y.
        
    returns
    -------
        X and Y:  numpy arrays    
    
    """
    if (isinstance(dim, int)):
        dim = [dim] * len(columns)    
    width = sum(dim)    
    
    X = np.zeros((len(dataset)-max(dim), width, 1))  #Keras requires 3D array
    Y = np.zeros(len(dataset)-max(dim))

    row_index = 0
    for i in range(max(dim), len(dataset)):
        col_index = 0
        for j in range(len(columns)):
            for k in range(dim[j]):
                X[row_index, col_index, 0] = dataset[columns[j]][i-dim[j]+k]
                col_index=col_index + 1
        Y[row_index] = dataset[targetcolumn][i]        
        row_index = row_index + 1
    return X, Y 


# Let's divide the dataset into 3 sets:
# 1. Training set (all the samples except the last 24 + 12 samples)
# 2. Validation set (24 samples)
# 3. Test set (the last 12 samples)

# In[37]:


def BuildTrainingValiTestData(X, Y):
    trainX = X[:-36, :, :]
    trainY = Y[:-36]
    valiX = X[-36:-12, :, :]
    valiY = Y[-36:-12]
    testX = X[-12:, :, :]
    testY = Y[-12:]
    return trainX, trainY, valiX, valiY, testX, testY


# In[3]:


def RMSE(predictions, targets=0):
    if (isinstance(targets, int)):
        targets = np.zeros(predictions.shape[0])
    return np.sqrt(np.mean((predictions-targets)**2))


# In[68]:


def GetValidationError(dataset, columns, dim, targetcolumn):
    x, y = BuildSupervisedData(dataset, columns, dim, targetcolumn)
    trainX, trainY, valiX, valiY, testX, testY = BuildTrainingValiTestData(x, y)
    
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(10, input_shape=(trainX.shape[1], trainX.shape[2])))#, return_sequences=True
    model.add(keras.layers.Dense(5))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=250, batch_size=1, verbose=0)
    
    predictions = model.predict(valiX)    
    return RMSE(predictions, valiY)


# ### Univariate forecasting

# RMSE can be a misleading metric on its own. We need to gauge it with a benchmark. We need to measure the RMSE of a prediction system that simply predicts the next value to be the current value.

# In[58]:


baseline_RMSE = RMSE(valiY)
print(baseline_RMSE)


# Any prediction system needs to achieve lower than baseline RMSE of <font color="red">77.311</font>

# Let's perform univariate forecasting of 'Sexual assault' using a range of past values (from 1 to 15)

# In[48]:


results = []
for i in range(1, 16):
    error = GetValidationError(data_diff, ['Sexual assault'], i, 'Sexual assault')
    results.append(error)
    print("Using", i, "past values produces RMSE", error)


# The results are not good. The RMSE is directly proportional to the number of past values. This is probably due to overfitting since there are less than 300 training samples. The RMSE values are all higher than the baseline RMSE.

# In[52]:


plt.plot(range(1, 16), results)
plt.xlabel('Number of time steps from history used to predict future')              
plt.ylabel('RMSE')
plt.title('Performance of univariate forecasting')            
plt.show()


# ### Multivariate forecasting

# Let's use all the exogenous variables to perform forecasting.

# In[69]:


for i in range(1,4):
    error = GetValidationError(data_diff, ['Sexual assault', 
                                           'Intimidation, stalking and harassment',
                                        'Fraud',
                                        'Possession and/or use of other drugs',
                                        'Other drug offences',
                                        'Pornography offences',
                                        'Breach Apprehended Violence Order',
                                        'Breach bail conditions',
                                        'Transport regulatory offences'],
                               i, 'Sexual assault')
    print("Using", i, "past time step(s):", error)


# Still above the baseline error rate.
# 
# Let's use pairs of exogenous variables.

# In[71]:


columns = ['Sexual assault', 
           'Intimidation, stalking and harassment',
           'Fraud',
           'Possession and/or use of other drugs',
           'Other drug offences',
           'Pornography offences',
           'Breach Apprehended Violence Order',
           'Breach bail conditions',
           'Transport regulatory offences']

for column in columns:
    if column!='Sexual assault':
        exogenous = ['Sexual assault', column]
        error = GetValidationError(data_diff,exogenous, 3, 'Sexual assault')
        print("Exogenous=" + column + ":", error)


# Since the results are not good. Let's just try statistical forecasting techniques.

# # Forecasting using statistical techniques

# ### Univariate forecasting

# Let's divide the dataset into 3 sets:
# 1. Training set (all the samples except the last 24 + 12 samples)
# 2. Validation set (24 samples)
# 3. Test set (the last 12 samples)

# In[5]:


def Evaluate(exog_list):
    
    endog = data['Sexual assault'].iloc[1:-36]
    exog =  data[exog_list].iloc[:-37].to_numpy()
    
    validation_endog = data['Sexual assault'].iloc[-35:-12]
    validation_exog = data[exog_list].iloc[-36:-13].to_numpy()
    
    optimization = pm.auto_arima(y=endog, exogenous=exog, 
                             start_p=1, max_p=24,
                             start_q=1, max_q=24,   
                             start_d=1, max_d=24, 
                             start_P=0, max_P=24,
                             start_Q=0, max_Q=24,
                             start_D=1, max_D=24, 
                             m=12,                                                      
                             seasonal=True,
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)
    
    model = sm.tsa.statespace.SARIMAX(endog=endog, exog = exog,
                                  order=optimization.order,
                                  seasonal_order=optimization.seasonal_order,
                                  trend='c')

    residue = model.fit(disp=False)
    predictions = residue.forecast(23, exog=validation_exog)
    error = RMSE(predictions, validation_endog)
    return error, optimization.order, optimization.seasonal_order


# In[8]:


error, o, s = Evaluate(['Sexual assault'])
print('error:', error)
print('order:', o)
print('seasonal_order:', s)


# The RMSE is only 37.32, much less than the baseline error rate. Let's plot the predicted values.

# In[12]:


endog = data['Sexual assault'].iloc[1:-36]
exog =  data['Sexual assault'].iloc[:-37].to_numpy()
model = sm.tsa.statespace.SARIMAX(endog=endog, exog = exog,
                                  order=(0, 1, 1),
                                  seasonal_order=(1, 0, 2, 12),
                                  trend='c')

residue = model.fit(disp=False)
validation_exog = data['Sexual assault'].iloc[-36:-13]
predictions = residue.forecast(23, exog=validation_exog)

validation_exog.plot(kind = 'line', color = 'red',label = 'Actual',linewidth=2,alpha = 1,grid = True,linestyle = '-')
pd.Series(predictions).plot(kind = 'line', color = 'blue',label = 'Prediction',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('')            
plt.show()


# Although the model can correctly predict troughs and peaks, it is still not good enough. Let's go for the multivariate approach.

# ### Multivariate forecasting

# We can frame the problem of multivariate forecasting as predicting the next value of endogenous varialbe ('Sexual assult') given previous (not current) values of endogenous variables. We have to shift each endogenous variable to the left by one time step to achive this. There is also a problem of selecting endogenous variables. In fact, there are many combinations of them.

# In[186]:


variables = ['Sexual assault',
            'Intimidation, stalking and harassment',
           'Fraud',
           'Possession and/or use of other drugs',
           'Other drug offences',
           'Pornography offences',
           'Breach Apprehended Violence Order',
           'Breach bail conditions',
           'Transport regulatory offences']

for i in range(1, len(variables)+1):
    for subset in itertools.combinations(variables, i):
        print(subset)


# In[149]:


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


# In[ ]:


min_error = 5000
min_subset = []
order= []
seasonal_order = []

for i in range(1, len(variables)+1):
    for subset in itertools.combinations(variables, i):       
        with suppress_stdout():           
            error, o, s = Evaluate(list(subset))    
            
        print("Exog =", subset, ", error=", error)
        if error < min_error:
            min_error = error
            min_subset = subset
            order = o
            seasonal_order = s
print('Minimum error:', min)
print('Variables:', min_subset)
print('Order:', order)
print('Seasonal_order:', order)


# The lowest error rate (18.6103) is obtained by a combination of three variables ['Sexual assault', 'Pornography offences', 'Breach bail conditions'].
# 
# The parameters of the model are:
# order: (0, 0, 1)
# seasonal_order: (2, 0, 0, 12)
# 
# Let's verify it on the test set and see whether the model can achieve similar results.

# In[7]:


exog_list = ['Sexual assault', 'Pornography offences', 'Breach bail conditions']

endog = data['Sexual assault'].iloc[1:-12]
exog =  data[exog_list].iloc[:-13].to_numpy()
test_endog = data['Sexual assault'].iloc[-11:]
test_exog = data[exog_list].iloc[-12:-1].to_numpy()

model = sm.tsa.statespace.SARIMAX(endog=endog, exog = exog,
                                  order=(0, 0, 1),
                                  seasonal_order=(2, 0, 0, 12),
                                  trend='c')

residue = model.fit(disp=False)
predictions = residue.forecast(11, exog=test_exog)
error = RMSE(predictions, test_endog)
print(error)


# It's higher than the one obtained using the validation set but still slightly lower than that of the univariate approach using the validation set.

# # Out-of-dataset prediction (Foreseeable future)

# Let's retrain with all the data (303 samples) and let the model perform forecasting for the next 30 months. Multi-step forecasting using exogenous variables is challenging because the model requires unavailable future data from exogeneous variables. We can solve this by performing univariate forecasting of these exogeneous variables along the way.

# In[9]:


# Sub-model to predict pornograpy offences
optimization = pm.auto_arima(data['Pornography offences'], 
                             start_p=1, max_p=24,
                             start_q=1, max_q=24,   
                             start_d=1, max_d=24, 
                             start_P=0, max_P=24,
                             start_Q=0, max_Q=24,
                             start_D=1, max_D=24, 
                             m=12,                                                      
                             seasonal=True,
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)
    
model = sm.tsa.statespace.SARIMAX(endog=data['Pornography offences'],
                                  order=optimization.order,
                                  seasonal_order=optimization.seasonal_order,
                                  trend='c')
submodel1 = model.fit(disp=False)


# In[10]:


# Sub-model to predict 'Breach bail conditions'
optimization = pm.auto_arima(data['Breach bail conditions'], 
                             start_p=1, max_p=24,
                             start_q=1, max_q=24,   
                             start_d=1, max_d=24, 
                             start_P=0, max_P=24,
                             start_Q=0, max_Q=24,
                             start_D=1, max_D=24, 
                             m=12,                                                      
                             seasonal=True,
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)
    
model = sm.tsa.statespace.SARIMAX(endog=data['Breach bail conditions'],
                                  order=optimization.order,
                                  seasonal_order=optimization.seasonal_order,
                                  trend='c')
submodel2 = model.fit(disp=False)


# In[14]:


endog = data['Sexual assault'].iloc[1:]
exog =  data[exog_list].iloc[:-1].to_numpy()


model = sm.tsa.statespace.SARIMAX(endog=endog, exog = exog,
                                  order=(0, 0, 1),
                                  seasonal_order=(2, 0, 0, 12),
                                  trend='c')

predictions = []
dates = []
residue = model.fit(disp=False)
ex=[[data['Sexual assault'][-1], data['Pornography offences'][-1], data['Breach bail conditions'][-1]]]
for i in range(30):    #30 months
    prediction = residue.forecast(i+1, exog=np.array(ex, dtype=float))[i]
    predictions.append(prediction)
    dates.append(data.index[-1] + relativedelta(months=+(i+1))) 
    ex.append([prediction, submodel1.forecast(i+1)[i], submodel2.forecast(i+1)[i]])
       

a = pd.Series(predictions)
a.index = pd.Series(dates)
print(a)


# In[15]:


data['Sexual assault'].plot(kind = 'line', color = 'red',label = 'Current data',linewidth=2,alpha = 1,grid = True,linestyle = '-')
a.plot(kind = 'line', color = 'blue',label = 'Forecast',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.legend(bbox_to_anchor=(1.05, 1))  
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Out-of-dataset prediction')            
plt.show()


# # Conclusion

# Sexual crimes in NSW are expected to continue to rise in the foreseeable future. Out of many crimes that are correlated with sexual crimes, pornography offences and breaching bail conditions seem to play a significant role in predicting sexual crimes. It is unclear how the rate of breaching bail can predict sexual crimes. But it is most likely that sexual crimes and breaching bail are caused by common external factors. 
# 
