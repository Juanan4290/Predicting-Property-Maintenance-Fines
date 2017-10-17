
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[111]:

import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[3]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[81]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    #Leemos el archivo
    uniTown = pd.read_csv('university_towns.txt', sep="\n", header = None, names=["list"])
    
    #Separamos las regiones a partir del '(' y nos quedamos con el primer elemento
    uniTown['RegionName'] = uniTown['list'].apply(lambda x: x.split(' (')[0])
    #Nos quedamos con las regiones que tengan '[' (estados), hacemos un drop del resto y rellenamos automaticamente 
    uniTown['State'] = (uniTown['RegionName'].apply(lambda x: x.split('[')[0].strip() if x.count('[') > 0 else np.NaN)
                        .fillna(method="ffill"))
    #Eliminamos los '['
    uniTown['RegionName'] = uniTown['RegionName'].apply(lambda x: x.split('[')[0])
    
    #Convertimos en NaN los estados de la columna 'RegionName'
    uniTown.ix[(uniTown['State'] == uniTown['RegionName']),'RegionName'] = np.NaN
    
    #Quitamos las filas con NaN, la columna 'list' inicial y reseteamos los índices
    uniTown = uniTown.dropna().drop('list', axis = 1).reset_index(drop = True)
    #Intercambiamos las columnas
    uniTown = uniTown[['State','RegionName']]
    
    return uniTown

get_list_of_university_towns()


# In[6]:

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    
    gdp = (pd.read_excel('gdplev.xls', skiprows = 5, header = 0).ix[214:,(4,6)]
           .rename(columns = {'Unnamed: 4': 'time',
                              'GDP in billions of chained 2009 dollars.1': 'GDP'})
           .reset_index(drop = True))
    
    start = []
    
    for i in range(0,(len(gdp)-2)):
        if (gdp['GDP'][i] > gdp['GDP'][i+1] and gdp['GDP'][i+1] > gdp['GDP'][i+2]):
            start.append(gdp['time'][i])
            
    return start[1]

get_recession_start()


# In[7]:

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    
    gdp = (pd.read_excel('gdplev.xls', skiprows = 5, header = 0).ix[214:,(4,6)]
           .rename(columns = {'Unnamed: 4': 'time',
                              'GDP in billions of chained 2009 dollars.1': 'GDP'})
           .reset_index(drop = True))
    
    start = get_recession_start()
    finish = []
    
    for i in range(0,(len(gdp)-2)):
        if (gdp['GDP'][i] < gdp['GDP'][i+1] and gdp['GDP'][i+1] < gdp['GDP'][i+2]):
            if (gdp['time'][i] >= start):
                finish.append(gdp['time'][i]) 
        
    return finish[2]

get_recession_end()


# In[8]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    
    gdp = (pd.read_excel('gdplev.xls', skiprows = 5, header = 0).ix[214:,(4,6)]
           .rename(columns = {'Unnamed: 4': 'time',
                              'GDP in billions of chained 2009 dollars.1': 'GDP'})
           .reset_index(drop = True))
    
    start = get_recession_start()
    finish = get_recession_end()
    
    start_index = gdp.ix[gdp['time'] == start,:].index[0]
    finish_index = gdp.ix[gdp['time'] == finish,:].index[0]
    
    gdp_crisis = gdp.iloc[start_index:finish_index]
    gdp_min = np.min(gdp_crisis['GDP'])
    
    time = str((gdp_crisis[gdp_crisis['GDP'] == gdp_min])['time'])
    
    return time[6:12]

get_recession_bottom()


# In[9]:

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    
    housingData = pd.read_csv('City_Zhvi_AllHomes.csv')
    housingData = housingData.drop(housingData.columns[3:51],axis = 1).drop('RegionID', axis = 1)
    
    for year in range(2000,2016):
        housingData[str(year)+'q1'] = housingData.ix[:,[str(year)+'-01',str(year)+'-02',str(year)+'-03']].mean(axis = 1)
        housingData[str(year)+'q2'] = housingData.ix[:,[str(year)+'-04',str(year)+'-05',str(year)+'-06']].mean(axis = 1)
        housingData[str(year)+'q3'] = housingData.ix[:,[str(year)+'-07',str(year)+'-08',str(year)+'-09']].mean(axis = 1)
        housingData[str(year)+'q4'] = housingData.ix[:,[str(year)+'-10',str(year)+'-11',str(year)+'-12']].mean(axis = 1)
    year = 2016
    housingData[str(year)+'q1'] = housingData.ix[:,[str(year)+'-01',str(year)+'-02',str(year)+'-03']].mean(axis = 1)
    housingData[str(year)+'q2'] = housingData.ix[:,[str(year)+'-04',str(year)+'-05',str(year)+'-06']].mean(axis = 1)
    housingData[str(year)+'q3'] = housingData.ix[:,[str(year)+'-07',str(year)+'-08']].mean(axis = 1)

    housingData = housingData.drop(housingData.columns[2:202],axis = 1)
    housingData['State'] = housingData['State'].map(states)
    
    return housingData.set_index(['State', 'RegionName'])

convert_housing_data_to_quarters()


# In[128]:

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    #Leyendo las funciones de preguntas anteriores para el test
    start = get_recession_start()
    bottom = get_recession_bottom()
    housingData = convert_housing_data_to_quarters()
    
    #Índice de la columna inicio de la recesión (start) y punto del precio más bajo (bottom)
    startIndex = housingData.columns.get_loc(start)
    bottomIndex = housingData.columns.get_loc(bottom)
    
    #Cargo los precios de las casas sólo desde el inicio hasta el punto más bajo
    housingDataRecession = housingData.ix[:,startIndex-1:bottomIndex+1]
    data = housingDataRecession.reset_index()
    
    #Precios de las cosas en University towns
    uniTowns = get_list_of_university_towns()
    uniHouse = pd.merge(uniTowns, data, how = 'inner', on = ['State','RegionName'])
    uniHouse['ratio'] = uniHouse.ix[:,2] / uniHouse.ix[:,6]
    uniHouse = uniHouse.set_index(['State','RegionName'])
    
    #Precios de las casa en el resto de ciudades
    nonUniHouse = pd.merge(uniTowns, data, how = 'outer', on = ['State','RegionName'], indicator = True)
    nonUniHouse = nonUniHouse[nonUniHouse['_merge'] == 'right_only'].drop('_merge',axis = 1)
    nonUniHouse['ratio'] = nonUniHouse.ix[:,2] / nonUniHouse.ix[:,6]
    nonUniHouse = nonUniHouse.set_index(['State','RegionName'])
    
    #Lanzamos el test
    ttest = ttest_ind(uniHouse['ratio'].dropna(),nonUniHouse['ratio'].dropna())
    different = True if ttest[1] <= 0.01 else False
    better = 'university town' if uniHouse['ratio'].mean() < nonUniHouse['ratio'].mean() else 'non-university town'
    
    return (different,ttest[1],better)

run_ttest()


# In[ ]:




# In[ ]:



