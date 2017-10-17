
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 3 - More Pandas
# This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

# ### Question 1 (20%)
# Load the energy data from the file `Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']`
# 
# Convert `Energy Supply` to gigajoules (there are 1,000,000 gigajoules in a petajoule). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these, 
# 
# e.g. 
# 
# `'Bolivia (Plurinational State of)'` should be `'Bolivia'`, 
# 
# `'Switzerland17'` should be `'Switzerland'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*

# In[1]:

def answer_one():
    import os
    import pandas as pd
    import numpy as np
    from string import digits
    #################LOADING AND CLEANING ENERGY INDICATORS
    ei = (pd.read_excel('Energy Indicators.xls', skiprows = 16, skip_footer = 38, header = 0)
          .ix[1:,2:]) #Reading the excel file skiping 16 first rows and 38 last rows
    ei.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'] #rename columns

    NaNindex = ei[ei['Energy Supply per Capita'].str.startswith('...') == True].ix[:,[1,2]].index #index with '...' values
    ei.ix[NaNindex,[1,2]] = np.NaN #'...' --> NaN value

    ei['Energy Supply'] = ei['Energy Supply']*10**6 #Transforming petajoules into gigajoules

        #Remove parenthesis and numbers from country names
    names_countries = ei['Country'].str.split('\s\(') # split the index by '('
    ei['Country'] = names_countries.str[0] #Remove parenthesis
    ei['Country'] = ei['Country'].apply(lambda x: x.translate(str.maketrans('', '', digits))) #Remove numbers from names

        #Rename the following list of countries (for use in later questions):
    ei.ix[ei['Country'] == 'Republic of Korea','Country'] = 'South Korea'
    ei.ix[ei['Country'] == 'United States of America','Country'] = 'United States'
    ei.ix[ei['Country'] == 'United Kingdom of Great Britain and Northern Ireland','Country'] = 'United Kingdom'
    ei.ix[ei['Country'] == 'China, Hong Kong Special Administrative Region','Country'] = 'Hong Kong'

    energy = ei

    #################LOADING AND CLEANING GDP
    GDP = pd.read_csv('world_bank.csv', skiprows = 4, header = 0)
    GDP.rename(columns={'Country Name':'Country'}, inplace=True)

        #Rename the following list of countries: Other different way to do it
    GDP['Country'] = GDP['Country'].replace("Korea, Rep.", "South Korea")
    GDP['Country'] = GDP['Country'].replace("Iran, Islamic Rep.", "Iran")
    GDP['Country'] = GDP['Country'].replace("Hong Kong SAR, China", "Hong Kong")

        #Use only the last 10 years (2006-2015) of GDP data
    GDP = GDP.ix[:,['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]

    #################LOADING AND CLEANING ScimEn

    ScimEn = pd.read_excel('scimagojr-3.xlsx')
        #Only the top 15 countries by Scimagojr 'Rank'
    ScimEn = ScimEn.iloc[0:15]


    #################MERGIN DATA FRAMES

    df1 = (pd.merge(ScimEn, energy, how = 'inner', left_on = 'Country', right_on = 'Country')
           .merge(GDP, how = 'inner', on = 'Country')
           .set_index('Country'))
 
    return df1

answer_one()


# # Question 2 (6.6%)
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[2]:

#%%HTML
#<svg width="800" height="300">
#  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />
#  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />
#  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />
#  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>
#  <text  x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>
#</svg>


# In[108]:

def answer_two():
    import os
    import pandas as pd
    import numpy as np
    from string import digits
    #################LOADING AND CLEANING ENERGY INDICATORS
    ei = (pd.read_excel('Energy Indicators.xls', skiprows = 16, skip_footer = 38, header = 0)
          .ix[1:,2:]) #Reading the excel file skiping 16 first rows and 38 last rows
    ei.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'] #rename columns

    NaNindex = ei[ei['Energy Supply per Capita'].str.startswith('...') == True].ix[:,[1,2]].index #index with '...' values
    ei.ix[NaNindex,[1,2]] = np.NaN #'...' --> NaN value

    ei['Energy Supply'] = ei['Energy Supply']*10**6 #Transforming petajoules into gigajoules

        #Remove parenthesis and numbers from country names
    names_countries = ei['Country'].str.split('\s\(') # split the index by '('
    ei['Country'] = names_countries.str[0] #Remove parenthesis
    ei['Country'] = ei['Country'].apply(lambda x: x.translate(str.maketrans('', '', digits))) #Remove numbers from names

        #Rename the following list of countries (for use in later questions):
    ei.ix[ei['Country'] == 'Republic of Korea','Country'] = 'South Korea'
    ei.ix[ei['Country'] == 'United States of America','Country'] = 'United States'
    ei.ix[ei['Country'] == 'United Kingdom of Great Britain and Northern Ireland','Country'] = 'United Kingdom'
    ei.ix[ei['Country'] == 'China, Hong Kong Special Administrative Region','Country'] = 'Hong Kong'

    energy = ei

    #################LOADING AND CLEANING GDP
    GDP = pd.read_csv('world_bank.csv', skiprows = 4, header = 0)
    GDP.rename(columns={'Country Name':'Country'}, inplace=True)

        #Rename the following list of countries: Other different way to do it
    GDP['Country'] = GDP['Country'].replace("Korea, Rep.", "South Korea")
    GDP['Country'] = GDP['Country'].replace("Iran, Islamic Rep.", "Iran")
    GDP['Country'] = GDP['Country'].replace("Hong Kong SAR, China", "Hong Kong")

        #Use only the last 10 years (2006-2015) of GDP data
    GDP = GDP.ix[:,['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]

    #################LOADING AND CLEANING ScimEn
    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    
    df1 = pd.merge(ScimEn, energy, how = 'inner', left_on = 'Country', right_on = 'Country')
    df1 = pd.merge(df1, GDP, how = 'inner', left_on = 'Country', right_on = 'Country').set_index('Country')       
        
    df2 = pd.merge(ScimEn, energy, how = 'outer', left_on = 'Country', right_on = 'Country')
    df2 = pd.merge(df2, GDP, how = 'outer',left_on = 'Country', right_on = 'Country').set_index('Country')  
    
    return len(df2)-len(df1)

answer_two()


# <br>
# 
# Answer the following questions in the context of only the top 15 countries by Scimagojr Rank (aka the DataFrame returned by `answer_one()`)

# ### Question 3 (6.6%)
# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[7]:

def answer_three():
    import numpy as np
    Top15 = answer_one()
    df3 = (Top15.apply(lambda x: np.mean(x[10:]),axis = 1) #np.mean remove nan values
           .sort_values(ascending = False))
    return df3

answer_three()


# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[117]:

def answer_four():
    import numpy as np
    Top15 = answer_one()
    name6 = (Top15.apply(lambda x: np.mean(x[10:]),axis = 1) #np.mean remove nan values
           .sort_values(ascending = False)).index[5]
    df4 = Top15[Top15.index == name6]
    return df4.ix[0,19]-df4.ix[0,10]

answer_four()


# ### Question 5 (6.6%)
# What is the mean `Energy Supply per Capita`?
# 
# *This function should return a single number.*

# In[122]:

def answer_five():
    import numpy as np
    Top15 = answer_one()
    df5 = np.mean(Top15['Energy Supply per Capita'])
    return df5.astype(type('float', (float,), {}))

answer_five()


# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[14]:

def answer_six():
    import numpy as np
    Top15 = answer_one()
    maxValue = np.max(Top15['% Renewable'])
    Country = Top15[Top15['% Renewable'] == maxValue].index
    df6 = (Country[0],maxValue)
    return df6

answer_six()


# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[16]:

def answer_seven():
    import numpy as np
    Top15 = answer_one()
    Top15['ratioCitations'] = Top15.apply(lambda x: x['Self-citations'] / x['Citations'],axis = 1)
    maxValue = np.max(Top15['ratioCitations'])
    Country = Top15[Top15['ratioCitations'] == maxValue].index
    df7 = (Country[0],maxValue)
    return df7

answer_seven()


# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

# In[17]:

def answer_eight():
    Top15 = answer_one()
    Top15['PopEst'] = Top15.apply(lambda x: x['Energy Supply'] / x['Energy Supply per Capita'],axis = 1)
    Country = Top15.sort_values('PopEst',ascending = False).ix[2,:].name
    return Country

answer_eight()


# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[174]:

def answer_nine():
    import numpy as np
    Top15 = answer_one()
    Top15['PopEst'] = np.float64(Top15['Energy Supply'] / Top15['Energy Supply per Capita'])
    Top15['CitableDocPerCapita'] = np.float64(Top15['Citable documents'] / Top15['PopEst'])
    Top15['Energy Supply per Capita'] = np.float64(Top15['Energy Supply per Capita'])
    df9 = Top15['CitableDocPerCapita'].corr(Top15['Energy Supply per Capita'])
    return df9

answer_nine()


# In[ ]:

#def plot9():
#    import matplotlib as plt
#    %matplotlib inline
    
#    Top15 = answer_one()
#    Top15['PopEstOp'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
#    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEstOp']
#    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])



# In[ ]:

#plot9() # Be sure to comment out plot9() before submitting the assignment!


# ### Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[20]:

def answer_ten():
    import numpy as np
    Top15 = answer_one()
    median = np.median(Top15['% Renewable'])
    Top15['HighRenew'] = Top15.apply(lambda x: 1 if x['% Renewable'] >= median else 0,axis = 1)
    df10 = (Top15.groupby('HighRenew',sort = False).apply(lambda x: x.sort('Rank', ascending = True)).ix[:,'HighRenew']
    .reset_index(level=0, drop=True))
    return df10

answer_ten()


# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[141]:

def answer_eleven():
    import numpy as np
    from collections import OrderedDict
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15.apply(lambda x: x['Energy Supply'] / x['Energy Supply per Capita'],axis = 1)
    df11 = Top15.reset_index('Country')
    df11['Continent'] = df11['Country'].map(ContinentDict)
    df11 = df11.groupby('Continent')['PopEst'].agg(OrderedDict([
                                                    ('size', np.size),
                                                    ('sum', np.sum),
                                                    ('mean',np.mean),
                                                    ('std ', np.std),
                                                    ]))
                                                    
    return df11

answer_eleven()


# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[26]:

def answer_twelve():
    import pandas as pd
    import numpy as np
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    
    Top15 = answer_one()
    df12 = Top15.reset_index('Country')
    df12['Continent'] = df12['Country'].map(ContinentDict)
    df12['bins'] = pd.cut(df12['% Renewable'],5)
    df12 = df12.groupby(['Continent','bins']).agg({'Country': np.size})
    return df12['Country']

answer_twelve()


# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# 
# e.g. 317615384.61538464 -> 317,615,384.61538464
# 
# *This function should return a Series `PopEst` whose index is the country name and whose values are the population estimate string.*

# In[162]:

def answer_thirteen():
    #import locale
    #locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita'].astype(float)
    df13 = Top15['PopEst']
    df13 = df13.apply(lambda x: '{:,}'.format(x))
    return  df13

answer_thirteen()


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[ ]:

#def plot_optional():
#    import matplotlib as plt
#    %matplotlib inline
#    Top15 = answer_one()
#    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
#                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
#                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
#                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

#    for i, txt in enumerate(Top15.index):
#        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

#    print("This is an example of a visualization that can be created to help understand the data. \
#This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' \
#2014 GDP, and the color corresponds to the continent.")


# In[ ]:

#plot_optional() # Be sure to comment out plot_optional() before submitting the assignment!


# In[ ]:




# In[ ]:



