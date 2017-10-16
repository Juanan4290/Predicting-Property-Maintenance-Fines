
# # Assignment 2 - Pandas Introduction

# ## Part 1
# The following code loads the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on 
# [All Time Olympic Games Medals](https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table), and does some basic data cleaning. 
 
# The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals, total # number of games, 
# total # of medals. Use this dataset to answer the questions below.

import pandas as pd
import numpy as np

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()

# ###Which country has won the most gold medals in summer games?
# 
# This function returns a single string value.*

def answer_one():
    
    return df.index[(df['Gold'] == np.max(df['Gold']))][0]

answer_one()


# ### Which country had the biggest difference between their summer and winter gold medal counts?
# 
# This function returns a single string value.*

def answer_two():
    df2 = df['Gold'] - df['Gold.1']
    df2=df2.abs()
    return df2.index[df2==np.max(df2)][0]

answer_two()


# ### Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to 
# their total gold medal count? 

# $$\frac{Summer~Gold - Winter~Gold}{Total~Gold}$$
 
# Only countries that have won at least 1 gold in both summer and winter are included.
# This function returns a single string value.*

def answer_three():
    df3 = df[(df['Gold'] > 0) & (df['Gold.1'] > 0)]
    df3 = (df3['Gold'] - df3['Gold.1']) / df3['Gold.2']
    return df3.index[df3==np.max(df3)][0]

answer_three()


# ### Write a function that creates a Series called "Points" which is a weighted value where each gold medal (`Gold.2`) counts for 
# 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point. The function should return only the column (a Series object) which you created.
# 
# *This function returns a Series named `Points` of length 146*

def answer_four():
    df4 = pd.DataFrame({'Points':df['Gold.2']*3 + df['Silver.2']*2 + df['Bronze.2']})
    return df4['Points']

answer_four()


# ## Part 2
# For the next set of questions, we will be using census data from the [United States Census Bureau](http://www.census.gov/popest/data/counties/totals/2015/CO-EST2015-alldata.html). 
# Counties are political and geographic subdivisions of states in the United States. 
# This dataset contains population data for counties and states in the US from 2010 to 2015. [See this document](http://www.census.gov/popest/data/counties/totals/2015/files/CO-EST2015-alldata.pdf) for a description of the variable names.
 
# ### Which state has the most counties in it? 
# (hint: consider the sumlevel key carefully! You'll need this for future questions too...)

census_df = pd.read_csv('census.csv')

def answer_five():
    
    df5 = census_df[census_df['SUMLEV'] == 50] # when SUMLEV = 40 the row contains state level data
                                               # when SUMLEV = 50 the row contains county level data
    df5 = df5.groupby(['STNAME']).sum()['COUNTY'].reset_index()
    return df5.iloc[df5['COUNTY'].idxmax()]['STNAME']

answer_five()


# ### Only looking at the three most populous counties for each state, 
# what are the three most populous states (in order of highest population to lowest population)? Use `CENSUS2010POP`.

def answer_six():
    df6 = census_df[census_df['SUMLEV'] == 50]
    df6 = df6.groupby('STNAME').apply(lambda x: x.sort_values('CENSUS2010POP',ascending = False)).groupby('STNAME').apply(lambda x: x.ix[:3,]).groupby('STNAME').sum()['CENSUS2010POP'].reset_index()
    df6 = df6.sort('CENSUS2010POP',ascending = False).reset_index().ix[:2,'STNAME']
    return df6.tolist()

answer_six()


# ### Which county has had the largest absolute change in population within the period 2010-2015? 
# (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)
 
# e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, then its largest change in the period would be |130-80| = 50.

def answer_seven():
    census = census_df[census_df['SUMLEV'] == 50]
    diff = pd.DataFrame(data = (census.ix[:,(9,10,11,12,13,14)].apply(lambda x: np.max(x) - np.min(x),axis=1)), columns = ['Diff'])
    df7 = diff.join(census.ix[:,6]).sort('Diff', ascending = False).iloc[0]['CTYNAME']
    return df7

answer_seven()


# ### In this datafile, the United States is broken up into four regions using the "REGION" column. 
# Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.

def answer_eight():
    df8 = census_df[census_df['CTYNAME'].str.startswith('Washington')] #Counties that start with 'Washington'
    df8 = df8[df8['REGION'] == 1].append(df8[df8['REGION'] == 2]) #Region 1 or 2
    df8 = df8[df8['POPESTIMATE2015'] > df8['POPESTIMATE2014']] #Population in 2015 greater than in 2014
    return df8.ix[:,['STNAME','CTYNAME']]

answer_eight()