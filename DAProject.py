# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:07:57 2018

@author: nehap
"""

import pandas as pd
import os

# display and set working/data directory
os.getcwd()

# import the data; note the separator
health_data = pd.read_csv("C:/Users/nehap/Desktop/500_Cities__Local_Data_for_Better_Health__2017_release.csv", sep=",", encoding="utf-8")
health_data.head(3)
len(health_data)
#1. Most common unhealthy behaviour observed throughout United States
behavior = health_data[health_data['Category'].str.match('Unhealthy Behavior')]
count = behavior['Measure'].value_counts()

#2. which city of Virginia state is more prone to cancer
val1 = health_data[health_data['Short_Question_Text'].str.match('Cancer')]
val1.head()
val2 = health_data[health_data['StateAbbr'].str.match('VA') & health_data['Short_Question_Text'].str.match('Cancer')]

val_corr = health_data[health_data['StateAbbr'].str.match('VA')]
freq1 = val2['CityName'].value_counts()
freq1



val_corr.dtypes
#3. null hypothesis - H0 - correlation between unhealthy behaviour and health outcome is 0
# alternative hypothesis H1- there is a correlation between unhealthy behaviour and health outcome (positive or negative)
val2.dtypes
valcorr_unhealthy_behavior = val_corr['Category'] == 'Unhealthy Behaviors', 'Data_Value'
valcorr_health_outcomes = val_corr.loc[val_corr['Category'] == 'Health Outcomes', 'Data_Value']
corr_data = pd.DataFrame({'Unhealthy Behaviors':[valcorr_unhealthy_behavior], 'Health Outcomes':[valcorr_health_outcomes]})
corr_data.corr()
val_corr.plot.scatter('PopulationCount','Data_Value')


health_data['Category_factor'].corr(health_data['Data_Value'], method='pearson')
import matplotlib.pyplot as plt

plt.matshow(health_data['Category_factor'].corr(health_data['Data_Value'], method='pearson'))

health_data['PopulationCount'].corr(health_data['Data_Value'], method='spearman')


#regression analysis
import matplotlib.pyplot as plt
sample1 = Pop_temp1.sample(n=100)
sample2 = Pop_temp1.sample(n=100)
x = pd.factorize(sample1['PopulationCount'].values)[0].reshape(-1, 1)
y = pd.factorize(sample2['Data_Value'].values)[0].reshape(-1, 1)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(X_train,Y_train)
y_predict_values = simpleLinearRegression.predict(X_test)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,simpleLinearRegression.predict(X_train))
plt.title('Regression Analysis of Population Count with Data Values')
plt.xlabel('Data Value')
plt.ylabel('Population Count')
plt.show()

,x='Data Value',y='Population Count',main='Regression Analysis of Population Count with Data Values'
#average population count at geographic level city
grouped = val_corr['PopulationCount'].groupby(['CityName'],['GeographicLevel']).mean()
pop_count = grouped[grouped['GeographicLevel'].str.match('City')]

Pop_temp1 = pd.read_csv("C:/Users/nehap/Documents/AIT 580/Big data project/Pop_temp.csv", sep=",")
Pop_temp1.head(3)
Pop_temp


health_data.head(3)
category_factor = pd.factorize(health_data['Category'])
category_factor=pd.DataFrame({"Category1":category_factor})
health_data.append(category_factor)
category_factor.describe()
for i in range(0,10):
    print(category_factor[i])
health_data.head(6)
cor_test = health_data['Category_Factor'].corr(health_data['Data_Value'])
#------------------------------------------------------------------------
# H0 two samples are independent
# H1 there is a dependency between two samples
# Tests whether two categorical variables are related or independent. 
from scipy.stats import chi2_contingency
from scipy import stats
import numpy as np
obs = np.array([[health_data['Category_factor']],[health_data['Data_Value']]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print(p)
health_data.describe()
health_data.dtypes
