# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:18:21 2020

@author: Tejan
"""


import tensorflow as ts
import keras as ks
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as py
from plotly.offline import plot 
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

dataset = pd.read_csv('complete.csv')

dataset.isnull().sum()
dataset.describe()

dataset['confirmed'] = dataset.iloc[:,2] + dataset.iloc[:, 3]
dataset.drop(['Total Confirmed cases ( Foreign National )'], axis = 1, inplace = True)
dataset = dataset.rename(columns = {'Name of State / UT': 'state', 'Cured/Discharged/Migrated' : 'recovered', 'Death': 'dead', 'Date' : 'date', 
                                    'Latitude' : 'lat', 'Longitude' : 'long'})


#state vs confirmed
fig = px.bar(dataset[['state', 'confirmed']].sort_values('confirmed', ascending = 'False'),
            y = 'confirmed', x = 'state', color = 'state', log_y = True,
             template = 'ggplot2', title = 'confirmed cases')
plot(fig)

#state vs recovered
fig = px.bar(dataset[['state', 'recovered']].sort_values('recovered', ascending = 'False'),
            y = 'recovered', x = 'state', color = 'state', log_y = True,
             template = 'ggplot2', title = 'recovered cases')
plot(fig)

#state vs dead
fig = px.bar(dataset[['state', 'dead']].sort_values('dead', ascending = 'False'),
            y = 'dead', x = 'state', color = 'state', log_y = True,
             template = 'ggplot2', title = 'dead cases')
plot(fig)

plt.figure(figsize = (50, 15))
plt.bar(dataset.date, dataset.confirmed, label = 'confirmed')
plt.bar(dataset.date, dataset.recovered, label = 'Recovered')
plt.bar(dataset.date, dataset.dead, label = 'Deaths')
plt.xlabel('Dates')
plt.ylabel('count')
plt.legend(frameon = True, fontsize = 12)
plt.title('Confirmed vs Recovered vs Dead')
fig = plt.gcf()
plt.savefig('India_crd.png', dpi = 100)
plt.show()

f, ax = plt.subplots(figsize = (23, 10))
ax = sns.scatterplot(x = 'date', y = 'confirmed', data = dataset,
                     color = 'black', label = 'confirmed')
ax = sns.scatterplot(x = 'date', y = 'recovered', data = dataset,
                     color = 'blue', label = 'recovered')
ax = sns.scatterplot(x = 'date', y = 'dead', data = dataset,
                     color = 'red', label = 'dead')
plt.plot(dataset.date, dataset.confirmed, zorder = 1, color = 'black')
plt.plot(dataset.date, dataset.recovered, zorder = 1, color = 'blue')
plt.plot(dataset.date, dataset.dead, zorder = 1, color = 'red')

abc = dataset.groupby('date')[['confirmed', 'recovered', 'dead']].sum().reset_index()
r_cm = (abc.recovered/abc.confirmed)
d_cm = (abc.dead/abc.confirmed)

#Prophet on confirmed
prop_con = dataset.iloc[:,[0,6]]
prop_con.columns = ['ds', 'y']
m1 = Prophet()
m1.fit(prop_con)
future = m1.make_future_dataframe(periods = 365)
forecast_con = m1.predict(future)
figure = plot_plotly(m1, forecast_con)
py.iplot(figure)
figure = m1.plot(forecast_con, xlabel = 'date', ylabel = 'confirmed')
figure = m1.plot_components(forecast_con)





