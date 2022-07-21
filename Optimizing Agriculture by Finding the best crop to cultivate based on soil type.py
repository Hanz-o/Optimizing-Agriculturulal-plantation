#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the required libraries


#for manipulation
import numpy as np
import pandas as pd

#for visualization
import matplotlib.pyplot as plt
import seaborn as sns

#for interacting
import ipywidgets
from ipywidgets import interact


# In[2]:


#Reading the data set

data = pd.read_csv("data.csv")


# In[3]:


#check shape
print("Shape of the dataset :",data.shape)


# In[4]:


#Check the head of the dataset (shows first few contents of the data base)
data.head()


# In[5]:


#check for null values
data.isnull().sum()


# In[6]:


#check the crops present in the data base
data['label'].value_counts()     #shows unique values count in a column


# In[7]:


#check summary for all the crops
print("Average ratio of nitrogen in the soil : {0:.2f}".format(data['N'].mean()))
print("Average ratio of phosphorous in the soil : {0:.2f}".format(data['P'].mean()))
print("Average ratio of potassium in the soil : {0:.2f}".format(data['K'].mean()))
print("Average Temperature in celsius : {0:.2f}".format(data['temperature'].mean()))
print("Average relative  humidity in % is : {0:.2f}".format(data['humidity'].mean()))
print("Average pH value of  the soil : {0:.2f}".format(data['ph'].mean()))
print("Average rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))


# In[8]:


#Cropwise statistics


@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    print("---------------------------------------------")
    print("Statistics for Nitrogen")
    print("Minimum Nitrigen required :", x['N'].min())
    print("Average Nitrogen required :", x['N'].mean())
    print("Maximum Nitrogen required :", x['N'].max()) 
    print("---------------------------------------------")
    print("Statistics for Phosphorous")
    print("Minimum Phosphorous required :", x['P'].min())
    print("Average Phosphorous required :", x['P'].mean())
    print("Maximum Phosphorous required :", x['P'].max()) 
    print("---------------------------------------------")
    print("Statistics for Potassium")
    print("Minimum Potassium required :", x['K'].min())
    print("Average Potassium required :", x['K'].mean())
    print("Maximum Potassium required :", x['K'].max()) 
    print("---------------------------------------------")
    print("Statistics for Temperature")
    print("Minimum Temperature required : {0:.2f}".format(x['temperature'].min()))
    print("Average Temperature required : {0:.2f}".format(x['temperature'].mean()))
    print("Maximum Temperature required : {0:.2f}".format(x['temperature'].max()))
    print("---------------------------------------------")
    print("Statistics for Humidity")
    print("Minimum Humidity required : {0:.2f}".format(x['humidity'].min()))
    print(f"Minimum Humidity required : {round(x['humidity'].min(),2)}")
    print("Average Humidity required : {0:.2f}".format(x['humidity'].mean()))
    print("Maximum Humidity required : {0:.2f}".format(x['humidity'].max()))
    print("---------------------------------------------")
    print("Statistics for PH")
    print("Minimum PH required : {0:.2f}".format(x['ph'].min()))
    print("Average PH required : {0:.2f}".format(x['ph'].mean()))
    print("Maximum PH required : {0:.2f}".format(x['ph'].max()))
    print("---------------------------------------------")
    print("Statistics for Rainfall")
    print("Minimum Rainfall required : {0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall required : {0:.2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall required : {0:.2f}".format(x['rainfall'].max()))


# In[9]:


#compare the average requirement for each crop with average conditions

#Comparing Average requirement and conditions for each crop

@interact
def compare(condition = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']):
    print("Average Value for", condition, "is {0:.2f}".format(data[condition].mean()))
    print("-------------------------------------------")
    print("Rice : {0:.2f}".format(data[(data['label'] == 'rice')][condition].mean()))
    print("Black grams : {0:.2f}".format(data[(data['label'] == 'blackgram')][condition].mean()))
    print("Banana : {0:.2f}".format(data[(data['label'] == 'banana')][condition].mean()))
    print("Jute : {0:.2f}".format(data[(data['label'] == 'jute')][condition].mean()))
    print("Coconut : {0:.2f}".format(data[(data['label'] == 'coconut')][condition].mean()))
    print("Apple : {0:.2f}".format(data[(data['label'] == 'apple')][condition].mean()))
    print("Papaya : {0:.2f}".format(data[(data['label'] == 'papaya')][condition].mean()))
    print("Muskmelon : {0:.2f}".format(data[(data['label'] == 'muskmelon')][condition].mean()))
    print("Grapes : {0:.2f}".format(data[(data['label'] == 'grapes')][condition].mean()))
    print("Watermelon : {0:.2f}".format(data[(data['label'] == 'watermelon')][condition].mean()))
    print("Kidney Beans : {0:.2f}".format(data[(data['label'] == 'kidneybeans')][condition].mean()))
    print("Mung Beans : {0:.2f}".format(data[(data['label'] == 'mungbean')][condition].mean()))
    print("Oranges : {0:.2f}".format(data[(data['label'] == 'orange')][condition].mean()))
    print("Chick Peas : {0:.2f}".format(data[(data['label'] == 'chickpea')][condition].mean()))
    print("Lentils : {0:.2f}".format(data[(data['label'] == 'lentil')][condition].mean()))
    print("Cotton : {0:.2f}".format(data[(data['label'] == 'cotton')][condition].mean()))
    print("Maize : {0:.2f}".format(data[(data['label'] == 'maize')][condition].mean()))
    print("Moth Beans : {0:.2f}".format(data[(data['label'] == 'mothbeans')][condition].mean()))
    print("Pigeon Peas : {0:.2f}".format(data[(data['label'] == 'pigeonpeas')][condition].mean()))
    print("Mango : {0:.2f}".format(data[(data['label'] == 'mango')][condition].mean()))
    print("Pomegranate : {0:.2f}".format(data[(data['label'] == 'pomegranate')][condition].mean()))
    print("Coffee : {0:.2f}".format(data[(data['label'] == 'coffee')][condition].mean()))


# In[10]:


#more comparison


@interact
def compare(condition=['N','P','K','humidity','temperature','ph','rainfall']):
    print("Crops which are greater than average",condition,"\n")
    print(data[data[condition]>data[condition].mean()]['label'].unique())
    print("----------------------------------------------")
    print("Crops which are lesser than average",condition)
    print(data[data[condition]<data[condition].mean()]['label'].unique())
        
            


# In[12]:


# Distriute the crops based on season

print("Summer crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("----------------------------------------")
print("Winter crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("----------------------------------------")
print("Rainy crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())


# In[13]:


from sklearn.cluster import KMeans

#remove the labels column
x = data.drop(['label'], axis=1)

#select all he values
x = x.values

print(x.shape)


# In[14]:


# Implement the K Means algo for cluster analysis

km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

# Cluster for each crops
print("Result after applying K Means cluster analyis \n")
print("crops in first cluster:", z[z['cluster'] == 0]['label'].unique())
print("--------------------------------------------------------------")
print("crops in second cluster:", z[z['cluster'] == 1]['label'].unique())
print("--------------------------------------------------------------")
print("crops in third cluster:", z[z['cluster'] == 2]['label'].unique())
print("--------------------------------------------------------------")
print("crops in forth cluster:", z[z['cluster'] == 3]['label'].unique())


# In[15]:


# Predict the model

y = data['label']
x = data.drop(['label'], axis = 1)

print("shape of x:", x.shape)
print("shape of y:", y.shape)


# In[16]:


# train and test the result
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[17]:


# create predictive model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[18]:


from sklearn.metrics import confusion_matrix

plt.rcParams['figure.figsize'] = (10, 10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = "Wistia")
plt.title('confusion Matrix for Logistic Regression', fontsize = 15)
plt.show()


# In[19]:


#Classification Report

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


# In[21]:


data.head()


# In[22]:


prediction = model.predict((np.array([[90, 40, 40, 20, 80, 7, 100]])))
print("The suggested crop for given climatic condition is:", prediction)

