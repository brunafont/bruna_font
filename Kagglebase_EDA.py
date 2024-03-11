#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1. Import libraries


# In[1]:


get_ipython().system('pip install missingno')
     

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #https://towardsdatascience.com/a-major-seaborn-plotting-tip-i-wish-i-had-learned-earlier-d8209ad0a20e
import pandas as pd


# In[ ]:


2. Load Dataset and distinguish attributes


# In[ ]:


2.1 - Visually inspecting the dataset


# In[5]:


df = pd.read_csv('C:\creditcard.csv') 
df.head()


# In[9]:


df.tail()


# In[ ]:


2.2 - Checking columns and data types


# In[6]:


# df.columns
df.info(verbose=True, show_counts=True)


# In[ ]:


2.3 - Checking basic statistics - first insight on distributions


# In[7]:


df.describe()


# In[8]:


print(df.select_dtypes(include='number').columns)
print(df.select_dtypes(include='object').columns)
print(df.select_dtypes(include='category').columns)


# In[ ]:


3. Check for missing values


# In[10]:


df.isnull().sum()


# In[11]:


#Checking for wrong entries like symbols -,?,#,*,etc.
for col in df.columns:
    print('{} : {}'.format(col, df[col].unique()))
     


# In[12]:


for col in df.columns:
    df[col].replace({'?': np.nan},inplace=True)
    
df.info()


# In[13]:


df.isnull().sum()
     


# In[ ]:


3.1 Visualizing the missing values


# In[14]:


plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')


# In[15]:


import missingno as msno


# In[16]:


msno.bar(df)     


# In[17]:


msno.matrix(df)


# In[18]:


msno.heatmap(df)


# In[19]:


msno.dendrogram(df)


# In[ ]:


3.2. Replacing the missing values


# In[20]:


df.select_dtypes(include='number').head()


# In[21]:


df.select_dtypes(include='object').head()
     


# In[23]:


num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm','price']
for col in num_col:
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(), inplace=True)
df.head()


# In[ ]:


4. Checking Data Distributions


# In[24]:


numeric_cols = df.select_dtypes(include='number').columns
numeric_cols
     


# In[25]:


for col in numeric_cols:
    plt.figure(figsize=(18,5))
    plt.subplot(1,2,1)
    #sns.distplot(df[col])
    sns.histplot(df[col], kde=True)
    plt.subplot(1,2,2)
    sns.boxplot(x=col, data=df)
    plt.show()
     


# In[ ]:


#set the style we wish to use for our plots
sns.set_style("darkgrid")

#plot the distribution of the DataFrame "Price" column
plt.figure(figsize=(8,12))
#sns.histplot(df['price'])
sns.displot(df['peak-rpm'], kde=True, bins=50, height=8, aspect=2)


# In[ ]:


fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x="peak-rpm", data=df, ax=ax)
     


# In[ ]:


4.1.2 - Analizing distributions on categorical variable


# In[ ]:


fig, ax = plt.subplots(figsize=(8,8))
plt.pie(df["body-style"].value_counts(sort=False), labels=df["body-style"].unique())
plt.show()


# In[27]:


df["body-style"].value_counts().plot(kind="bar", figsize=(10,6))


# In[28]:


fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(df, x="body-style")


# In[ ]:


4.2 Bivariate Analysis


# In[ ]:


plt.figure(figsize=(10,10))
sns.pairplot(df.select_dtypes(include='number'))


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), cbar=True, annot=True, cmap='vlag', vmin = -1, vmax = 1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




