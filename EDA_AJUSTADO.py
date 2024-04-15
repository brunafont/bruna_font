#!/usr/bin/env python
# coding: utf-8

# # Data Science workflow  
# 
# In this sequence of notebooks we will exemplify the inner steps in the Data Science workflow.  
# We are not going to discuss the business requirements and deployment strategies, but just the phases below:
# 
# ### I - Exploratory Data Analysis (this notebook)  
# ##### II - Feature Engineering and Selection 
# ##### III - Modeling  
# ##### IV - Evaluation  
# 
# This notebook will cover the Exploratory Data Analysis (EDA)

# ## I - Exploratory Data Analysis  
# 
# Exploratory Data Analysis is a set of techniques developed by John Wilder Tukey in 1970. The philosophy behind this approach was to examine the data before building a model.  
# John Tukey encouraged statisticians to explore the data, and possibly formulate hypotheses that could lead to new data collection and experiments.  
# 
# Today data scientists and analysts spend most of their time in Data Wrangling and Exploratory Data Analysis also known as EDA. But what is this EDA and why is it so important? 
# Exploratory Data Analysis (EDA) is a step in the data science workflow, where a number of techniques are used to better understand the dataset being used.
# 
# ‘Understanding the dataset’ can refer to a number of things including but not limited to…
# 
# + Get maximum insights from a data set
# + Uncover underlying structure
# + Extracting important variables and leaving behind useless variables
# + Identifying outliers, anomalies, missing values, or human error
# + Understanding the relationship(s), or lack thereof, between variables
# + Testing underlying assumptions
# + Ultimately, maximizing your insights in a dataset and minimizing potential error that may occur later in the process

# ##### Let's see how exploratory data analysis is regarded in CRISP-DM and CRISP-ML:

# ## CRISP-DM
# 
# The CRoss Industry Standard Process for Data Mining ([CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/)) is a process model that serves as the base for a data science process.  
# It has six sequential phases:
# 
# + Business understanding – What does the business need?
# + Data understanding – What data do we have / need? Is it clean?
# + Data preparation – How do we organize the data for modeling?
# + Modeling – What modeling techniques should we apply?
# + Evaluation – Which model best meets the business objectives?
# + Deployment – How do stakeholders access the results?
# 
# 
# ![CRISP-DM Process](https://miro.medium.com/max/736/1*0-mnwXXLlMB_bEQwp-706Q.png)

# The machine learning community is still trying to establish a standard process model for machine learning development. As a result, many machine learning and data science projects are still not well organized. Results are not reproducible.  
# In general, such projects are conducted in an ad-hoc manner. To guide ML practitioners through the development life cycle, the Cross-Industry Standard Process for the development of Machine Learning applications with Quality assurance methodology ([CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml)) was recently proposed.  
# 
# There is a particular order of the individual stages. Still, machine learning workflows are fundamentally iterative and exploratory so that depending on the results from the later phases we might re-examine earlier steps.

# ## CRISP-ML
# 
# ![CRISP-ML Process](https://ml-ops.org/img/crisp-ml-process.jpg)  
# [Source](https://ml-ops.org/content/crisp-ml)

# If we explode the EDA phase in each of the previous frameworks, we would have something like this:
# 
# ![EDA](https://www.researchgate.net/publication/329930775/figure/fig3/AS:873046667710469@1585161954284/The-fundamental-steps-of-the-exploratory-data-analysis-process_W640.jpg)  
# [Source](https://www.researchgate.net/publication/329930775_A_comprehensive_review_of_tools_for_exploratory_analysis_of_tabular_industrial_datasets)

# ### Starting the EDA

# ### 1. Import libraries

# In[2]:


get_ipython().system('pip install missingno')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #https://towardsdatascience.com/a-major-seaborn-plotting-tip-i-wish-i-had-learned-earlier-d8209ad0a20e
import pandas as pd


# ### 2. Load Dataset and distinguish attributes

# ##### 2.1 - Visually inspecting the dataset

# In[4]:


df = pd.read_csv('C:\creditcard.csv')
df.head()


# ##### 2.2 - Checking columns and data types

# In[5]:


# df.columns
df.info(verbose=True, show_counts=True)


# ##### 2.3 - Checking basic statistics - first insight on distributions

# In[6]:


df.describe()


# ##### At this moment, you look for columns that shall be transformed/converted later in the workflow.

# In[10]:


print(df.select_dtypes(include='number').columns)
print(df.select_dtypes(include='object').columns)
print(df.select_dtypes(include='category').columns)


# ### 3. Check for missing values

# In[7]:


df.isnull().sum()


# It seems there are no missing values, but that may be misleading. Let's explore a bit more:

# In[8]:


#Checking for wrong entries like symbols -,?,#,*,etc.
for col in df.columns:
    print('{} : {}'.format(col, df[col].unique()))


# There are null values in our dataset in form of ‘?’. Pandas is not recognizing them so we will replace them with [`np.nan`](https://numpy.org/doc/stable/reference/constants.html#numpy.nan).

# In[9]:


for col in df.columns:
    df[col].replace({'?': np.nan},inplace=True)
    
df.info()


# In[10]:


df.isnull().sum()


# #### 3.1 Visualizing the missing values  
# Now the missing values are identified in the dataframe.
# With the help of [`heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html), we can see the amount of data that is missing from the attribute.
# With this we can make decisions whether to drop these missing values or to replace them.
# Usually dropping the missing values is not advisable but sometimes it may be helpful.

# In[15]:


plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')


# Now observe that there are many missing values in 'normalized_losses' while other columns have fewer missing values. We can’t drop the 'normalized_losses' column as it may be important for our prediction.  
# We can also use the [**missingno**](https://github.com/ResidentMario/missingno) libray for a better evaluation of the missing values. First we can check the quantity and how they distribute among the rows:

# In[11]:


import missingno as msno


# In[12]:


msno.bar(df)


# In[14]:


msno.matrix(df)


# The missingno [correlation heatmap](https://github.com/ResidentMario/missingno?tab=readme-ov-file#heatmap) measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another

# In[13]:


msno.heatmap(df)


# The [dendrogram](https://github.com/ResidentMario/missingno?tab=readme-ov-file#dendrogram) allows you to more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap.

# In[14]:


msno.dendrogram(df)


# #### 3.2. Replacing the missing values
# We will be replacing these missing values with mean because the number of missing values is not great (we could have used the median too).  
# Later, in the data preparation phase, we will learn other imputation techniques.

# In[15]:


df.select_dtypes(include='number').head()


# In[17]:


df.select_dtypes(include='int64').head()


# Now let's transform the mistaken datatypes for numeric values and fill with the mean, using the strategy we have chosen.

# In[22]:


num_col = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class']
for col in num_col:
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(), inplace=True)
df.head()


# ### 4. Checking Data Distributions
# 
# This is the most important step in EDA. 
# - This step will decide how much insight you can get.
# - Checking the distributions is fundamental for feature selection and the modeling phase
# - This step varies from person to person in terms of their questioning ability. 
# 
# Let's check the univariate and bivariate distributions and correlation between different variables, this will give us a roadmap on how to proceed further.

# #### 4.1 Univariate Analysis  
# 
# The goal here is to check the distribution of numeric and categorical variables (more about this later in the course)  
# We can quickly check the distributions of every numeric column:

# In[23]:


numeric_cols = df.select_dtypes(include='number').columns
numeric_cols


# In[21]:


for col in numeric_cols:
    plt.figure(figsize=(18,5))
    plt.subplot(1,2,1)
    #sns.distplot(df[col])
    sns.histplot(df[col], kde=True)
    plt.subplot(1,2,2)
    sns.boxplot(x=col, data=df)
    plt.show()


# ##### 4.1.1 - Analizing distributions on numerical variables - Spotting outliers
# 
# ![Outliers](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717-Module6-RandomError/Normal%20Distribution%20deviations.png)

# Assuming the data would follow a normal distribution, we can choose some of the graphs to examine the data in more detail:

# In[24]:


#set the style we wish to use for our plots
sns.set_style("darkgrid")

#plot the distribution of the DataFrame "Price" column
plt.figure(figsize=(8,12))
#sns.histplot(df['price'])
sns.displot(df['V19'], kde=True, bins=50, height=8, aspect=2)  


# In[25]:


fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x="V19", data=df, ax=ax)


# In[26]:


#set the style we wish to use for our plots
sns.set_style("darkgrid")

#plot the distribution of the DataFrame "Price" column
plt.figure(figsize=(8,12))
#sns.histplot(df['price'])
sns.displot(df['V22'], kde=True, bins=50, height=8, aspect=2)  


# In[17]:


fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x="V22", data=df, ax=ax)


# We will not treat outliers during Exploratory Data Analysis, but we will get back to them in the Data Preparation phase.

# ##### 4.1.2 - Analizing distributions on categorical variables

# Although it is not one of the recommended plots, we can always use the pie plots in special situations:

# In[30]:


fig, ax = plt.subplots(figsize=(8,8))
plt.pie(df["Amount"].value_counts(sort=False), labels=df["Amount"].unique())
plt.show()


# [Barplots](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.plot.bar.html) with frequencies can be created in Matplotlib.

# In[ ]:


df["V18"].value_counts().plot(kind="bar", figsize=(10,6))


# There is no need to separately calculate the count when using the [`sns.countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function

# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(df, x="V19")


# #### 4.2 Bivariate Analysis  
# 
# Now we want to check the relationships between pais of variables. We can start by drawing a [pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html) and a correlation plot.

# In[ ]:


plt.figure(figsize=(10,10))
sns.pairplot(df.select_dtypes(include='number'))


# The pairplot can help us gaining quick insights on the correlations of variables, but can get cluttered if we have many features.  
# We can also try the [heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) of correlations:

# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), cbar=True, annot=True, cmap='vlag', vmin = -1, vmax = 1)


# ##### Positive Correlation  
# + 'Price' – 'wheel-base', 'length', 'width', 'curb_weight', 'engine-size', 'bore', 'horsepower'  
# + 'wheel-base' – 'length', 'width', 'height', 'curb_weight', 'engine-size', 'price'  
# + 'horsepower' – 'length', 'width', 'curb-weight', 'engine-size', 'bore', 'price'  
# + 'Highway mpg' – 'city-mpg'  
# 
# ##### Negative Correlation  
# + 'Price' – 'highway-mpg', 'city-mpg'  
# + 'highway-mpg' – 'wheel base', 'length', 'width', 'curb-weight', 'engine-size', 'bore', 'horsepower', 'price'  
# + 'city' – 'wheel base', 'length', 'width', 'curb-weight', 'engine-size', 'bore', 'horsepower', 'price'  
# 
# This heatmap has given us great insights into the data.  
# Now let us apply domain knowledge and ask the questions which will affect the price of the automobile.

# ##### 4.2.1 - Checking some columns in more detail  
# We can draw a vertical boxplot grouped by a categorical variable:

# In[ ]:


fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x="fuel-type", y="horsepower", data=df, ax=ax)


# And even add a third component:  
# https://seaborn.pydata.org/tutorial/categorical.html

# In[ ]:


#sns.catplot(x="fuel-type", y="horsepower", hue="num-of-doors", kind="box", data=df, height=8, aspect=2)
sns.catplot(x="fuel-type", y="horsepower", hue="num-of-doors", kind="violin", inner="stick", split=True, palette="pastel", data=df, height=8, aspect=2)


# ### 5. Asking questions based on the analysis

# Try to ask questions related to independent variables and the target variable.  
# Example questions related to this dataset could be:  
# 
# + How does 'fuel-type' affect the price of the car?   
# + How does the 'horsepower' affect the price?  
# + What is the relationship between 'engine-size' and 'price'?  
# + How does 'highway-mpg' affects 'price'?  
# + What is the relation between no. of doors and 'price'?

# #### 5.1 How does 'fuel_type' will affect the price?  
# 
# Let's compare categorical data with numerical data. We are going to use a catplot from Seaborn, but there are other options for categorical variables:  
# https://seaborn.pydata.org/tutorial/categorical.html

# In[ ]:


plt.figure(figsize=(12,10))
#https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot
sns.catplot(x='fuel-type',y='price', data=df, height=8)
plt.xlabel('Fuel Type')
plt.ylabel('Price')


# #### 5.2 How does the horsepower affect the price?  
# 
# Matplotlib and Seaborn have very nice graphs to visualize numerical relationships:  
# https://seaborn.pydata.org/tutorial/relational.html  
# https://matplotlib.org/stable/gallery/index.html

# In[ ]:


plt.figure(figsize=(12,10))
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html  
plt.scatter(x='horsepower',y='price', data=df)
plt.xlabel('Horsepower')
plt.ylabel('Price')


# In[ ]:


#https://seaborn.pydata.org/generated/seaborn.jointplot.html

sns.jointplot(x='horsepower',y='price', data=df)
sns.jointplot(x='horsepower',y='price', data=df, kind='hex')


# We can see that most of the horsepower values lie between 50-150 with a price mostly between 5000-25000. There are outliers as well (between 200-300).  
# Let’s see a count between 50-100 i.e univariate analysis of horsepower.

# In[ ]:


plt.figure(figsize=(12,10))
#https://seaborn.pydata.org/generated/seaborn.histplot.html
sns.histplot(df.horsepower,bins=10)


# The average count between 50-100 is 50 and it is positively skewed.

# #### 5.3 What is the relation between engine-size and price?

# In[ ]:


plt.figure(figsize=(12,10))
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html  
plt.scatter(x='engine-size',y='price',data=df)
plt.xlabel('Engine size')
plt.ylabel('Price')


# In[ ]:


sns.jointplot(x='engine-size',y='price', data=df, kind='reg')
sns.jointplot(x='engine-size',y='price', data=df, kind='kde')


# We can observe that the pattern is similar to horsepower vs price.

# #### 5.4 How does highway_mpg affects price?

# In[ ]:


plt.figure(figsize=(12,10))
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html  
plt.scatter(x='highway-mpg',y='price',data=df)
plt.xlabel('Higway mpg')
plt.ylabel('Price')


# We can see price decreases with an increase in 'higway-mpg'.

# #### 5.5 What is the relation between no. of doors and price?  
# 
# Let us first check the number of doors.

# In[ ]:


# Unique values in num_of_doors
df["num-of-doors"].value_counts().plot(kind="bar", figsize=(10,6))


# In[ ]:


plt.figure(figsize=(12,8))
#https://seaborn.pydata.org/generated/seaborn.boxplot.html
sns.boxplot(x='price', y='num-of-doors',data=df)


# With this boxplot we can conclude that the average price of a vehicle with two doors is 10000,  and the average price of a vehicle with four doors is close to 12000.  
# With this plot, we have gained enough insights from the  data and our data is ready to build a model.

# ##### There are ways to explore relationships between more than two variables; although it can get a bit more complicated to interpret.

# In[ ]:


# Create a pivot table for car manufactures and fuel with horsepower rate as values
grouped = pd.pivot_table(data=df,index='make',columns='fuel-type',values='horsepower',aggfunc='mean')

# Create a heatmap to visualize manufactures, fuel type and horse power
plt.figure(figsize=[12,10])
sns.heatmap(grouped, annot=True, cmap='coolwarm', center=0.117)

plt.title("Horse Power per Manufacturer")
plt.show()


# ## II - Feature Engineering and Selection

# This will be developed in the next modules

# ## III - Modeling

# This will be developed in the next modules

# ## IV - Evaluation

# This will be developed in the next modules

# # Your Turn!

# #### Open the datasets available for the use cases and start the EDA.  
# #### You will be able to make a better decision on which one to use and how to exploit them.

# In[ ]:


df = pd.read_csv("../../../3_artificial_use_case/1_Classification_RECOMMENDED/Bank_Dataset/bank-additional-full.csv", sep=";")
df.head()


# In[ ]:


df = pd.read_csv("../../../3_artificial_use_case/2_Regression_RECOMMENDED/Datasets/2015.csv")
df.head()

