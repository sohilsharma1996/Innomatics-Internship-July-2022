#!/usr/bin/env python
# coding: utf-8

# # DATA DESCRIPTION :

# An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.
# 
# The dataset contains 15 columns, in which the Target Field: Income (The Income is divided into two classes: <=50K and >50K).
# 
# Other Number of Attributes: 14. These are the Demographics and other features needed to describe a person.

# # PROBLEM STATEMENT :

# The Prediction Task is to determine whether a Person makes over 50K Income in a year.
# 
# Note : We can explore the possibility in predicting income level based on the Individual’s Personal Information.
# 
# Also, Objective is to perform EDA, find the missing values if any, find the outliers, and lastly build various Machine Learning models considering ‘income’ as target variable and compare the performance of each of the ML Model.

# # Steps to be Followed :

# Step - 1 : Introduction -> Give a detailed data description and objective
# 
# Step - 2 : Import the data and perform basic pandas operations
# 
# Step - 3 : Univariate Analysis -> PDF, Histograms, Boxplots, Countplots, etc.. Understand the probability and frequency distribution of each numerical column Understand the frequency distribution of each categorical Variable/Column Mention observations after each plot
# 
# Step - 4 : Bivariate Analysis Discover the relationships between numerical columns using Scatter plots, hexbin plots, pair plots, etc.. Identify the patterns between categorical and numerical columns using swarmplot, boxplot, barplot, etc.. Mention observations after each plot.
# 
# Step - 5 : In the above steps you might have encountered many missing values and outliers Find and treat the outliers and missing values in each column
# 
# Step - 6 : Apply appropriate hypothesis tests to verify the below mentioned questions
# 
# Is there a relationship between occupation and gender? (i.e. does the preference of occupation depend on the gender)
# 
# Is there a relationship between gender and income?
# 
# Step - 7 : Split the data into train and test. After which you need to perform feature transformation: For Numerical Features -> Do Column Standardization For Categorical -> if more than 2 categories, use dummy variables. Otherwise convert the feature to Binary. You are free to explore other feature transformations.
# 
# Step - 8 : Build various Machine Learning models considering ‘income’ as target variable. Also make sure to perform Hyperparameter tuning to avoid Overfitting of models.
# 
# Step - 9 : Create a table to compare the performance of each of the ML Model

# # About Dataset : Adult Census Income Level Prediction (adult.csv)

# The dataset contains a total of 15 columns, in which : 
#     
# 1. Target Field: Income (The Income is divided into two classes: <=50K and >50K).
# 
# 2. Other Number of Attributes: 14. These are the Demographics and other features needed to describe a person, such as age , workclass , fnlwgt , educational-num , etc. ,out of which some are Numerical Data Columns and some are Categorical Data Columns.
# 
# 3. Adult dataset has 48842 instances and 15 attributes. The last one is the qualitative attribute which is called ‘income’ in the code.
# 
# 4. There are 48842 instances with 14 quantitative attributes and 1 qualitative attribute which all clearly describing its meaning. 14 quantitative attributes: 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',' occupation', 'relationship', 'race', 'sex', 'capitalgain', 'capital-loss', 'hours-per-week', 'native-country'.
# 
# 5. Qualitative attribute: 'income'. ‘income’ has two values which is ‘<=50k’ (less or equal to 50k/yr) and ‘>50k’ (more than 50k/yr). The quantitative attributes are the features and the qualitative attribute is the target. There are 9 attributes are ‘string’ within raw data: ‘workclass’, ‘education’, ‘marital-status’, ‘occupation’, ‘relationship’, ‘race’, ‘sex’,’ native-country’ and ’income’.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')


# # Representing the DataType and Description for all the Variables :

# In[2]:


pd.read_excel('Task - 2 Description.xlsx', sheet_name='Sheet1')


# In[3]:


df = pd.read_csv('adult.csv')
df


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe(include='all')


# In[7]:


df.shape


# In[8]:


df.describe().T


# In[9]:


df.describe(include='all').T


# In[10]:


df.describe(include='object')


# In[11]:


df.describe(include='int')


# In[12]:


df.income.unique()


# In[13]:


df.age.unique()


# In[14]:


df.workclass.unique()


# In[15]:


df.education.unique()


# In[16]:


sorted(df['educational-num'].unique())


# In[17]:


df['marital-status'].unique()


# In[18]:


df['occupation'].unique()


# In[19]:


df['relationship'].unique()


# In[20]:


df['race'].unique()


# In[21]:


df['gender'].unique()


# In[22]:


df['capital-gain'].unique()


# In[23]:


df['capital-loss'].unique()


# In[24]:


df['hours-per-week'].unique()


# In[25]:


df['native-country'].unique()


# In[26]:


uniqueValues = df.nunique()
print(uniqueValues)


# In[27]:


df['workclass'].value_counts()


# In[28]:


df['occupation'].value_counts()


# In[29]:


df['native-country'].value_counts()


# # Observations : 

# 1. There are 2799 , 2809 and 857 Missing Values in columns - workclass , occupation and native-country , represented as '?'.
# 2. DataSet consists of 48842 Rows and 15 Columns.
# 3. No Null values are present in the Dataset. If any , they are filled with '?'.

# In[30]:


df.boxplot(figsize=(15,15))


# As per the Above BoxPlot , lots of Outliers are found in fnlwgt column.
# Also,There are 2799 , 2809 and 857 Missing Values in columns - workclass , occupation and native-country , represented as '?'. So need to do Data Cleaning first.

# # Data Cleaning : 

# In[31]:


from numpy import nan
df = df.replace('?',nan)
df.head()


# In[32]:


df['income'].value_counts()


# In[34]:


null_values = df.isnull().sum()
null_values


# In[35]:


null_values = df.isnull().sum()
null_values = pd.DataFrame(null_values,columns=['null'])
j=1
sum_total=len(df)
null_values['percentage'] = null_values['null']/sum_total
round(null_values,3).sort_values('percentage',ascending=False)


# Observations : No Null Values are present on Columns other than workclass , relationship , native-country

# In[36]:


df['native-country'].fillna(df['native-country'].mode()[0],inplace = True)
df['workclass'].fillna(df['workclass'].mode()[0],inplace = True)
df['occupation'].fillna(df['occupation'].mode()[0],inplace = True)


# In[37]:


df.isnull().sum()


# # No Null Values present operating null values with all the mode in respective columns.

# # Outlier Detection for each Column :

# One of the methods to detect Outliers is by using Inter-Quartile Range Method(IQR). 
# In this method by using Inter Quartile Range(IQR), we detect outliers. IQR tells us the variation in the data set.Any value, which is beyond the range of -1.5 x IQR to 1.5 x IQR treated as outliers.
# 
# * Q1 represents the 1st quartile/25th percentile of the data.
# * Q2 represents the 2nd quartile/median/50th percentile of the data.
# * Q3 represents the 3rd quartile/75th percentile of the data.
# * (Q1–1.5*IQR) represent the smallest value in the data set and (Q3+1.5*IQR) represnt the largest value in the data set.

# Let's say I want to remove the Outliers for 'age' Column : 

# In[116]:


#AGE OUTLIERS

Q1 = np.percentile(df['age'] , 25)
Q3 = np.percentile(df['age'] , 75)
IQR = Q3 - Q1
ul = Q3+1.5*IQR
ll = Q1-1.5*IQR
age_outliers = df['age'][(df['age'] > ul) | (df['age'] < ll)]
age_outliers.value_counts()


# Similarly, we can remove Outliers for other Columns too :

# In[121]:


#FINAL WEIGHT OUTLIERS

Q1 = np.percentile(df['fnlwgt'] , 25)
Q3 = np.percentile(df['fnlwgt'] , 75)
IQR = Q3 - Q1
ul = Q3+1.5*IQR
ll = Q1-1.5*IQR
fnlwgt_outliers = df['fnlwgt'][(df['fnlwgt'] > ul) | (df['fnlwgt'] < ll)]
fnlwgt_outliers.unique()


# In[123]:


#EDUCATION NUMBERS OUTLIERS

Q1 = np.percentile(df['educational-num'] , 25)
Q3 = np.percentile(df['educational-num'] , 75)
IQR = Q3 - Q1
ul = Q3+1.5*IQR
ll = Q1-1.5*IQR
educationalnum_outliers = df['educational-num'][(df['educational-num'] > ul) | (df['educational-num'] < ll)]
educationalnum_outliers.value_counts()


# In[127]:


# CAPITAL GAIN OUTLIERS

Q1 = np.percentile(df['capital-gain'] , 25)
Q3 = np.percentile(df['capital-gain'] , 75)
IQR = Q3 - Q1
ul = Q3+1.5*IQR
ll = Q1-1.5*IQR
capitalgain_outliers = df['capital-gain'][(df['capital-gain'] > ul) | (df['capital-gain'] < ll)]
capitalgain_outliers.value_counts()


# In[128]:


# CAPITAL LOSS OUTLIERS

Q1 = np.percentile(df['capital-loss'] , 25)
Q3 = np.percentile(df['capital-loss'] , 75)
IQR = Q3 - Q1
ul = Q3+1.5*IQR
ll = Q1-1.5*IQR
capitalloss_outliers = df['capital-loss'][(df['capital-loss'] > ul) | (df['capital-loss'] < ll)]
capitalloss_outliers.value_counts()


# In[131]:


# HOURS PER WEEK OUTLIERS

Q1 = np.percentile(df['hours-per-week'] , 25)
Q3 = np.percentile(df['hours-per-week'] , 75)
IQR = Q3 - Q1
ul = Q3+1.5*IQR
ll = Q1-1.5*IQR
capitalloss_outliers = df['hours-per-week'][(df['hours-per-week'] > ul) | (df['hours-per-week'] < ll)]
capitalloss_outliers.value_counts()


# # Observations :

# All the Number of Outliers for all Numerical Columns have been found and mentioned in Output accordingly.

# # Univariate Analysis : 

# Univariate analysis is the simplest form of analyzing data. “Uni” means “one”, so in other words your data has only one variable. It doesn’t deal with causes or relationships (unlike regression ) and it’s major purpose is to describe; It takes data, summarizes that data and finds patterns in the data.

# # Gender Distribution 

# In[38]:


num_males = df['gender'].where(df['gender'] == 'Male').count()
num_females = df['gender'].where(df['gender'] == 'Female').count()
print('Number of male candidates : {}'.format(num_males))
print('Number of female candidates : {}'.format(num_females))
print('Male to Female ratio: {:.2f}'.format(num_males/num_females))


# In[39]:


sns.displot(df["gender"])
plt.show()


# # Income Distribution

# In[40]:


num_less_than_50k = df['income'].where(df['income'] == '<=50K').count()
num_more_than_50k = df['income'].where(df['income'] == '>50K').count()
print('Number of People with less than 50k Income : {}'.format(num_less_than_50k))
print('Number of People with more than 50k Income : {}'.format(num_more_than_50k))


# In[41]:


sns.countplot(df["income"])
plt.show()


# # Age Distribution

# In[42]:


df['age'].hist(figsize = (5,5))
plt.show


# # Final-Weight Distribution

# In[43]:


df['fnlwgt'].hist(figsize = (5,5))
plt.show()


# # Capital Gain Distribution 

# In[44]:


df['capital-gain'].hist(figsize=(5,5))
plt.show()


# # Capital Loss Distribution

# In[45]:


df['capital-loss'].hist(figsize=(5,5))
plt.show()


# # Hours per Week Distribution

# In[46]:


df['hours-per-week'].hist(figsize=(5,5))
plt.show()


# # Workclass Distribution

# In[47]:


plt.figure(figsize=(20,10))
sns.countplot(x='workclass',data=df)
plt.show()


# # Educational Distribution

# In[48]:


plt.figure(figsize=(20,10))
a= sns.countplot(x='education',data=df)
plt.show()


# # Occupational Distribution

# In[49]:


plt.figure(figsize=(20,8))
ax = sns.countplot(x="occupation", data=df)
plt.show()


# # Observation :

# 1. The Hours Per Week atrribute varies within the range of 1 to 99. By observations , 30-40 hrs people work per week,around 27000 people. There are also few people who works 80-100 hours per week and some less than 20 which is unusual.
# 2. Number of male candidates : 32650 ; Number of female candidates : 16192. Thus , Male to Female ratio: 2.02.
# 3. Number of People with less than 50k Income : 37155. Number of People with more than 50k Income : 11687
# 4. Age Attribute and Final-Weight are Right-Skewed and not Symmetric.
# 5. Capital-gain shows that either a person has no gain or has gain in a very large amount.
# 6. As per Capital-Loss Distribution , most of the Capital observed stands on 0.
# 7. As per Hours per Week ,hours per week atrribute varies within the range of 1 to 99. By observation,30-40 hrs people work per week,around 27000 people. There are also few people who works 80-100 hours per week and some less than 20 which is unusual.
# 8. As per Workclass Distribution ,it seems that most of them working Private sector.
# 9. As per Educational Distribution , HS-grade has the highest and pre school has the min
# 10. As per Occupcational Distribution , Prof-specialty has the maximum count. Armed-Forces has minimum samples in the occupation attribute.

# # Bivariate Analysis :

# We need to check if all Other Parameters have any effect on the Incomes.
# 
# First , lets analyze the Incomes with Categorical Columns.

# # Gender vs Income Plot : 

# In[85]:


plt.figure(figsize=(10,7))
sns.countplot(x="gender", hue="income",data=df)
plt.show()


# # Workclass vs Income Plot :

# In[56]:


fig = plt.figure(figsize=(10,5))
sns.countplot(x='workclass',hue ='income',data=df).set_title("workclass vs count")
plt.show()


# # Relationship vs Income Plot :

# In[57]:


plt.figure(figsize=(10,7))
sns.countplot(x="relationship", hue="income",data=df)
plt.show()


# # Gender vs Occupation  Plot:

# In[58]:


plt.figure(figsize=(20,10))
sns.countplot(x="occupation", hue="gender",data=df)
plt.show()


# # Race vs Income Plot :

# In[59]:


plt.figure(figsize=(20,5))
sns.catplot(y="race", hue="income", kind="count",col="gender", data=df)
plt.show()


# # Observations :

# 1. Income less than 50k from the median by using boxplot it seems that at the age of 34 and Income greater than 50k from the medain by using boxplot it seems that at the age of 42
# 2. People having Income more than 50k and less than 50k are working in Private Sector only
# 3. As per the Relationship - Income Plot , Count for Husbands working is the greatest.
# 4. As per Occupation- Income Plot , in craft sector , transport female workers are so less than male worker. Also, in Armed Forces no Female Workers are seen. Largest Number of Female Employees work in Adm-clerical , and Largest Number of Male Employees work in Craft-Repair.

# Now, lets analyze Incomes with Numerical Columns :

# In[62]:


attributes = ['age' ,'fnlwgt', 'educational-num', 'capital-gain' ,'capital-loss' ,'hours-per-week']
sns.set_style('whitegrid')
fig , axes = plt.subplots(nrows = 3 , ncols = 2 , figsize = (18,20))
fig.subplots_adjust(top = 1.2)
count = 0

for i in range(3):
    for j in range(2):
        sns.boxplot(data = df , x = 'income' , y = attributes[count] , ax = axes[i,j])
        axes[i,j].set_title(f"Income vs {attributes[count]}", fontsize=16)
        count += 1


# # Observations :

# 1) Income vs Age :
#     Age Median Value for Income >50k is higher than those with Income <=50k.
#     Age Range for Income >50k lies between around 35 and 51. And Range for Income <=50k lies between 25 and 48.
# 
# 2) Income vs Final Weight :
#     For both categories in Income, Median Values for Final Weight are almost Same.
#     Number of Outliers for Income <=50k are higher than those for Income>50k.
#     
# 3) Income vs education-num :
#     People having Income>50k have an Educational Experience from 10 - 13 yrs.
#     People having Income<=50k have an Educational Experience from 9 - 10 yrs.
#     Median Value for Employees with Income>50k is 12 Years.
#     
# 4) Income vs capital-gain :
#     No inferences can be made from Capital Gain. Just Outliers are present.
# 
# 5) Income vs capital-loss :
#     No inferences can be made from Capital Gain. Just Outliers are present.
# 
# 6) Income vs hours-per-week :
#     Employees who devote more than 40 hrs per week have a Income more than 50k. And for those with less than 40 hrs per week, they have Income less than 50k.
#     Number of Outliers for Income<=50k is greater than Income>50k.

# In[63]:


sns.pairplot(df)


# # Correlation between Different Parameters : 

# In[67]:


dc = df.corr()
dc


# In[68]:


sns.heatmap(dc,annot=True,center=0)
plt.show()


# # Observations : 

# All the Numerical Variables aren't related much to each other, since they have a Bad Correlation Coefficient.

# # Relation between Gender and Occupation :

# Relation between any 2 Categorical Columns can be found out using Chi Square Testing.
# 
# A chi-square test is a statistical test used to compare observed results with expected results. The purpose of this test is to determine if a difference between observed data and expected data is due to chance, or if it is due to a relationship between the variables you are studying.

# In[69]:


from scipy.stats import chi2
from scipy.stats import chi2_contingency


# In[105]:


df['gender'].value_counts()


# In[72]:


df['occupation'].value_counts()


# In[104]:


# Looking at the freqency distribution

pd.crosstab(df.occupation, df.gender, margins=True)


# In[76]:


# Observed frequencies

observed = pd.crosstab(df.occupation, df.gender)
observed


# In[77]:


# Chi2_contigency returns chi2 test statistic, p-value, degree of freedoms, expected frequencies

chi2_contingency(observed)


# In[78]:


# Computing chi2 test statistic, p-value, degree of freedoms

chi2_test_stat = chi2_contingency(observed)[0]
pval = chi2_contingency(observed)[1]
dr = chi2_contingency(observed)[2]


# In[103]:


confidence_level = 0.95

alpha = 1 - confidence_level
chi2_critical = chi2.ppf(1 - alpha, dr)
chi2_critical


# In[99]:


# Ploting the chi2 distribution to visualise

# Defining the x minimum and x maximum

x_min = 0
x_max = 100

# Ploting the graph and setting the x limits
x = np.linspace(x_min, x_max, 100)
y = chi2.pdf(x, dr)
plt.xlim(x_min, x_max)
plt.plot(x, y)


# Setting Chi2 Critical value 
chi2_critical_right = chi2_critical

# Shading the right rejection region
x1 = np.linspace(chi2_critical_right, x_max, 100)
y1 = chi2.pdf(x1, dr)
plt.fill_between(x1, y1, color='red')


# In[83]:


if(chi2_test_stat > chi2_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# In[102]:


if(pval < alpha):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# # Observations :

# For 95% Confidence , There is a Definite, Consequential Relationship between the two Categorical Columns.

# # Relation between Gender and Income :

# In[220]:


df['income'].value_counts()


# In[221]:


# Looking at the freqency distribution

pd.crosstab(df.income, df.gender, margins=True)


# In[222]:


# Observed frequencies

observed_1 = pd.crosstab(df.income, df.gender)
observed_1


# In[223]:


# Chi2_contigency returns chi2 test statistic, p-value, degree of freedoms, expected frequencies

chi2_contingency(observed_1)


# In[224]:


# Computing chi2 test statistic, p-value, degree of freedoms

chi2_test_stat_1 = chi2_contingency(observed_1)[0]
pval_1 = chi2_contingency(observed_1)[1]
dr_1 = chi2_contingency(observed_1)[2]


# In[225]:


confidence_level = 0.95

beta = 1 - confidence_level
chi2_critical_1 = chi2.ppf(1 - beta, dr_1)
chi2_critical_1


# In[226]:


# Ploting the chi2 distribution to visualise

# Defining the x minimum and x maximum

x_min = 0
x_max = 100

# Ploting the graph and setting the x limits
x_1 = np.linspace(x_min, x_max, 100)
y_1 = chi2.pdf(x_1, dr_1)
plt.xlim(x_min, x_max)
plt.plot(x_1, y_1)


# Setting Chi2 Critical value 
chi2_critical_right_1 = chi2_critical_1

# Shading the right rejection region
x1_1 = np.linspace(chi2_critical_right_1, x_max, 100)
y1_1 = chi2.pdf(x1_1, dr_1)
plt.fill_between(x1_1, y1_1, color='red')


# In[227]:


if(chi2_test_stat_1 > chi2_critical_1):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# In[228]:


if(pval_1 < beta):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# # Observations :

# For 90% Confidence , There is a Definite, Consequential Relationship between the two Categorical Columns.

# # Insights :

# 1. There are 2799 , 2809 and 857 Missing Values in columns - workclass , occupation and native-country , represented as '?'.
# 2. The Hours Per Week atrribute varies within the range of 1 to 99. By observations , 30-40 hrs people work per week,around 27000 people. There are also few people who works 80-100 hours per week and some less than 20 which is unusual.
# 3. Number of male candidates : 32650 ; Number of female candidates : 16192. Thus , Male to Female ratio: 2.02.
# 4. Number of People with less than 50k Income : 37155. Number of People with more than 50k Income : 11687
# 5. Age Attribute and Final-Weight are Right-Skewed and not Symmetric.
# 6. Capital-gain shows that either a person has no gain or has gain in a very large amount.
# 7. As per Capital-Loss Distribution , most of the Capital observed stands on 0.
# 8. As per Hours per Week ,hours per week atrribute varies within the range of 1 to 99. By observation,30-40 hrs people work per week,around 27000 people. There are also few people who works 80-100 hours per week and some less than 20 which is unusual.
# 9. As per Workclass Distribution ,it seems that most of them working Private sector.
# 10. As per Educational Distribution , HS-grade has the highest and pre school has the min
# 11. As per Occupcational Distribution , Prof-specialty has the maximum count. Armed-Forces has minimum samples in the occupation attribute.
# 12. Income less than 50k from the median by using boxplot it seems that at the age of 34 and Income greater than 50k from the medain by using boxplot it seems that at the age of 42
# 13. People having Income more than 50k and less than 50k are working in Private Sector only
# 14. As per the Relationship - Income Plot , Count for Husbands working is the greatest.
# 15. As per Occupation- Income Plot , in craft sector , transport female workers are so less than male worker. Also, in Armed Forces no Female Workers are seen. Largest Number of Female Employees work in Adm-clerical , and Largest Number of Male Employees work in Craft-Repair.
# 16. Income vs Age : 
#     Age Median Value for Income >50k is higher than those with Income <=50k. Age Range for Income >50k lies between around 35 and 51. And Range for Income <=50k lies between 25 and 48.
# 17. Income vs Final Weight : 
#     For both categories in Income, Median Values for Final Weight are almost Same. Number of Outliers for Income <=50k are higher than those for Income>50k.
# 18. Income vs education-num : 
#     People having Income>50k have an Educational Experience from 10 - 13 yrs. People having Income<=50k have an Educational Experience from 9 - 10 yrs. Median Value for Employees with Income>50k is 12 Years.
# 19. Income vs capital-gain : 
#     No inferences can be made from Capital Gain. Just Outliers are present.
# 20. Income vs capital-loss : 
#     No inferences can be made from Capital Gain. Just Outliers are present.
# 21. Income vs hours-per-week : 
#     Employees who devote more than 40 hrs per week have a Income more than 50k. And for those with less than 40 hrs per week, they have Income less than 50k. Number of Outliers for Income<=50k is greater than Income>50k.
# 22. All the Numerical Variables aren't related much to each other, since they have a Bad Correlation Coefficient.
# 23. For 95% Confidence , There is a Definite, Consequential Relationship between Gender and Occupation.
# 24. For 95% Confidence , There is a Definite, Consequential Relationship between Gender and Income.

# # Data Preparation for ML :
a) Train Test Split
b) Encoding for Categorical Columns
   Ordinal : LabelEncoding or OrdinalEncoding
   Nominal : OneHotEncoding or get_dummies
c) Encoding for Numerical Columns
   Standardization (z-transformation)
# We will be following below mentioned steps:
# 
# 1. Identify the Target Variable and Splitting the Data into train and test
# 2. Separating Categorical and Numerical Columns
# 3. Applying OneHotEncoding on Categorical Columns
# 4. Encoding Ordinal Columns
# 5. Rescaling Numerical Columns (Standardization or z-transformation)
# 6. Concatinating the Encoded Categorical Features and Scaled Numerical Features

# # Identify the Target Variable and Splitting the Data into train and Test

# In[136]:


# Identifying the inputs (X) and output (y)

y = df['income']

X = df[['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race','gender','capital-gain','capital-loss','hours-per-week']]


# In[135]:


# split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)


# In[137]:


X_train.head()


# In[138]:


print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)


# # Separating Categorical and Numerical Columns:
# 

# In[139]:


X_train.head()


# In[140]:


X_train.dtypes


# In[141]:


X_train_cat = X_train.select_dtypes(include=['object'])

X_train_cat.head()


# In[142]:


X_train_num = X_train.select_dtypes(include=['int64', 'float64'])

X_train_num.head()


# # Applying OneHotEncoding on Categorical Columns : 

# In[144]:


X_train_cat['workclass'].value_counts(normalize=True)


# In[145]:


X_train_cat['education'].value_counts(normalize=True)


# In[146]:


X_train_cat['marital-status'].value_counts(normalize=True)


# In[147]:


X_train_cat['occupation'].value_counts(normalize=True)


# In[148]:


X_train_cat['relationship'].value_counts(normalize=True)


# In[149]:


X_train_cat['race'].value_counts(normalize=True)


# In[150]:


X_train_cat['gender'].value_counts(normalize=True)


# In[153]:


# OneHotEncoding the categorical features

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)

# column names are (annoyingly) lost after OneHotEncoding
# (i.e. the dataframe is converted to a numpy ndarray)

X_train_cat_ohe = pd.DataFrame(encoder.fit_transform(X_train_cat),columns=encoder.get_feature_names(X_train_cat.columns), index = X_train_cat.index)

X_train_cat_ohe.head()


# In[162]:


get_ipython().system('pip install -U scikit-learn')


# In[163]:


import sklearn

print(sklearn.__version__)


# In[165]:


# OneHotEncoding the categorical features

from sklearn.preprocessing import OneHotEncoder

encoder_ = OneHotEncoder(drop='first', sparse=False)

# column names are (annoyingly) lost after OneHotEncoding
# (i.e. the dataframe is converted to a numpy ndarray)

X_train_cat_ohe_ = pd.DataFrame(encoder_.fit_transform(X_train_cat),columns=encoder_.get_feature_names(X_train_cat.columns),index = X_train_cat.index)

X_train_cat_ohe_.head()


# In[176]:


encoder_.categories_


# # Encoding Ordinal Columns:

# In[177]:


X_train_cat_le = pd.DataFrame(index=X_train_cat.index)

X_train_cat_le.head()


# In[180]:


X_train_cat.workclass.unique()


# In[182]:


workclass_encoder = {'Private' : 1, 'Self-emp-inc' : 2, 'Self-emp-not-inc' : 3, 'Local-gov' : 4, 'State-gov' : 5,'Federal-gov' : 6,'Without-pay' : 7,'Never-worked' : 8}

X_train_cat_le['workclass'] = X_train_cat['workclass'].apply(lambda x : workclass_encoder[x])

X_train_cat_le.head()


# In[183]:


X_train_cat.education.unique()


# In[184]:


education_encoder = {'5th-6th' : 1, 'Assoc-voc' : 2, 'Some-college' : 3, '11th' : 4, '7th-8th' : 5,'HS-grad' : 6,'Doctorate' : 7,'Bachelors' : 8,'Assoc-acdm': 9,'Prof-school' : 10,'10th':11,'12th':12,'Masters':13,'9th':14,'1st-4th':15,'Preschool':16}

X_train_cat_le['education'] = X_train_cat['education'].apply(lambda x : education_encoder[x])

X_train_cat_le.head()


# In[190]:


X_train_cat['marital-status'].unique()


# In[191]:


maritalstatus_encoder = {'Married-civ-spouse' : 1, 'Divorced' : 2, 'Widowed' : 3, 'Separated' : 4, 'Never-married' : 5,'Married-spouse-absent' : 6,'Married-spouse-absent' : 7,'Married-AF-spouse' : 8}

X_train_cat_le['marital-status'] = X_train_cat['marital-status'].apply(lambda x : maritalstatus_encoder[x])

X_train_cat_le.head()


# In[195]:


X_train_cat['occupation'].unique()


# In[194]:


occupation_encoder = {'Transport-moving' : 1, 'Sales' : 2, 'Adm-clerical' : 3, 'Other-service' : 4, 'Craft-repair' : 5,'Prof-specialty' : 6,'Exec-managerial' : 7,'Farming-fishing' : 8,'Machine-op-inspct': 9,'Tech-support' : 10,'Handlers-cleaners':11,'Protective-serv':12,'Priv-house-serv':13,'Armed-Forces':14}

X_train_cat_le['occupation'] = X_train_cat['occupation'].apply(lambda x : occupation_encoder[x])

X_train_cat_le.head()


# In[196]:


X_train_cat['relationship'].unique()


# In[197]:


relationship_encoder = {'Husband' : 1, 'Not-in-family' : 2, 'Other-relative' : 3, 'Own-child' : 4, 'Unmarried' : 5,'Wife' : 6}

X_train_cat_le['relationship'] = X_train_cat['relationship'].apply(lambda x : relationship_encoder[x])

X_train_cat_le.head()


# In[200]:


X_train_cat['race'].unique()


# In[199]:


race_encoder = {'White' : 1, 'Black' : 2, 'Other' : 3, 'Asian-Pac-Islander' : 4, 'Amer-Indian-Eskimo' : 5}

X_train_cat_le['race'] = X_train_cat['race'].apply(lambda x : race_encoder[x])

X_train_cat_le.head()


# In[201]:


X_train_cat['gender'].unique()


# In[202]:


gender_encoder = {'Male' : 1, 'Female' : 2}

X_train_cat_le['gender'] = X_train_cat['gender'].apply(lambda x : gender_encoder[x])

X_train_cat_le.head()


# # Scaling the Numerical Features :

# In[203]:


X_train_num.head()


# In[204]:


# scaling the numerical features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# column names are (annoyingly) lost after Scaling
# (i.e. the dataframe is converted to a numpy ndarray)

X_train_num_rescaled = pd.DataFrame(scaler.fit_transform(X_train_num),columns = X_train_num.columns,index = X_train_num.index)

X_train_num_rescaled.head()


# # Concatinating the Encoded Categorical Features and Scaled Numerical Features:

# In[205]:


X_train_transformed = pd.concat([X_train_num_rescaled, X_train_cat_le], axis=1)

X_train_transformed.head()


# # Preparing Test Data :

# In[206]:


X_test.head()


# In[207]:


X_test.info()


# In[208]:


X_test_cat = X_test.select_dtypes(include=['object'])

X_test_cat.head()


# In[209]:


X_test_num = X_test.select_dtypes(include=['int64', 'float64'])

X_test_num.head()


# In[210]:


X_test_num_rescaled = pd.DataFrame(scaler.transform(X_test_num),columns = X_test_num.columns, index = X_test_num.index)

X_test_num_rescaled.head()


# In[211]:


X_test_cat_le = pd.DataFrame(index = X_test_cat.index)

X_test_cat_le.head()


# In[213]:


X_test_cat_le['workclass'] = X_test_cat['workclass'].apply(lambda x : workclass_encoder[x])

X_test_cat_le['education'] = X_test_cat['education'].apply(lambda x : education_encoder[x])

X_test_cat_le['marital-status'] = X_test_cat['marital-status'].apply(lambda x : maritalstatus_encoder[x])

X_test_cat_le['occupation'] = X_test_cat['occupation'].apply(lambda x : occupation_encoder[x])

X_test_cat_le['relationship'] = X_test_cat['relationship'].apply(lambda x : relationship_encoder[x])

X_test_cat_le['race'] = X_test_cat['race'].apply(lambda x : race_encoder[x])

X_test_cat_le['gender'] = X_test_cat['gender'].apply(lambda x : gender_encoder[x])

X_test_cat_le.head()


# In[214]:


X_test_transformed = pd.concat([X_test_num_rescaled, X_test_cat_le], axis=1)

X_test_transformed.head()

