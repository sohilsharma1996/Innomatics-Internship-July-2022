#!/usr/bin/env python
# coding: utf-8

# # DATA DESCRIPTION :

# - The dataset was released by Aspiring Minds from the Aspiring Mind Employment Outcome 2015 (AMEO). The study is primarily limited only to students with engineering disciplines.
# 
# - The dataset contains the employment outcomes of engineering graduates as dependent variables (Salary, Job Titles, and Job Locations) along with the standardized scores from three different areas – cognitive skills, technical skills and personality skills.
# 
# - The dataset also contains demographic features. The dataset contains around 40 independent variables and 4000 data points.
# 
# - The independent variables are both continuous and categorical in nature. The dataset contains a unique identifier for each candidate.

# # PROBLEM STATEMENT :

# - Times of India article dated Jan 18, 2019 states that “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.” 
#   
#   Test this claim with the data given to you.
#   
# 
# - Is there a relationship between gender and specialisation? (i.e. Does the preference of Specialisation depend on the Gender?)

# # About Dataset :

# The Data contains Information like - ID , Salary , Designation , JobCity , and others :

# BASED ON THE DATA -
# 
# We will try to identify the characteristics of the Target Audience for each type of Specialization and Designation offered by the company based on their AMCAT Scores, to provide a Better Recommendation to the Future Audience.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', None)


# # Representing the DataType and Description for all the Variables :

# In[2]:


pd.read_excel('aspiring_minds_employability_outcomes_2015.xlsx', sheet_name='Sheet2')


# In[3]:


df = pd.read_excel('aspiring_minds_employability_outcomes_2015.xlsx', sheet_name='Sheet1')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe(include='all')


# In[7]:


df['DOL'].describe()


# In[8]:


print(f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}")


# In[9]:


df['Designation'].nunique()


# In[10]:


df['JobCity'].nunique()


# In[11]:


df['CollegeCityID'].nunique()


# In[12]:


df['CollegeID'].nunique()


# In[13]:


df['Degree'].unique()


# In[14]:


sorted(df['CollegeTier'].unique())


# In[15]:


sorted(df['CollegeCityTier'].unique())


# In[16]:


df['Specialization'].nunique()


# In[17]:


df['Specialization'].describe()


# In[18]:


df['Specialization'].unique()


# In[19]:


df['CollegeState'].nunique()


# In[20]:


df['Gender'].unique()


# In[21]:


gy_list = sorted(df['GraduationYear'].unique())


# # Observations :

# 1. As can be seen in df.info(), there are no missing values (null values )in the data. 
# 2. The Dataset comprises of the AMCAT exams held from year 2007 to 2017 , except for 2008.
# 3. A total of 46 Specializations are available for 3998 Students included in the Dataset, who have completed either of the Degrees : B.Tech/B.E. , MCA, M.Tech./M.E. , M.Sc. (Tech.)
# 4. A total of 418 unique Designations are designated among all students in 339 Cities or 26 States.
# 5. Number of Unique ID's identifying the college which the candidate attended is equal to the Number of Unique ID's to identify the city in which the college is located in.
# 6. Maximum Designations among all the Students in this Dataset is for Software Engineer (540).
# 7. Maximum number of Students have their Job City as Bangalore(627).
# 8. Majority of the Students are still working in their Respective Companies(1875).
# 9. Majority of the Students completed their Specialization in Electronics and Communication Engineering.

# In[22]:


df = df.drop(['ID'], axis=1)


# In[23]:


df


# # UNIVARIATE ANALYSIS :

# Univariate analysis is the simplest form of analyzing data. “Uni” means “one”, so in other words your data has only one variable. It doesn’t deal with causes or relationships (unlike regression ) and it’s major purpose is to describe; It takes data, summarizes that data and finds patterns in the data.
# 
# We have to first understand the distribution of the data for the following attributes(of INT datatype) :
#     
# 1. Salary
# 2. Designation
# 3. 10percentage
# 4. 12percentage
# 5. 12graduation 
# 6. Specialization
# 7. GraduationYear
# 8. JobCity

# In[24]:


num_males = df['Gender'].where(df['Gender'] == 'm').count()
num_females = df['Gender'].where(df['Gender'] == 'f').count()
print('Number of male candidates : {}'.format(num_males))
print('Number of female candidates : {}'.format(num_females))
print('Male to Female ratio: {:.2f}'.format(num_males/num_females))


# In[25]:


sns.displot(df["Gender"])
plt.show()


# In[26]:


df['CollegeTier'].value_counts().plot(kind='kde', figsize=(5,5))


# In[27]:


df['CollegeTier'].value_counts()


# In[28]:


df['CollegeTier'].value_counts().plot(kind='bar', figsize=(5,5))


# In[29]:


df['CollegeCityTier'].value_counts().plot(kind='kde', figsize=(5,5))


# In[30]:


df['CollegeCityTier'].value_counts()


# In[31]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
s = sns.countplot(data=df, x='CollegeCityTier')

s.set_title("CollegeCityTier - Count", fontsize=16)
plt.show()


# In[32]:


fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
fig.subplots_adjust()

sns.histplot(data=df, x="10percentage", kde=True, ax=axis[0,0])
sns.histplot(data=df, x="12percentage", kde=True, ax=axis[0,1])
sns.histplot(data=df, x="12graduation", kde=True, ax=axis[0,2])
sns.histplot(data=df, x="CollegeTier", kde=True, ax=axis[1,0])
sns.histplot(data=df, x="CollegeCityTier", kde=True, ax=axis[1,1])

plt.show()


# In[33]:


df['12graduation'].value_counts().sort_values(ascending=False)


# In[34]:


df['collegeGPA'].value_counts().sort_values(ascending=False)


# In[35]:


df['12board'].value_counts().sort_values(ascending=False)


# In[36]:


df['JobCity'].value_counts().sort_values(ascending=False)


# # Observations : 

# 1. Majority of the Students are Males(3041) and the rest 957 are Females considered for this DataSet.
# 2. Male to Female Ratio = 3.18
# 3. Maximum Number of Students got around 88% in Class 10th and 70% in Class 12th.
# 4. Maximum Number of Students graduated from Class 12th between Mid 2007 and 2010.
# 5. Majority of the Colleges have Tier = 2 (3701). Rest of them have Tier = 1 (297). 
# 6. Majority of the Cities(where colleges are present) have Tier = 0 (2797). Rest of them have Tier = 1 (1201).

# # Outliers detection using BoxPlots :

# In[37]:


fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
fig.subplots_adjust(top=2.5,bottom = 1.84)

sns.boxplot(data=df, x="Salary", orient='h', ax=axis[0,0])
sns.boxplot(data=df, x="10percentage", orient='h', ax=axis[0,1])
sns.boxplot(data=df, x="12percentage", orient='h', ax=axis[0,2])
sns.boxplot(data=df, x="collegeGPA", orient='h', ax=axis[1,0])
sns.boxplot(data=df, x="12graduation", orient='h', ax=axis[1,1])
plt.show()


# # Observations :

# 1. 12percentage has only 1 Outlier
# 2. Salary , 10percentage , 12graduation and collegeGPA have many Outliers.

# Also, Calculating the Data Percentage - Wise , we have :

# In[38]:


df1 = df[['Gender', '12graduation', 'Degree']].melt()


# In[39]:


df1


# In[40]:


df2 = df1.groupby(['variable', 'value'])[['value']].count()


# In[41]:


df2


# In[42]:


df2 / len(df)


# # Observations : 
1. 12graduation :
    
    26% Students completed their 12th class in Year 2009 (Maximum)
    0.025% Students completed their 12th class in Year 1995 , 1998 and 1999 (Minimum)

2. Degree :
    
    92% Students have considered B.Tech/B.E. as their Graduation Option (Maximum)
    0.05% Students opted for M.Sc. (Tech.) (Minimum)
    
3. Gender :
    
    76% Students are Males. 24% Students are Females.
    
# # Bivariate Analysis :

# We need to check if all Other Parameters have any effect on the Salaries.
First we will check if the Following Parameters have any effect on Salaries or not :
    
1. Gender - Candidate’s gender (Male / Female)

2. Degree - Degree obtained/pursued by the candidate (B.Tech/B.E. , MCA , M.Tech./M.E. , M.Sc. (Tech.))

3. CollegeTier - Tier of college (1 or 2)

4. CollegeCityTier - The tier of the city in which the college is located (0 or 1)

5. 12graduation - Year of graduation - senior year high school

6. GraduationYear - Year of graduation (Bachelor’s degree)
# In[43]:


attributes = ['Gender','Degree','CollegeTier','CollegeCityTier','12graduation','GraduationYear']
sns.set_style('whitegrid')
fig , axes = plt.subplots(nrows = 3 , ncols = 2 , figsize = (18,20))
fig.subplots_adjust(top = 1.2)
count = 0

for i in range(3):
    for j in range(2):
        sns.boxplot(data = df , y = 'Salary' , x = attributes[count] , ax = axes[i,j])
        axes[i,j].set_title(f"Salary vs {attributes[count]}", fontsize=16)
        count += 1


# # Observations : 
1) Salary vs Gender :
    Salaries for Male and Female have the Same Salary Median Values.
    Maximum Salary for Males seems to be higher than those for Females.
    Male Salary have more Outliers than Female Salary.

2) Salary vs Degree :
    Salaries for M.Tech and M.Sc(Tech) Students have the Same Salary Median Values.
    B.Tech Degree have the highest number of Outliers than the rest of the Degrees.
    Salary Range for M.Tech Students is greater than those of B.Tech Students.
    
3) Salary vs CollegeTier :
    Salary Range for Tier 1 Colleges is greater than Tier 2 Colleges.
    Tier 2 Colleges have more Outliers than Tier 1 Colleges.
    
4) Salary vs CollegeCityTier :
    Salary Range for Tier 0 Cities is slightly greater than Tier 1 Cities.
    Tier 0 Cities have more Outliers than Tier 1 Cities.
    Salary for Tier 0 and Tier 1 College Cities have the same Salary Median Values.

5) Salary vs 12graduation :
    Off all the years considered , Salary Range for the Year 2006 is the largest.
    Salary Median Values for Years 2003 , 2004 , 2005 , 2007 , 2008 , 2009 , 2010 and 2011 are the Same.
    
6) Salary vs GraduationYear :
    Off all the years considered , Salary Range for the Year 2010 is the largest.
    Salary Median Values for Years 2009 , 2013 , 2014 and 2015 are the Same.
# Now, we will explore Relations between other Numerical Columns like AMCAT Scores wrt Salary as well : 

# In[44]:


plt.scatter(x='Quant', y='Salary', data = df)


# In[45]:


plt.scatter(x='English' , y='Salary', data = df)


# In[46]:


plt.scatter(x='Logical' , y='Salary', data = df)


# In[47]:


plt.scatter(x='Domain' , y='Salary', data = df)


# In[48]:


plt.scatter(x='ComputerProgramming' , y='Salary', data = df)


# In[49]:


plt.scatter(x='ComputerScience' , y='Salary', data = df)


# # Observations :

# 1. Quant vs Salary =>
#     
#    Most of the Quant Scores are lying between 0 and 1(e^6) values for Salaries.
#    Outliers are present beyond this Salary Range.
#     
#     
# 2. English vs Salary =>
# 
#    Most of the English Scores are lying between 0 and 1(e^6) values for Salaries.
#    Outliers are present beyond this Salary Range.
# 
# 
# 3. Logical vs Salary =>
# 
#    Most of the Logical Scores are lying between 0 and 1(e^6) values for Salaries.
#    Outliers are present beyond this Salary Range.
# 
# 
# 4. Domain vs Salary =>
# 
#    Most of the Domain Scores are lying between 0 and 1(e^6) values for Salaries.
#    Outliers are present beyond this Salary Range. Also, Values of (-1) are an Outlier , where values weren't mentioned.
#    
#    
# 5. ComputerProgramming vs Salary =>
# 
#    Most of the ComputerProgramming Scores are lying between 0 and 1(e^6) values for Salaries.
#    Outliers are present beyond this Salary Range. Many outliers are present at ZERO Scores.
#    
#    
# 6. ComputerScience vs Salary =>
# 
#    Most of the ComputerProgramming Scores are lying between 0 and 1(e^6) values for Salaries.
#    Outliers are present beyond this Salary Range. Many outliers are present at ZERO Scores.

# # Correlation between Different Parameters :

# In[50]:


dc = df.corr()
dc


# In[51]:


plt.figure(figsize= (7,7), dpi=100)
sns.heatmap(dc)


# # Observations :

# 1. CollegeGPA matters a bit less than 10th Percentage and 12th Percentage , when comparing it wrt Salary.
# 2. CollegeCityTier matters more in comparison to CollegeTier , when considering a Student's Salary.
# 3. Domain and ComputerProgramming have good correlation, considering Domain Section. This means that in Domain Test, Problems related to Computer Programming will be asked the most, besides Domain Questions.
# 4. Computer Science was also considered for 12th Graduation for a good part.

# # Relation between Gender and Specialization :

# Relation between any 2 Categorical Columns can be found out using Chi Square Testing.
# 
# A chi-square test is a statistical test used to compare observed results with expected results. The purpose of this test is to determine if a difference between observed data and expected data is due to chance, or if it is due to a relationship between the variables you are studying.

# In[52]:


from scipy.stats import chi2
from scipy.stats import chi2_contingency


# In[53]:


df['Gender'].value_counts()


# In[54]:


df['Specialization'].value_counts()


# In[55]:


# Looking at the freqency distribution

pd.crosstab(df.Specialization, df.Gender, margins=True)


# In[56]:


# Observed frequencies

observed = pd.crosstab(df.Specialization, df.Gender)
observed


# In[57]:


# Chi2_contigency returns chi2 test statistic, p-value, degree of freedoms, expected frequencies

chi2_contingency(observed)


# In[58]:


# Computing chi2 test statistic, p-value, degree of freedoms

chi2_test_stat = chi2_contingency(observed)[0]
pval = chi2_contingency(observed)[1]
dr = chi2_contingency(observed)[2]


# In[59]:


confidence_level = 0.95

alpha = 1 - confidence_level
chi2_critical = chi2.ppf(1 - alpha, dr)
chi2_critical


# In[60]:


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


# In[61]:


if(chi2_test_stat > chi2_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# In[62]:


if(pval < alpha):
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# # Observations :

# For 95% Confidence , There is a Definite, Consequential Relationship between the two Categorical Columns.

# # Summarising the Specializations:

# In[63]:


df


# In[64]:


specialization_map = {'electronics and communication engineering' : 'EC',
 'computer science & engineering' : 'CS',
 'information technology' : 'CS' ,
 'computer engineering' : 'CS',
 'computer application' : 'CS',
 'mechanical engineering' : 'ME',
 'electronics and electrical engineering' : 'EC',
 'electronics & telecommunications' : 'EC',
 'electrical engineering' : 'EL',
 'electronics & instrumentation eng' : 'EC',
 'civil engineering' : 'CE',
 'electronics and instrumentation engineering' : 'EC',
 'information science engineering' : 'CS',
 'instrumentation and control engineering' : 'EC',
 'electronics engineering' : 'EC',
 'biotechnology' : 'other',
 'other' : 'other',
 'industrial & production engineering' : 'other','chemical engineering' : 'other',
 'applied electronics and instrumentation' : 'EC',
 'computer science and technology' : 'CS',
 'telecommunication engineering' : 'EC',
 'mechanical and automation' : 'ME',
 'automobile/automotive engineering' : 'ME',
 'instrumentation engineering' : 'EC',
 'mechatronics' : 'ME',
 'electronics and computer engineering' : 'CS',
 'aeronautical engineering' : 'ME',
 'computer science' : 'CS',
 'metallurgical engineering' : 'other',
 'biomedical engineering' : 'other',
 'industrial engineering' : 'other',
 'information & communication technology' : 'EC',
 'electrical and power engineering' : 'EL',
 'industrial & management engineering' : 'other',
 'computer networking' : 'CS',
 'embedded systems technology' : 'EC','power systems and automation' : 'EL',
 'computer and communication engineering' : 'CS',
 'information science' : 'CS',
 'internal combustion engine' : 'ME',
 'ceramic engineering' : 'other',
 'mechanical & production engineering' : 'ME',
 'control and instrumentation engineering' : 'EC',
 'polymer technology' : 'other',
 'electronics' : 'EC'}


# In[65]:


df['Specialization'] = df['Specialization'].map(specialization_map)
df['Specialization'].unique()


# In[66]:


df


# In[67]:


# df['Specialization'] = df['Specialization'].map(specialization_map)
df['Specialization'].value_counts().plot(kind='bar', figsize=(15,5))
print(df['Specialization'].unique())


# In[68]:


df['Specialization'].value_counts()


# # Observations :

# 1. A total of 2289 Students are from Computer Science Specialization.
# 2. A total of 1319 Students are from Electronics Specialization.
# 3. A total of 220 Students are from Mechanical Specialization.
# 4. A total of 85 Students are from Electrical Specialization.
# 5. A total of 56 Stundets are from Other Specializations.
# 6. A total of 29 Students are from Civil Engineering Specializations.

# # Times of India article dated Jan 18, 2019 states that “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.” Test this claim with the data given to you.

# In[76]:


sns.boxplot(data = df , y = 'Specialization' , x = 'Salary' )


# # Conclusions :

# Yes , its definitely possible for a Candidate with Computer Science Engineering to earn upto 2.5-3 lakhs as a fresh graduate, since the Entire Range for Salary for CS Students lie between 0 and around 7,00,000. Also, a lot of Outliers are present for Salaries for CS Students too.

# # Insights :

# 1. Majority of the Students are Males(3041) and the rest 957 are Females considered for this DataSet (76% Students are Males. 24% Students are Females).
# 2. Male to Female Ratio = 3.18
# 3. Maximum Number of Students got around 88% in Class 10th and 70% in Class 12th.
# 4. Maximum Number of Students graduated from Class 12th between Mid 2007 and 2010.
# 5. Majority of the Colleges have Tier = 2 (3701). Rest of them have Tier = 1 (297). 
# 6. Majority of the Cities(where colleges are present) have Tier = 0 (2797). Rest of them have Tier = 1 (1201).
# 7. 26% Students completed their 12th class in Year 2009 (Maximum). 0.025% Students completed their 12th class in Year 1995 , 1998 and 1999 (Minimum)
# 8. 92% Students have considered B.Tech/B.E. as their Graduation Option (Maximum) 0.05% Students opted for M.Sc. (Tech.) (Minimum)
# 9. Salary vs Gender :
#         Salaries for Male and Female have the Same Salary Median Values.
#         Maximum Salary for Males seems to be higher than those for Females.
#         Male Salary have more Outliers than Female Salary.
# 10. Salary vs Degree :
#         Salaries for M.Tech and M.Sc(Tech) Students have the Same Salary Median Values.
#         B.Tech Degree have the highest number of Outliers than the rest of the Degrees.
#         Salary Range for M.Tech Students is greater than those of B.Tech Students.   
# 11. Salary vs CollegeTier :
#         Salary Range for Tier 1 Colleges is greater than Tier 2 Colleges.
#         Tier 2 Colleges have more Outliers than Tier 1 Colleges.
# 12. Salary vs CollegeCityTier :
#         Salary Range for Tier 0 Cities is slightly greater than Tier 1 Cities.
#         Tier 0 Cities have more Outliers than Tier 1 Cities.
#         Salary for Tier 0 and Tier 1 College Cities have the same Salary Median Values.
# 13. Salary vs 12graduation :
#         Off all the years considered , Salary Range for the Year 2006 is the largest.
#         Salary Median Values for Years 2003 , 2004 , 2005 , 2007 , 2008 , 2009 , 2010 and 2011 are the Same.
# 14. Salary vs GraduationYear :
#         Off all the years considered , Salary Range for the Year 2010 is the largest.
#         Salary Median Values for Years 2009 , 2013 , 2014 and 2015 are the Same.
# 15. CollegeGPA matters a bit less than 10th Percentage and 12th Percentage , when comparing it wrt Salary.
# 16. CollegeCityTier matters more in comparison to CollegeTier , when considering a Student's Salary.
# 17. Domain and ComputerProgramming have good correlation, considering Domain Section. This means that in Domain Test, Problems related to Computer Programming will be asked the most, besides Domain Questions.
# 18. Computer Science was also considered for 12th Graduation for a good part.
# 19. A total of 2289,1319,220,85,56,29 Students are from Computer Science , Electronics , Mechanical , Electrical , Other and Civil Specialization respectively.

# # Recommendations :

# 1. Colleges should focus most on educating their Students with Computer Science , so as to get Good PLacements, since the count for maximum students belong to CS Specialization.
# 2. As per Chi - Square Test done , Gender impacts Specialization , since majority of the Students chose CS and EC Specializations.
# 3. Yes , its definitely possible for a Candidate with Computer Science Engineering to earn upto 2.5-3 lakhs as a fresh graduate, since the Entire Range for Salary for CS Students lie between 0 and around 7,00,000. Also, a lot of Outliers are present for Salaries for CS Students too.
