#!/usr/bin/env python
# coding: utf-8

# # Write a Jupyter Notebook explaining all the Descriptive Statistics. 
# 
# # 1. Mean
# # 2. Median
# # 3. Mode
# # 4. Variance
# # 5. Standard Deviation
# # 6. Correlation
# # 7. Normal Distribution (use references)
# # 8. Feature of Normal Distribution
# # 9. Positively Skewed & Negatively Skewed Normal Distribution
# # 10. Effect on Mean, Median and Mode due to Skewness
# # 11. Explain QQ Plot and show the implementation of the same
# # 12. Explain Box Cox and show the implementation of the same
# # 13. Explain each topic (mentioned above) with the help of images, code   examples (with and without library functions) and formulas  
# 
# # Note :Your Jupyter Notebook should look like a properly documented book.
# 
# # Use this dataset for writing code examples - data.csv

# # Here, 
# 
# 
# # Mthly_HH_Income = Monthly Household Income for a Family
# # Mthly_HH_Expense = Monthly Household Expenses for a Family
# # No_of_Fly_Members = Number of Family Members living in a Family
# # Emi_or_Rent_Amt = Monthly EMI/ Rent amount Deducted for a Family per month
# # Annual_HH_Income = Annual Household Income for a Family
# # Highest_Qualified_Member = Highest Educated/Qualified Member in Family
# # No_of_Earning_Members = Number of Earning Members for a Family

# In[530]:


# Matplotlib and Seaborn are Libraries used for Data Visualisation.
# Numpy Library helps us in Computations related to the Numbers. 
# Pandas library loads the Data and Display the Dataframes
# Statistics Library is a Built-in Python library for Descriptive Statistics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import statistics


# In[531]:


# Importing csv file in Jupyter Notebook
dg = pd.read_csv("data.csv")


# In[532]:


# Reading the File 
dg


# In[533]:


# info method will return the information about the dataset like the non null objects , and the datatype of each of the elements in the data.

dg.info()


# In[534]:


# As can be seen from data.info() , no NULL data is present. Thus , No need to Segregate our Data itself. 
# Also, Data Type for all Columns are INT64 and OBJECT.

# We can check the Number of the Null Values using the isna().sum() method

dg.isna().sum()


# In[535]:


# Describe Function will return the Necessary Information ike Count,Mean,Std Deviation, etc. for the Entire Dataset.
stat = dg.describe()


# In[536]:


# Round off the Data.csv file upto 2 Decimal Places using .round() function - 
stat = stat.round(2)


# In[537]:


stat

Note : 

A quartile divides data into three points— Lower Quartile, Median, and Upper quartile —to form four groups of the dataset. 

The lower quartile, or first quartile, is denoted as Q1 and is the middle number that falls between the smallest value of the dataset and the median.

The second quartile, Q2, is also the Median.

Here 25% , 50% , 75% denote the First , Second and Third Quartiles
# # MEAN
The mean of a dataset is the sum of all values divided by the total number of values. 
It’s the most commonly used measure of central tendency and is often referred to as the “average.”

The mean is the most widely used measure of central tendency because it uses all values in its calculation. 
The best measure of central tendency depends on your type of variable and the shape of your distribution.

The mean can only be calculated for quantitative variables (e.g., height), and it can’t be found for categorical variables (e.g., gender).

In categorical variables, data is placed into groupings without exact numerical values, so the mean cannot be calculated. 
For categorical variables, the mode is the best measure of central tendency because it tells you the most common characteristic or popular choice for your sample.a) Using Library function , mean can be calculated as :
# In[538]:


cols = ['Mthly_HH_Income', 'Mthly_HH_Expense', 'No_of_Fly_Members','Emi_or_Rent_Amt','Annual_HH_Income','No_of_Earning_Members']
print(dg[cols].mean())

b) Without using Library Function , mean can be calculated as :
# In[539]:


n_num = [1, 2, 3, 4, 5]
n = len(n_num)
  
get_sum = sum(n_num)
mean = get_sum / n
  
print("Mean / Average is: " + str(mean))

Similarly , Mean for other Columns can also be calculatedA quartile divides data into three points—a lower quartile, median, and upper quartile—to form four groups of the dataset.

The lower quartile, or first quartile, is denoted as Q1 and is the middle number that falls between the smallest value of the dataset and the median.

The second quartile, Q2, is also the median.

Here 25% , 50% , 75% denote the First , Second and Third Quartiles
# # STANDARD DEVIATION
A standard deviation (or σ) is a measure of how dispersed the data is in relation to the mean. Low standard deviation means data are clustered around the mean, and high standard deviation indicates data are more spread out.

The standard deviation is calculated as the square root of variance by determining each data point's deviation relative to the mean. If the data points are further from the mean, there is a higher deviation within the data set; thus, the more spread out the data, the higher the standard deviation.a) Using Library function , Standard Deviation can be calculated as :
# In[540]:


cols = ['Mthly_HH_Income', 'Mthly_HH_Expense', 'No_of_Fly_Members','Emi_or_Rent_Amt','Annual_HH_Income','No_of_Earning_Members']
print(dg[cols].std())

b) Without using Library Function , Standard Deviation can be calculated as :
# In[541]:


num1 = [1, 2, 3, 4, 5]
no = len(num1)
summ = sum(num1)
mean = summ / no
variance = sum([((x - mean) ** 2) for x in num1]) / len(num1)
res = variance ** 0.5

print("The Standard Deviation of all the Monthly House Incomes is", str(res))


# # VARIANCE
The variance is a measure of variability. It is calculated by taking the average of squared deviations from the mean. 
Variance tells you the degree of spread in your data set. 
The more spread the data, the larger the variance is in relation to the mean.a) Using Library function , Variance can be calculated as :
# In[542]:


cols = ['Mthly_HH_Income', 'Mthly_HH_Expense', 'No_of_Fly_Members','Emi_or_Rent_Amt','Annual_HH_Income','No_of_Earning_Members']
np.var(dg[cols])

b) Without using Library Function , Variance can be calculated as :
# In[543]:


num1 = [1, 2, 3, 4, 5]
no = len(num1)
summ = sum(num1)
mean = summ / no
variance = sum([((x - mean) ** 2) for x in num1]) / len(num1)

print("The Variance of all the Monthly House Incomes is", str(variance))


# # MEDIAN
The median is the middle point in a dataset—half of the data points are smaller than the median and half of the data points are larger.
To find the median:
Arrange the data points from smallest to largest.
a) If the number of data points is Odd, the Median is the middle data point in the list.
b) If the number of data points is Even, the Median is the average of the two middle data points in the list.

Median is sometimes used as opposed to the mean when there are outliers in the sequence that might skew the average of the values.a) Using Library function , Median can be calculated as :
# In[544]:


cols = ['Mthly_HH_Income', 'Mthly_HH_Expense', 'No_of_Fly_Members','Emi_or_Rent_Amt','Annual_HH_Income','No_of_Earning_Members']
dg[cols].median()

b) Without using Library Function , Median can be calculated as :
# In[545]:


num_list = [2,3,4,5,7,9]
# Sort the list
num_list.sort()
# Finding the position of the median
if len(num_list) % 2 == 0:
   first_median = num_list[len(num_list) // 2]
   second_median = num_list[len(num_list) // 2 - 1]
   median = (first_median + second_median) / 2
else:
   median = num_list[len(num_list) // 2]
print(num_list)
print("Median of above list is: " + str(median))


# # MODE
Mode is that number in the list which occurs most frequently. 
We calculate it by finding the frequency of each number present in the list and then choosing the one with highest frequency.a) Using Library function , Mode can be calculated as :
# In[546]:


cols = ['Mthly_HH_Income', 'Mthly_HH_Expense', 'No_of_Fly_Members','Emi_or_Rent_Amt','Annual_HH_Income','No_of_Earning_Members']
dg[cols].mode()

b) Without using Library Function , Mode can be calculated as :
# In[547]:


y= [11, 8, 8, 3, 4, 4, 5, 6, 6, 6, 7, 8]
y.sort()

L=[]
i = 0
while i < len(y) :
    L.append(y.count(y[i]))
    i += 1

d1 = dict(zip(y, L1))

# your d1 will be {3: 1, 4: 2, 5: 1, 6: 3, 7: 1, 8: 3, 11: 1}
# now what you need to do is to filter the k values with the highest v values.

d2={k for (k,v) in d1.items() if v == max(L1) }
print("Mode(s) is/are :" + str(d2))


# # Correlation
Correlation is a statistic that measures the degree to which two securities move in relation to each other. 
Correlations are computed as the correlation coefficient, which has a value that must fall between -1.0 and +1.0.Here are the steps to calculate CORRELATION.

1. Gather data for your "x-variable" and "y variable.
2. Find the mean for the x-variable and find the mean for the y-variable.
3. Subtract the mean of the x-variable from each value of the x-variable. Repeat this step for the y-variable.
4. Multiply each difference between the x-variable mean and x-variable value by the corresponding difference related to  the y-variable.
5. Square each of these differences and add the results.
6. Determine the square root of the value obtained in Step 5.
7. Divide the value in Step 4 by the value obtained in Step 6.Using Library function , Correlation can be calculated as :
# In[548]:


dg.corr()


# In[549]:


# EXAMPLE => 

# Positive correlation - Blue represents Positive Correlation
x = np.arange(start=0, stop=25, step=1)
plt.plot(x, 'o')
# Negative correlation - Orange represents Negative Correlation
x = np.arange(start=25, stop=0, step=-1)
plt.plot(x, 'o')
# No correlation - - Green represents No Correlation
x = np.random.rand(25)
plt.plot(x, 'o')


# In[550]:


# Similarly , considering the Data.csv , Correlation can be found as :
dg.corr().plot(kind='bar')

# kind='bar' represents the Bar Graph as the type of Representation we prefer to choose


# In[551]:


# HeatMap => Another way for showing the Variations between 2 Variables

sns.heatmap(dg.corr(),annot=True)
plt.show()

The Result of the corr() method is a table with a lot of numbers that represents how well the relationship is between two columns.

The number varies from -1 to 1.

1 means that there is a 1 to 1 relationship (a Perfect Correlation), and for this data set, each time a value went up in the first column, the other one went up as well.

A Good Correlation depends on the use, but I think it is safe to say you have to have at least 0.6 (or -0.6) to call it a good correlation.

Other Correlations outside of this Range are probably Bad Correlations. Means that if One value goes Up, then the Other Value goes Down.
# # Normal Distribution
Normal Distribution (also known as Gaussian Distribution) is a type of continuous probability distribution for a real-valued random variable.

A random variable with a Gaussian distribution is said to be normally distributed, and is called a normal deviate.Using Python Libraries , Normal Distribution can be shown as :
# In[552]:


# import required libraries
from scipy.stats import norm

# Creating the distribution
data = np.arange(1,10,0.01)
pdf = norm.pdf(data , loc = 5.3 , scale = 1 )
 
#Visualizing the distribution
 
sb.set_style('whitegrid')
sb.lineplot(data, pdf , color = 'black')
plt.xlabel('Heights')
plt.ylabel('Probability Density')


# # Features of Normal Distribution :
1) The mean-mode-median is in the center.
    It is the mean because it is the ARITHMETIC average of all the scores.
    It is the mode because of all the scores the mean score happens MOST often.
    It is the median because when the scores are displayed from lowest to highest,the mean is the MIDDLE score,the median.
    The EXPECTED value is the mean.
    
2) The frequency curve is bell shaped.
    The bell shape has perfect bilateral symmetry - the left balances exactly with the right.
    The score at -2 is balanced by a score at +2 and the frequencies from 0 to +2 and from 0 to -2 are equal.
    The area under the curve from 0 to +2 is exactly the same as the area under the curve from 0 to -2.
    Fifty percent of the scores are above the mean and 50% are below the mean.
    
3) The probability a score is above the mean is 50% and the probability a score is below the mean is 50%.

4) Most of the scores are in the middle, about the mean, and few are in the tails, at the extremes.

5) The area under the curve is equal to 1.

6) The probability of an event that does not happen is 0.

7) The sum of the probabilities of all events is 1.

8) The standard deviation tells one how the scores are spread out and therefore the fatness or skinniness of the bell.

9) Because "the shape" of one normal distribution is "the shape" of all others, the spread of the area of one normal     distribution "is the same" as all others and the standard deviation is the scale.

10) The frequencies for the set of scores with a normal distribution are stated by a function which includes as controlling features both the mean, µ, and the standard deviation, , of the set of scores.

# # Positively Skewed & Negatively Skewed Normal Distribution
In the field of Statistics, we use Skewness to describe the symmetry of a distribution.

1) We say that a distribution of data values is left skewed if it has a “tail” on the left side of the distribution.
2) We say that a distribution is right skewed if it has a “tail” on the right side of the distribution.
3) And we say a distribution has no skew if it’s symmetrical on both sides.

The value for skewness can range from Negative Infinity to Positive Infinity.
Here’s how to interpret skewness values:

a) A Negative value for Skewness indicates that the Tail is on the left side of the Distribution, 
   which extends towards more negative values. (Skewness < 0)

b) A Positive value for Skewness indicates that the Tail is on the right side of the Distribution, 
   which extends towards more positive values. (Skewness > 0)

c) A value of Zero indicates that there is no Skewness in the distribution at all, 
   meaning the distribution is perfectly symmetrical. (Skewness = 0)

Positively skewed data will have a mean that is higher than the median. 
The mean of negatively skewed data will be less than the median, which is the exact reverse of what happens in a positively skewed distribution. No matter how long or fat the tails are, the distribution exhibits zero skewness if the data is graphed symmetrically.To find the Skewness of Data using Python, we import the following Library :
# In[553]:


from scipy.stats import skew


# In[554]:


# EXAMPLE => 

x= np.random.normal(0,5,10)
print("X:",x)
print("Skewness for Data :",skew(x))


# In[555]:


# Dropping the 'Highest_Qualified_Member' column, since Str column.
df = dg.drop(['Highest_Qualified_Member'], axis=1)


# In[556]:


df


# In[557]:


for col in df:
    print(col)
    print(skew(df[col]))
    
    plt.figure()
    sns.histplot(df[col])
    plt.show()


# # Effect on Mean, Median and Mode due to Skewness
A distribution in which the values of mean, median and mode coincide (i.e. mean = median = mode) is known as a symmetrical distribution.

Conversely, when values of mean, median and mode are not equal the distribution is known as asymmetrical or skewed distribution.

1) If the distribution of data is skewed to the left, the mean is less than the median, which is often less than the mode. 
2) If the distribution of data is skewed to the right, the mode is often less than the median, which is less than the   mean.

Outs of all the Parameters (Mean,Median,Mode) 
The Mean tends to reflect skewing the most because it is affected the most by Outliers.
# # QQ Plot
Since normal distribution is of so much importance, we need to check if the collected data is normal or not. 
So we will use the Q-Q plot to check the normality of skewness of data. 
Q stands for quantile and therefore, Q-Q plot represents quantile-quantile plot.

QQ plot is a Scatter Plot created by plotting 2 sets of Quantiles against one another. If the Quantiles are from the same distribution , then we can observe almost a Straight Line.
Most common use for QQ plot is to check if a sample data is from a Normal Distribution or not.

When the whole data is sorted, 50th quantile means 50% of the data falls below that point and 50% of the data falls above that point. 
That is the median point. When we say 1st quantile, only 1% of the data falls below that point and 99% is above that.
25th and 75th quantile points are also known as quartiles. There are three quartiles is the dataset.

Q1 = first quartile = 25th quantile
Q2 = second quartile = 50th quantile = median
Q3 = third quartile = 75th quantile

(Quantile are sometimes called percentile)The x-axis of a Q-Q plot represents the quantiles of standard normal distribution. 
Let’s say we have a normal data and we want to standardize it. 
Standardizing means subtracting mean from each data point and dividing it by standard deviation. 
The resultant is also known as z-score. If we sort those z-scores and plot, then we observe that:
Statistically, 99.7& of the data falls between this range.The first step to find the x-axis values of Q-Q plot is to determine the quantiles/percentiles of this normally distributed standard data. 
This way we can obtain the quantiles which are pretty much standard across all Q-Q plots. 
When we use these z-scores, the x-axis will roughly stretch from -3 to +3.

Once we obtain the values to plot along x-axis, we then need to apply the same method for our data of interest. 
Therefore, we will plot the z-scores of out data against the z-scores of the standard normal data. 
If our data is normal, the plot will be a straight line since we are plotting very close values against each other. 
If the data is not normally distributed, the line will deviate from the straight track and signal different scenarios.Note : If the data is skewed on high side (as shown above), we will obtain a Q-Q plot.
       If the data exhibits bimodality, the Q-Q plot will signal that too.
       
Python has statsmodels library which has handy qqplot module to use.
# # Implementation of QQ PLOT :-

# In[558]:


import statsmodels.api as sm

for col in df:
    print(col)
    
    plt.figure()
    sm.qqplot(df[col],line='45')
    plt.show()


# # BOX-COX Plot
Imagine you are watching a horse race and like any other race, there are fast runners and slow runners. 
So, logically speaking, the horse which came first and the fast horses along with it will have the smaller difference of completion time whereas the slowest ones will have a larger difference in their completion time. 

We can relate this to a very famous term in statistics called variance which refers to how much is the data varying with respect to the mean.
Here in our example, there is an inconsistent variance (Heteroscedasticity) between the fast horses and the slow horses because there will be small variations for shorter completion time and vice versa. 

Hence, the distribution for our data will not be a bell curve or normally distributed as there will be a longer tail on the right side. 
These types of distributions follow Power law or 80-20 rule where the relative change in one quantity varies as the power of another.

When we plot this , we can see that Power-Law Distribution which is having peaked for short running times because of the small variance and heavy tail due to longer running times. 
These power-law distributions are found in the field of physics, biology, economics, etc.So, just think for a second that if these distributions are found in so many fields, what if we could transform it to a much comfortable distribution like normal distribution? That would make our life a lot easier. 
Fortunately, we have a way to transform power-law or any non-linear distribution to normal using a Box-Cox Transformation.

It is clear that if somehow we could inflate the variability for the left-hand side of non-normal distribution i.e peak and reduce the variability at the tails. 
In short, trying to move the peak towards the centre then we can get a curve close to the bell-shaped curve. 

A Box cox transformation is defined as a way to transform non-normal dependent variables in our data to a normal shape through which we can run a lot more tests than we could have.
# # a) Mathematics behind Box-Cox Transformation
How can we convert our intuitive thinking into a mathematical transformation function? 
Logarithmic transformation is all we need.

When a log transformation is applied to non-normal distribution, it tries to expand the differences between the smaller values because the slope for the logarithmic function is steeper for smaller values whereas the differences between the larger values can be reduced because, for large values, log distribution has a moderate slope.

That is what we thought of doing, right? Box-cox Transformation only cares about computing the value of lambda  which varies from – 5 to 5. A value of lambda is said to be best if it is able to approximate the non-normal curve to a normal curve.

The transformation equation is as follows:

y(lambda)=(y^(lambda)-1)/lambda , if lambda != 0
         = (log(y))             , if lambda =0 

This function requires input to be positive. 
Using this formula manually is a very laborious task thus many popular libraries provide this function.
# # b) Syntax for BoxCox :
scipy.stats.boxcox(x, lmbda=None, alpha=None, optimizer=None)

Return a dataset transformed by a Box-Cox power transformation.

Input Parameters :
    
x : ndarray
    Input array to be transformed.
    If lmbda is not None, this is an alias of scipy.special.boxcox. Returns nan if x < 0; returns -inf if x == 0 and lmbda     < 0.
    If lmbda is None, array must be positive, 1-dimensional, and non-constant.

lmbda : scalar, optional
    If lmbda is None (default), find the value of lmbda that maximizes the log-likelihood function and return it as the       second output argument.
    If lmbda is not None, do the transformation for that value.

alpha: float, optional
    If lmbda is None and alpha is not None (default), return the 100 * (1-alpha)% confidence interval for lmbda as the         third output argument. Must be between 0.0 and 1.0.
    If lmbda is not None, alpha is ignored.

optimizer: callable, optional
    If lmbda is None, optimizer is the scalar optimizer used to find the value of lmbda that minimizes the negative log-       likelihood function. optimizer is a callable that accepts one argument:

fun: callable
    The objective function, which evaluates the negative log-likelihood function at a provided value of lmbda 
    and returns an object, such as an instance of scipy.optimize.OptimizeResult, which holds the optimal value of lmbda in     an attribute x.
    If lmbda is not None, optimizer is ignored.

Returns : 
    
boxcox : ndarray
    Box-Cox power transformed array.

maxlog : float, optional
    If the lmbda parameter is None, the second returned argument is the lmbda that maximizes the log-likelihood function.

(min_ci, max_ci) : tuple of float, optional
    If lmbda parameter is None and alpha is not None, this returned tuple of floats represents the minimum and maximum         confidence limits given alpha.
# # c) Implementation
SciPy’s stats package provides a function called boxcox for performing box-cox power transformation that takes in original non-normal data as input and returns fitted data along with the lambda value that was used to fit the non-normal distribution to normal distribution. 
# # d) Example

# In[559]:


from scipy import stats

# generate non-normal data (exponential)
original_data = np.random.exponential(size = 1000)
 
# transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(original_data)
 
# creating axes to draw plots
fig, ax = plt.subplots(1, 2)
 
# plotting the original data(non-normal) and fitted data (normal)
sns.distplot(original_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Non-Normal", color ="red", ax = ax[0])
 
sns.distplot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Normal", color ="green", ax = ax[1])
 
# adding legends to the subplots
plt.legend(loc = "upper right")
 
# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)
 
print(f"Lambda value used for Transformation: {fitted_lambda}")

