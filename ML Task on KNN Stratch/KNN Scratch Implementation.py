#!/usr/bin/env python
# coding: utf-8

# # DATA DESCRIPTION :

# 1. Title : Diamonds Dataset
#     
# 2. The 7th Column "price" is the values to be predicted.
# 
# 3. Data Type : Mixed ( Numerical + Categorical)
#     
# 4. Dataset has nearly 54000 Instances.
# 
# 5. It has 10 Features.
# 
# 6. Features 
# 
#     price : price in US dollars (\$326--\$18,823)
#     
#     carat : weight of the diamond (0.2--5.01)
#     
#     cut   : quality of the cut (Fair, Good, Very Good, Premium, Ideal)
#     
#     color : diamond colour, from J (worst) to D (best)
#     
#     clarity : a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
#     
#     x : length in mm (0--10.74)
#     
#     y : width in mm (0--58.9)
#     
#     z : depth in mm (0--31.8)
#     
#     depth : total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
#     
#     table : width of top of diamond relative to widest point (43--95)
# 
# 7. Caution : Handle Categorical data before building a model.

# # PROBLEM STATEMENT :

# Task is to predict the Diamond Price.💎
# 
# Write the KNN code from scratch and make it work on the given dataset.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# In[3]:


df = pd.read_csv('diamonds.csv')


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.rename(columns={'x':'length','y':'width','z':'depth','depth':'depth%','table':'table%'},inplace=True)


# In[7]:


df.head()


# In[8]:


df['L/W']= df['length']/df['width']


# In[9]:


df.info()


# In[10]:


df.shape


# In[11]:


print(df['cut'].unique().tolist())
print(df['clarity'].unique().tolist())
print(df['color'].unique().tolist())


# In[12]:


df.describe()


# In[13]:


df.describe(include='all')


# In[14]:


df.isnull().sum()


# Now since Length , Width , L/W columns haver Null Values, we need to replace them too.

# In[15]:


df[['length','width','depth','L/W']]=df[['length','width','depth','L/W']].replace(0,np.NaN)


# In[16]:


df.dropna(inplace=True)


# In[17]:


df.isnull().sum()


# In[56]:


# Make a Split of Numerical and Categorical columns
numerical = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
categorical = df.loc[:, ~df.columns.isin(numerical)].columns


# In[57]:


numerical, categorical


# In[64]:


sns.pairplot(df)


# In[71]:


plt.figure(figsize = (10,7))
sns.heatmap(df.corr(), data = df, annot = True, cmap = 'RdBu_r')


# # Observations :

# 1. Price for any Diamond depends on the Following Factors - Carat , Length , Width and Depth.
# 2. Also, Length , Depth and Width for any Diamond are very much related to one another.

# In[18]:


df.loc[(df['length'] == 0) | (df['width'] == 0) | (df['depth'] == 0) | (df['L/W'] == 0)]


# In[19]:


sns.boxplot( y='price' , data=df , x='color' , palette='Set1' , width=0.3 , order=['D','E','F','G','H','I','J'])


# From this above plot, we can see that G , H , I and J Type colors have less number of Outliers in comparison to D , E and F

# In[20]:


sns.boxplot( y='price' , data=df , x='cut' , palette='Set1' , width=0.5 , order=['Ideal','Premium','Very Good','Good','Fair'])


# From the Above Plot , we can see that Lower the Quality of Cut, Higher the number of Outliers, except for the Ideal Cut Type.

# In[21]:


sns.boxplot( y='price' , data=df , x='clarity' , palette='Set1' , width=0.7 , order=['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])


# Frome the Above Plot , we can see that IF,VVS1 and VVS2 have a higher number of Outliers compared to the other Categories. Moreover, VS1 and VS2 are having less number of Outliers compared to others.

# In[22]:


clarity_cut_table = pd.crosstab(index=df['clarity'],columns=df['cut'])

clarity_cut_table.plot(kind='bar',figsize=(10,10),stacked=True)


# From the Above Plot, we can see that most of the people prefer to buy Diamonds of SI1 clarity followed by VS2, SI2 and VS1. In that, the Cut they prefer is Ideal , Premium , and Very Good's Diamond Cut Category. Moreover , we can infer that people aren't Taking the highest Clarity Diamonds, such as IF or VVS1 and others. And are ready to sacrifice on Clarity but are more focusing on the Cut of the Diamonds. 

# In[23]:


cut_clarity_table = pd.crosstab(index=df['cut'],columns=df['clarity'])

cut_clarity_table.plot(kind='bar',figsize=(10,10),stacked=True)


# We can see that People prefer Idea Cut over any other Cut Diamonds followed by Premium and Very Good. It suggests that People are focussing on Cut than Clarity.

# In[24]:


color_clarity_table = pd.crosstab(index=df['color'],columns=df['clarity'])

color_clarity_table.plot(kind='bar',figsize=(10,10),stacked=True)


# We can see that from above Plot, most of the People prefer G color followed by E,F and H. In the clarity, they mostly prefer SI1 or SI2 category.
# 
# Therefore, from above all the Plots, we can conclude that carat has high importance followed by cut , color and clarity in predicting the price of a Diamond.

# In[25]:


g = sns.pairplot(df , height = 3 , aspect = 1 , x_vars = ['carat','depth%','table%'] , y_vars = ['price'] , kind = 'reg')


# In[26]:


g = sns.pairplot(df , height = 3 , aspect = 1 , x_vars = ['length','width','depth','L/W'] , y_vars = ['price'] , kind = 'reg')


# We could see that carat , length , width and depth are showing linearity with price with fewer outliers and table% , depth% , and L/W are showing linearity but with high outliers.

# In[27]:


df.head()


# In[28]:


def CutEncoding(cut):
    value =-1
    if (cut=='Ideal'):
        value=4
    elif (cut=='Premium'):
        value=3
    elif (cut=='Good'):
        value=2
    elif (cut=='Very Good'):
        value=1
    elif (cut=='Fair'):
        value=0
    return value

print("Cut Feature before Encoding:",*df['cut'].unique())
df['cut'] = df['cut'].apply(lambda x:CutEncoding(x))
print("Cut Feature after Encoding:",*df['cut'].unique())


# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


ColorEncoder = LabelEncoder()
print("Color Feature before Encoding:",*df['color'].unique())
df.color = ColorEncoder.fit_transform(df.color)
print("Color Feature after Encoding:",*df['color'].unique())


# In[31]:


ClarityEncoder = LabelEncoder()
print("Clarity Feature before Encoding:",*df['clarity'].unique())
df.clarity = ClarityEncoder.fit_transform(df.clarity)
print("Clarity Feature after Encoding:",*df['clarity'].unique())


# In[32]:


X = df.drop('price', axis = 1)
y = df.price


# In[33]:


X.head()


# In[34]:


y.head()


# In[35]:


df.head(10)


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[37]:


def minkowski_distance(a,b,p=1):
    dim = len(a)
    distance = 0
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
    distance = distance**(1/p)
    return distance

# TEST THE FUNCTION :

minkowski_distance(a=X.iloc[0],b=X.iloc[1],p=1)


# In[38]:


from sklearn.neighbors import KNeighborsRegressor
score = []

for k in range(1,20):     # Running for Different K Values to know which yields the max accuracy 
    clf = KNeighborsRegressor(n_neighbors = k , weights = 'distance' , p = 1)
    clf.fit(X_train, y_train)
    score.append(clf.score(X_test,y_test))


# In[39]:


k_max = score.index(max(score)) + 1
print("At k = {}, Max Accuracy = {}".format(k_max,max(score)*100))


# In[40]:


clf = KNeighborsRegressor(n_neighbors = k , weights = 'distance' , p = 1)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
y_pred = clf.predict(X_test)
print(y_pred[0])


# In[41]:


y_train.head()


# In[42]:


def knn_predict(X_train,X_test,y_train,y_test,k,p):
    
    #Counter to help with Label Voting
    from collections import Counter
    
    #Make Predictions on the Test Data
    #Need output of 1 Prediction per test data point
    y_hat_test = []
    
    for test_point in X_test[:3]:
        distances = []
        print('first')
        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        print('second')
        
        #Store Distances in a Dataframe
        df_dists = pd.DataFrame(data=distances, columns = ['dist'], index = y_train.index)
        print('third')
        print(df_dists.shape)
        
        #Sort Distances , and only consider the k closest Points
        df_nn = df_dists.sort_values(by = ['dist'],axis=0)[:k]
        #Create Counter Object to Track the Labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])
        
        df_nn['y_train_pred'] = counter
        print('fourth')
        
        #Get most Common Label of all the Nearest Neighbors
        prediction = df_nn['y_train_pred'].mean()
        
        #Append Prediction to Output List
        y_hat_test.append(prediction)
        
    return y_hat_test


# In[43]:


y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k = 7, p = 1)      # WITHOUT SKLEARN
print(y_hat_test)


# In[44]:


clf = KNeighborsRegressor(n_neighbors = 7 , weights = 'distance' , p = 1)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
y_pred = clf.predict(X_test)
print(y_pred[0])
print(y_pred[1])
print(y_pred[2])


# In[45]:


class KNearestNeighbors():
    def __init__(self,k):
        self.k = 1
        
    def train(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        distances = self.compute_distance(X_test)
        return self.predict_labels(distances)
    
    def compute_distance(self ,X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test,num_train))
    
        for i in range(num_test):
            for j in range(num_train):
                distances[i,j] = np.sqrt(np.sum((X_test[i,:] - self.X_train[j,:])**2))
        return distances
    
    def predict_labels(self,distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            y_indices = np.argsort(distances[i,:])
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))
            
        return y_pred


# In[46]:


def main():
    
    df = pd.read_csv('diamonds.csv',nrows = 1000)
    
    label = LabelEncoder()
    
    df['cut'] = label.fit_transform(df['cut'].astype('str'))
    df['color'] = label.fit_transform(df['color'].astype('str'))
    df['clarity'] = label.fit_transform(df['clarity'].astype('str'))
    
    # Features
    X = df.drop('price',axis=1).values
    # Target Feature
    y = df['price'].values
    
    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = None)
    
    # Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # Fitting the Model
    KNN = KNearestNeighbors( k = 3 )
    KNN.train( X_train , y_train )
    
    # Prediction
    y_pred = KNN.predict(X_test)
    
    # Check Accuracy of Model
    print('Accuracy of the Stratch Model :',r2_score( y_test , y_pred ))
    
    # Fitting the Model by sklearn
    knn = KNeighborsRegressor( n_neighbors = 3 )
    knn.fit( X_train , y_train )
    y_pred_sklearn = knn.predict( X_test )
    
    # Check Accuracy of the sklearn Model
    print('Accuracy of the sklearn Model :',r2_score( y_test , y_pred_sklearn ))
    
    
if __name__ == '__main__' :
    main()


# # Conclusions : From Scratch and sklearn models, we get almost the same outputs for y_pred and y_hat_test

# In[51]:


import plotly.graph_objects as go


# In[52]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
import os


# In[54]:


trace0 = go.Scatter(
    y = y_test,
    x = np. arange(200),
    mode = 'lines',
    name = 'Actual Price',
    marker = dict(
    color = 'rgb(10,150,50)')
)

trace1 = go.Scatter(
    y = y_pred,
    x = np.arange(200),
    mode = 'lines',
    name = 'Predicted Price',
    line = dict(
        color = 'rgb(110 , 50 , 140)',
        dash = 'dot'
    )
)

layout = go.Layout(
    xaxis = dict(title = 'Index'),
    yaxis = dict(title = 'Normalized Price')
)

figure = go.Figure(data = [ trace0 , trace1 ], layout = layout)
iplot(figure)


# In[ ]:




