# -*- coding: utf-8 -*-
"""Quora_Project_PART1_EDA(MlOPS_TEAM_7).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FPQtlNF17xj33VCPNhnuqr2rx-FFFdJo

# **Problem Statement:**
Identify which questions asked on Quora are duplicates of questions that have already been asked.


*   This can be helpful for providing instant responses to questions that have already been addressed.

# **Objective:**
The goal of this project is to predict which of the provided pairs of questions contain two questions with the same meaning. 
# Real World/Business Objectives and Constraints:






*   The cost of a mis-classification can be very high.
*   You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
*   No strict latency concerns.
*   Interpretability is partially important.

# **About Dataset**

The dataset "Train.csv" contains **404,290 rows**and **5 columns**

#Data fields
**id** - the id of a training set question pair

**qid1, qid2** - unique ids of each question (only available in train.csv)

**question1, question2** - the full text of each question

**is_duplicate** - is the dependent variable, and the target variable is set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

"""> Load the Dataset"""

df = pd.read_csv('train.csv')

"""> Check the Shape of dataset"""

df.shape

df.head()

"""> Check the meta information about the dataset"""

df.info()

print("Available Columns : ", df.columns)
print("\nis_duplicates Class labels",df.is_duplicate.unique())
print("\nNo. of non-duplicate data points(0) and No. of duplicate data points(1) are :\n")
df.is_duplicate.value_counts()

"""# Distribution of data points among output classes

Number of duplicate(similar) and non-duplicate(non similar) questions
"""

sns.countplot(df["is_duplicate"])
plt.show()

print('Percentage of dissimilar pair of questions (is_duplicate = 0): {}%'.format(100 - round(df['is_duplicate'].mean()*100, 2)))
print('\nPercentage of similar pair of questions (is_duplicate = 1): {}%'.format(round(df['is_duplicate'].mean()*100, 2)))

"""> **63.08%** of questions pair are not duplicates and **36.92%** of question pairs are duplicates.

> We have 404290 training data points. And only 36.92% are positive. That means it is an imbalanced dataset.
"""

qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
unique_qids = len(np.unique(qids))
repeated_qids = np.sum(qids.value_counts() > 1)
print("Total no. unique qids: ", unique_qids)
print("\nToatal no. of repeated qids: ", repeated_qids)

x = ["unique_qids" , "repeated_qids"]
y =  [unique_qids , repeated_qids]

plt.figure(figsize=(10, 6))
plt.title ("Unique vs Repeated qids")
sns.barplot(x,y)
plt.show()

"""> Let's check whether there are any repeated pair of questions."""

pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()

print ("Number of duplicate pair of questions:",(pair_duplicates).shape[0] - df.shape[0])

"""> Let us visually plot the no. of times a question is repeated."""

occurences = np.bincount(qids)
plt.figure(figsize=(10,5)) 
plt.hist(occurences, bins=range(0,np.max(occurences)))
plt.yscale('log')
plt.xlabel('Number of times question repeated')
plt.ylabel('Number of questions')
plt.title('Question vs Repeatition')
plt.show()
print(np.min(occurences), np.max(occurences))

"""*      When we include both question1 and question2 then count of total number of questions are 808574.
*      Out of these 808574 questions 537929 are unique questions and rest are repeated questions.
*      Most of the questions are repeated very few times. Only a few of them are repeated multiple times.
*      And we can notice that there is One question which is the most repeated one and it is repeated 157 times.
*      There are some questions with very few characters, which does not make sense. It will be taken care of later with Data Cleaning.

> Check whether there are any rows with null values.
"""

df.isnull().sum()

"""> As we notice they are total 3 null values 1 in question1 and 2 in question2,  let us print those null rows"""

null_rows = df[df.isnull().any(1)]
print (null_rows)

"""> Fill the null rows with ' '"""

df = df.fillna('')

"""> Check wheather null rows are filled."""

null_rows = df[df.isnull().any(1)]
print (null_rows)

df.isnull().sum()

"""> As we notice know the null rows are filled i.e, our dataset doesn't contain any more null rows"""

df.duplicated().sum()

"""> There are no duplicates in our dataset

# Sampling 3000 data points to enhance training speed and optimize memory usage.
"""

new_df = df.sample(3000,random_state=2)

"""# **Basic EDA**"""

new_df.shape

new_df.head()

"""> Check the meta information about the dataset"""

new_df.info()

print("\nis_duplicates Class labels",new_df.is_duplicate.unique())
print("\nNo. of non-duplicate data points(0) and No. of duplicate data points(1) are :\n")
new_df.is_duplicate.value_counts()

print('Percentage of dissimilar pair of questions (is_duplicate = 0): {}%'.format(100 - round(new_df['is_duplicate'].mean()*100, 2)))
print('\nPercentage of similar pair of questions (is_duplicate = 1): {}%'.format(round(new_df['is_duplicate'].mean()*100, 2)))

"""> **62.73%** of questions pair are not duplicates and **37.27%** of question pairs are duplicates in sample dataframe(new_df).

> That means it is an imbalanced dataset.

# Distribution of data points among output classes

Number of duplicate(smilar) and non-duplicate(non similar) questions
"""

sns.countplot(new_df["is_duplicate"])
plt.show()

qids = pd.Series(new_df['qid1'].tolist() + new_df['qid2'].tolist())
unique_qids = len(np.unique(qids))
repeated_qids = np.sum(qids.value_counts() > 1)
print("Total no. unique qids: ", unique_qids)
print("\nToatal no. of repeated qids: ", repeated_qids)

x = ["unique_qids" , "repeated_qids"]
y =  [unique_qids , repeated_qids]

plt.figure(figsize=(10, 6))
plt.title ("Unique vs Repeated qids")
sns.barplot(x,y)
plt.show()

"""> Let's check whether there are any repeated pair of questions."""

pair_duplicates = new_df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()

print ("Number of duplicate pair of questions:",(pair_duplicates).shape[0] - new_df.shape[0])

"""# **Data Preprocessing**
Perform below actions on question1 and question2 columns of the dataset:
*   Convert entire text to lowercase
*   Remove html tags
*   Remove Stopwords
*   Expand Contractions
*   Remove Punctuations
*   Remove Special Characters
*   Remove hyperlinks
*   Apply Tokenization
*   Apply Stemming
*   Apply Lemmatization

"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
# Downloading wordnet before applying Lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

## initialise the inbuilt Stemmer
stemmer = PorterStemmer()

## We can also use Lemmatizer instead of Stemmer
lemmatizer = WordNetLemmatizer()

def preprocess(q, flag):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()
    
    # Stemming/Lemmatization
    if(flag == 'stem'):
        q = stemmer.stem(q)
    else:
        q = lemmatizer.lemmatize(q)
        
    return q
    # tokenize into words
    #tokens = q.split()
    
    # remove stop words                
    #clean_tokens = [t for t in tokens if not t in stopwords.words("english")]
    

    # Stemming/Lemmatization
    #if(flag == 'stem'):
        #clean_tokens = [stemmer.stem(word) for word in clean_tokens]
    #else:
        #clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]

    #q=' '.join([" ".join(clean_tokens)])
    #return q

q1 = 'Where is the capital of India?'
preprocess(q1,'lem')

!pip install tqdm

from tqdm import tqdm, tqdm_notebook

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`

tqdm.pandas()

"""> Add new columns "**clean_q1_stem**","**clean_q2_stem**" to the dataframe new_df.
> where, 
- **clean_q1_stem** column represents preprocessed_q1 column after applying preprocess function with flag=Stem 
- **clean_q2_stem** column represents preprocessed_q2 column after applying preprocess function with flag=Stem

> Add new columns "**clean_q1_lem**","**clean_q2_lem**" to the dataframe new_df.
> where, 
- **clean_q1_lem** column represents preprocessed_q1 column after applying preprocess function with flag=lem
- **clean_q2_lem** column represents preprocessed_q2 column after applying preprocess function with flag=lem
"""

#temp_df = X_train['question1'].progress_apply(lambda x: preprocess(x, 'stem'))

new_df['Clean_q1_stem'] = new_df['question1'].progress_apply(lambda x: preprocess(x,'Stem'))
new_df['Clean_q2_stem'] = new_df['question2'].progress_apply(lambda x: preprocess(x,'Stem'))

new_df['Clean_q1_lem'] = new_df['question1'].progress_apply(lambda x: preprocess(x,'lem'))
new_df['Clean_q2_lem'] = new_df['question2'].progress_apply(lambda x: preprocess(x,'lem'))
new_df.head()

"""# Word Clouds"""

from random import randint
from wordcloud import WordCloud, STOPWORDS

# To customize colours of wordcloud texts
def wc_blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(214, 67%%, %d%%)" % randint(60, 100)


# stopwords for wordcloud
def get_wc_stopwords():
    wc_stopwords = set(STOPWORDS)
    return wc_stopwords


# plot wordcloud
def plot_wordcloud(text, color_func):
    wc_stopwords = get_wc_stopwords()
    wc = WordCloud(stopwords=wc_stopwords, width=1200, height=600, random_state=0).generate(text)

    f, axs = plt.subplots(figsize=(10, 10))
    with sns.axes_style("ticks"):
        sns.despine(offset=10, trim=True)
        plt.imshow(wc.recolor(color_func=color_func, random_state=0), interpolation="bilinear")
        plt.xlabel('WordCloud')

"""# Word Cloud for Preprocessed question1"""

print("Word Cloud for preprocesssed question1:")
plot_wordcloud(' '.join(new_df.Clean_q1_lem	.values.tolist()), wc_blue_color_func)

"""# Word Cloud for Preprocessed question2"""

print("Word Cloud for preprocessed question2:")
plot_wordcloud(' '.join(new_df.Clean_q2_lem	.values.tolist()), wc_blue_color_func)

"""# Work Cloud for Duplicate Question Pairs"""

print ("Word Cloud for Duplicate Question pairs")
plot_wordcloud(' '.join(new_df['Clean_q1_lem'].loc[new_df['is_duplicate'] == 1].values.tolist() and new_df['Clean_q2_lem'].loc[new_df['is_duplicate'] == 1].values.tolist()), wc_blue_color_func)

"""# Work Cloud for Non-Duplicate Question Pairs"""

print ("Word Cloud for Non-Duplicate Question pairs")
plot_wordcloud(' '.join(new_df['Clean_q1_lem'].loc[new_df['is_duplicate'] == 0].values.tolist() and new_df['Clean_q2_lem'].loc[new_df['is_duplicate'] == 0].values.tolist()), wc_blue_color_func)

"""# Plot count of length of words in Clean_q1_lem and Clean_q2_lem"""

Question1_count = [len(sentence.split()) for sentence in new_df.Clean_q1_lem]
Question2_count = [len(sentence.split()) for sentence in new_df.Clean_q2_lem]

pd.DataFrame({'Question1':Question1_count, 'Question2': Question2_count}).hist(bins=100, figsize=(16, 6), range=[0, 30])
plt.show()

# To check how many rows in a column has length (of the text) <= limit
def get_word_percent(column, limit):
    count = 0
    for sentence in column:
        if len(sentence.split()) <= limit:
            count += 1

    return round(count / len(column), 2)


print("Percentage of Question1 having 0-30 words: ",get_word_percent(new_df.Clean_q1_lem,30 ))
print("\nPercentage of Question1 having 0-15 words: ",get_word_percent(new_df.Clean_q1_lem,15 ))

print("\nPercentage of Question2 having 0-30 words: ",get_word_percent(new_df.Clean_q2_lem, 30))
print("\nPercentage of Question2 having 0-15 words: ",get_word_percent(new_df.Clean_q2_lem, 15))

"""> We can notice that almost 99% of words are in range 0 to 15 for both Question1 and Question2.

> And they are very few questions in both Question1 and Question2 that have count of words greater than 15.

# **Feature Extraction**
we have divided this step in three parts
1. **Basic Features** : These features are extracted from preprocessed question1 and question2 and are stored in new dataframe named "bf_df"(where bf represents basic features)
2. **StopWords and Token Features** : These features are extracted based on stopwords from original question1 and question2 from new_df and are stored in new dataframe named "SW_df"(where sw represents stop words)
3. **Fuzzy Features** : These features are extracted from fuzzywords of preprocessed question1 and question2 and are stored in new dataframe named "fuzzy_df"

# 1. Basic Features :
Let us construct few basic features which are described below:

*     **freq_qid1** = Frequency of qid1's
*     **freq_qid2** = Frequency of qid2's
*     **f(q1+q2)** = sum total of frequency of qid1 and qid2
*     **f(q1-q2)** = absolute difference of frequency of qid1 and qid2
*     **Q1_char** = count of characters of Question1
*     **Q2_char**  = count of characters of Question2
*     **q1_num_words** = Count of words in Question 1
*     **q2_num_words** = Count of words in Question 2
*     **total_word_num** = Sum of Q1_n_words and Q2_n_words
*     **diff_word_num** = absolute difference of Q1_n_words and Q2_n_words
*     **same_first_word** = This feature is 1 if both questions have same first word otherwise 0.
*     **same_last_word**  = This feature is 1 if both questions have same last word otherwise 0.
*    **unique_common_words** = Count of unique words which are common for both Question 1 and Question 2
*   **same_common_words** =  Count of words which are same and common for both Question 1 and Question 2
*   **total_unique_word_share** = (unique_common_words)/(total_word_num)
*   **total_same_word_share** = (same_common_words)/(unique_common_words)
*   **min_common_word_ratio** = Same_common_words divided by minimum number of words between question 1 and question 2.
*   **max_common_word_ratio** = Same_common_words divided by maximum number of words between question 1 and question 2.
"""

#Basic_feature_df ->bf_df
bf_df=pd.concat([new_df['id'],new_df['qid1'],new_df['qid2'],new_df['Clean_q1_lem'],new_df['Clean_q2_lem'],new_df['is_duplicate']],axis=1)
bf_df1=pd.concat([new_df['id'],new_df['qid1'],new_df['qid2'],new_df['Clean_q1_lem'],new_df['Clean_q2_lem'],new_df['is_duplicate']],axis=1)

bf_df.head()

def doesMatch (q, match):
    q1, q2 = q['Clean_q1_lem'], q['Clean_q2_lem']
    q1 = q1.split()
    q2 = q2.split()
    if len(q1)>0 and len(q2)>0 and q1[match]==q2[match]:
        return 1
    else:
        return 0

bf_df['freq_qid1'] = bf_df.groupby('qid1')['qid1'].transform('count') 
bf_df['freq_qid2'] = bf_df.groupby('qid2')['qid2'].transform('count')
bf_df['f(q1+q2)'] = bf_df['freq_qid1']+bf_df['freq_qid2']
bf_df['f(q1-q2)'] = abs(bf_df['freq_qid1']- bf_df['freq_qid2'])

bf_df['Q1_char'] = bf_df.Clean_q1_lem.apply(len)
bf_df['Q2_char'] = bf_df.Clean_q2_lem.apply(len)

bf_df['Q1_n_words'] = bf_df['Clean_q1_lem'].apply(lambda row: len(row.split(" ")))
bf_df['Q2_n_words'] = bf_df['Clean_q2_lem'].apply(lambda row: len(row.split(" ")))
bf_df['total_word_num'] = bf_df['Q1_n_words']+bf_df['Q1_n_words']
bf_df['diff_word_num'] = abs(bf_df['Q1_n_words']-bf_df['Q1_n_words'])
bf_df['same_first_word'] = bf_df.apply(lambda x: doesMatch(x, 0) ,axis=1)
bf_df['same_last_word'] = bf_df.apply(lambda x: doesMatch(x, -1) ,axis=1)
bf_df['unique_common_words'] = bf_df.apply(lambda x: len(set(x.Clean_q1_lem.split()).union(set(x.Clean_q2_lem.split()))) ,axis=1)
bf_df['same_common_words'] = bf_df.apply(lambda x: len(set(x.Clean_q1_lem.split()).intersection(set(x.Clean_q2_lem.split()))) ,axis=1)
bf_df['total_unique_word_share'] = bf_df['unique_common_words']/bf_df['total_word_num']
bf_df['total_same_word_share'] = bf_df['same_common_words']/bf_df['unique_common_words']
bf_df['min_common_word_ratio'] = bf_df['same_common_words'] / bf_df.apply(lambda x: min(len(set(x.Clean_q1_lem.split())), len(set(x.Clean_q2_lem.split()))) ,axis=1) 
bf_df['max_common_word_ratio_max'] = bf_df['same_common_words'] / bf_df.apply(lambda x: max(len(set(x.Clean_q1_lem.split())), len(set(x.Clean_q2_lem.split()))) ,axis=1)

bf_df1['q1_len'] = bf_df1['Clean_q1_lem'].str.len() 
bf_df1['q2_len'] = bf_df1['Clean_q2_lem'].str.len()

bf_df1['q1_num_words'] = bf_df1['Clean_q1_lem'].apply(lambda row: len(row.split(" ")))
bf_df1['q2_num_words'] = bf_df1['Clean_q2_lem'].apply(lambda row: len(row.split(" ")))

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['Clean_q1_lem'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['Clean_q2_lem'].split(" ")))    
    return len(w1 & w2)

bf_df1['word_common'] = bf_df1.apply(common_words, axis=1)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['Clean_q1_lem'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['Clean_q2_lem'].split(" ")))    
    return (len(w1) + len(w2))

bf_df1['word_total'] = bf_df1.apply(total_words, axis=1)

bf_df1['word_share'] = round(bf_df1['word_common']/bf_df1['word_total'],2)
bf_df1.head()

"""> # Visualization of basic features"""

fig, ax =plt.subplots(1,2,figsize=(15,5))
ax[0].title.set_text('PDF of Total Number of Words')
sns.kdeplot(bf_df['Q1_n_words'], hue=bf_df.is_duplicate, ax=ax[0])
ax[1].title.set_text('PDF of different Number of Words')
sns.kdeplot(bf_df['Q1_n_words'], hue=bf_df.is_duplicate, ax=ax[1])
plt.show()

fig, ax =plt.subplots(1,2,figsize=(15,5))
ax[0].title.set_text('PDF of same first Words')
sns.kdeplot(bf_df['same_first_word'], hue=bf_df.is_duplicate, ax=ax[0])
ax[1].title.set_text('PDF of same last Words')
sns.kdeplot(bf_df['same_last_word'], hue=bf_df.is_duplicate, ax=ax[1])
plt.show()

fig, ax =plt.subplots(1,2,figsize=(15,5))
ax[0].title.set_text('PDF of unique common Words')
sns.kdeplot(bf_df['unique_common_words'], hue=bf_df.is_duplicate, ax=ax[0])
ax[1].title.set_text('PDF of same common Words')
sns.kdeplot(bf_df['same_common_words'], hue=bf_df.is_duplicate, ax=ax[1])
plt.show()

fig, ax =plt.subplots(1,2,figsize=(15,5))
ax[0].title.set_text('Boxplot of Ratio of number of Common words to Minimum of Unique words')
sns.boxplot(x=bf_df.is_duplicate, y=bf_df['min_common_word_ratio'], ax=ax[0])
ax[1].title.set_text('Boxplot of Ratio of number of Common words to maximum of Unique words')
sns.boxplot(x=bf_df.is_duplicate, y=bf_df['max_common_word_ratio_max'],ax=ax[1])
plt.show()

"""# 2. StopWords and Tokens Features : 
*   **cwc_min** : Ratio of common_word_count to min length of word count of Q1 and Q2
*   **cwc_max** : Ratio of common_word_count to max length of word count of Q1 and Q2
*   **csc_min** : Ratio of common_stop_count to min length of stop count of Q1 and Q2
*   **csc_max** : Ratio of common_stop_count to max length of stop count of Q1 and Q2
*   **ctc_min** : Ratio of common_token_count to min length of token count of Q1 and Q2
*   **ctc_max** : Ratio of common_token_count to max length of token count of Q1 and Q2
*   **last_word_eq** : Check if Last word of both questions is equal or not
*   **first_word_eq** : Check if First word of both questions is equal or not
*   **abs_len_diff** : Abs. length difference
*   **mean_len** : Average Token Length of both Questions
*   **longest_substr_ratio** : Ratio of length longest common substring to min length of token count of Q1 and Q2
"""

SW_df=pd.concat([new_df['question1'],new_df['question2'],new_df['is_duplicate']],axis=1)

SW_df.head()

!pip install distance

import distance
def fetch_token_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*11
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    # Absolute length features
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    token_features[10] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    
    return token_features

token_features = new_df.apply(fetch_token_features, axis=1)

SW_df["cwc_min"]       = list(map(lambda x: x[0], token_features))
SW_df["cwc_max"]       = list(map(lambda x: x[1], token_features))
SW_df["csc_min"]       = list(map(lambda x: x[2], token_features))
SW_df["csc_max"]       = list(map(lambda x: x[3], token_features))
SW_df["ctc_min"]       = list(map(lambda x: x[4], token_features))
SW_df["ctc_max"]       = list(map(lambda x: x[5], token_features))
SW_df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
SW_df["first_word_eq"] = list(map(lambda x: x[7], token_features))
SW_df['abs_len_diff']  = list(map(lambda x: x[8], token_features))
SW_df['mean_len']      = list(map(lambda x: x[9], token_features))
SW_df['longest_substr_ratio'] = list(map(lambda x: x[10], token_features))

SW_df.head()

"""> # Visualization of stopwords and tokens features"""

fig, ax =plt.subplots(1,2,figsize=(15,5))
ax[0].title.set_text('PDF of  Ratio of common_word_count to min length')
sns.kdeplot(SW_df['cwc_min'], hue=SW_df.is_duplicate, palette="Dark2", ax=ax[0])
ax[1].title.set_text('PDF of  Ratio of common_word_count to max length')
sns.kdeplot(SW_df['cwc_max'], hue=SW_df.is_duplicate, palette="Dark2", ax=ax[1])
plt.show()

sns.pairplot(SW_df[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')

sns.pairplot(SW_df[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')

sns.pairplot(SW_df[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')

"""#Observation: 
We can notice that the distribution of Stopword and tokens features are Overlapping for duplicate & Non-duplicate Question Pairs. 

So these features may not be useful in separating the duplicate & non-duplicate question pairs. Hence, we would not consider them for modelling.

> # 3. Fuzzy Features:
*      **fuzz_ratio** : fuzzyWuzzy has a ratio function that calculates the standard Levenshtein distance similarity ratio between two sequences.
*      **fuzz_partial_ratio** : The partial ratio helps us to perform substring matching. This takes the shortest string and compares it with all the substrings of the same length.
*      **token_set_ratio** : Token set ratio performs a set operation that takes out the common tokens instead of just tokenizing the strings, sorting, and then pasting the tokens back together. Extra or same repeated words do not matter.
*      **token_sort_ratio** : In token sort ratio, the strings are tokenized and pre-processed by converting to lower case and getting rid of punctuation. The strings are then sorted alphabetically and joined together. Post this, the Levenshtein distance similarity ratio is calculated between the strings.
"""

!pip install fuzzywuzzy

fuzzy_df=pd.concat([new_df['question1'],new_df['question2'],new_df['is_duplicate']],axis=1)

# Fuzzy Features
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

fuzzy_features = fuzzy_df.apply(fetch_fuzzy_features, axis=1)

# Creating new feature columns for fuzzy features
fuzzy_df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
fuzzy_df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
fuzzy_df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
fuzzy_df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))

fuzzy_df.head()

"""> Let's visualize above features."""

sns.pairplot(fuzzy_df,hue='is_duplicate')

fig, ax =plt.subplots(1,2,figsize=(15,5))
ax[0].title.set_text('PDF of Fuzzy Ratio')
sns.kdeplot(fuzzy_df['fuzz_ratio'], hue=fuzzy_df.is_duplicate, palette="Dark2", ax=ax[0])
ax[1].title.set_text('PDF of Fuzzy Partial Ratio')
sns.kdeplot(fuzzy_df['fuzz_partial_ratio'], hue=fuzzy_df.is_duplicate, palette="Dark2", ax=ax[1])
plt.show()

fig, ax =plt.subplots(1,2,figsize=(15,5))
ax[0].title.set_text('PDF of Token Sort Ratio')
sns.kdeplot(fuzzy_df['token_sort_ratio'], hue=fuzzy_df.is_duplicate, palette="Dark2", ax=ax[0])
ax[1].title.set_text('PDF of Token Set Ratio')
sns.kdeplot(fuzzy_df['token_set_ratio'], hue=fuzzy_df.is_duplicate, palette="Dark2", ax=ax[1])
plt.show()

feature_df=pd.concat([ new_df['id'],new_df[ 'qid1'], new_df['qid2'],new_df['Clean_q1_lem'],new_df[ 'Clean_q2_lem'],new_df[ 'is_duplicate'], bf_df1['q1_len'],bf_df1['q2_len'],bf_df1['q1_num_words'], bf_df1['q2_num_words'], 
       bf_df1['word_common'], bf_df1['word_total'], bf_df1['word_share'],SW_df['cwc_min'], SW_df['cwc_max'], SW_df['csc_min'], SW_df['csc_max'],
       SW_df['ctc_min'], SW_df['ctc_max'], SW_df['last_word_eq'], SW_df['first_word_eq'], SW_df['abs_len_diff'],SW_df['mean_len'], SW_df['longest_substr_ratio'],fuzzy_df['fuzz_ratio'], fuzzy_df['fuzz_partial_ratio'], fuzzy_df['token_sort_ratio'],fuzzy_df['token_set_ratio'] ],axis=1)
feature_df.columns

from google.colab import files

feature_df.to_csv('feature_df.csv')
files.download('feature_df.csv')

from google.colab import files

feature_df.to_csv('feature_df.csv')
files.download('feature_df.csv')