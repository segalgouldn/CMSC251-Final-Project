
# coding: utf-8

# # Introduction to Artificial Intelligence Final Project
# ## By Noah Segal-Gould and Tanner Cohan

# ### To implement:
# [K-Means clustering, hierarchical document clustering, and topic modeling](http://brandonrose.org/clustering)
# 
# [K-Means clustering](http://scikit-learn.org/dev/auto_examples/text/plot_document_clustering.html)

# #### Import libraries

# In[1]:


import pandas as pd
from glob import glob
from os.path import basename, splitext
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# #### Create lists of file names for all Twitter account CSVs

# In[2]:


house_accounts_filenames = glob("house/*.csv")


# In[3]:


senate_accounts_filenames = glob("senate/*.csv")


# #### Create lists of all dataframes for all CSVs

# In[4]:


house_accounts_dataframes = [pd.read_csv(filename).assign(account="@" + splitext(basename(filename))[0]) 
                             for filename in house_accounts_filenames]


# In[5]:


senate_accounts_dataframes = [pd.read_csv(filename).assign(account="@" + splitext(basename(filename))[0])
                              for filename in senate_accounts_filenames]


# #### Find which Tweets were most Retweeted and Favorited in each list of dataframes

# In[6]:


most_retweets_house_accounts_dataframes = [df.iloc[[df['Retweets'].idxmax()]] 
                                           for df in house_accounts_dataframes]


# In[7]:


most_favorites_house_accounts_dataframes = [df.iloc[[df['Favorites'].idxmax()]] 
                                            for df in house_accounts_dataframes]


# In[8]:


most_retweets_senate_accounts_dataframes = [df.iloc[[df['Retweets'].idxmax()]] 
                                            for df in senate_accounts_dataframes]


# In[9]:


most_favorites_senate_accounts_dataframes = [df.iloc[[df['Favorites'].idxmax()]] 
                                             for df in senate_accounts_dataframes]


# #### Create dataframes of the most Retweeted and Favorited Tweets for each account

# In[10]:


most_retweets_congress_dataframe = pd.concat(most_retweets_house_accounts_dataframes + most_retweets_senate_accounts_dataframes).reset_index(drop=True)


# In[11]:


most_favorites_congress_dataframe = pd.concat(most_favorites_house_accounts_dataframes + most_favorites_senate_accounts_dataframes).reset_index(drop=True)


# #### Show the Retweets dataframe

# In[12]:


most_retweets_congress_dataframe.sort_values('Retweets').tail()


# #### Show the Favorites dataframe

# In[13]:


most_favorites_congress_dataframe.sort_values('Favorites').tail()


# #### Combine all House of Representatives' accounts, all Senators' accounts, and then combine them together into all Congress accounts

# In[14]:


house_dataframe = pd.concat(house_accounts_dataframes)


# In[15]:


senate_dataframe = pd.concat(senate_accounts_dataframes)


# In[16]:


congress_dataframe = pd.concat([house_dataframe, senate_dataframe]).reset_index(drop=True)


# #### Remove columns with missing values

# In[17]:


congress_dataframe.dropna(inplace=True)


# #### Print some statistics

# In[18]:


print("Total number of Tweets for all accounts: " + str(len(congress_dataframe)))
print("Total number of accounts: " + str(len(set(congress_dataframe["account"]))))
print("Total number of house members: " + str(len(set(house_dataframe["account"]))))
print("Total number of senators: " + str(len(set(senate_dataframe["account"]))))


# #### Get NLTK English stopwords

# In[19]:


stopwords = stopwords.words('english')


# #### Instantiate SnowballStemmer as stemmer

# In[20]:


stemmer = SnowballStemmer("english")


# #### Load NLTK's Tweet Tokenizer

# In[21]:


tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)


# #### Define a function to return the list of stemmed words and the list of tokens which have been stripped of non-alphabetical characters and stopwords

# In[22]:


def tokenize_and_stem(text):
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [word for word in tokens if (word.isalpha() and word not in stopwords)]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [word for word in tokens if (word.isalpha() and word not in stopwords)]
    return filtered_tokens


# #### Define a function for getting lists of stemmed and tokenized Tweets

# In[23]:


def get_stemmed_and_tokenized_dict(tweets):
    stemmed = []
    tokenized = []
    for tweet in tweets:
        stemmed.extend(tokenize_and_stem(tweet))
        tokenized.extend(tokenize_only(tweet))
    return {"Stemmed": stemmed, "Tokenized": tokenized}


# #### Apply function to Tweets

# In[24]:


get_ipython().magic('time stemmed_and_tokenized_dict = get_stemmed_and_tokenized_dict(congress_dataframe["Text"])')


# #### Create a dataframe of stemmed and tokenized words

# In[25]:


vocab_frame = pd.DataFrame({'words': stemmed_and_tokenized_dict["Tokenized"]}, 
                           index = stemmed_and_tokenized_dict["Stemmed"])


# In[26]:


print("There are " + str(vocab_frame.shape[0]) + " items in vocab_frame")


# In[27]:


vocab_frame.head()


# In[28]:


vocab_frame.to_csv("vocab_frame.csv")


# #### Set up TF-IDF vectorizer from Scikit Learn and also apply the vectorizer to the Tweets

# In[29]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=2000,
                                   min_df=2, stop_words='english',
                                   use_idf=True)


# In[30]:


get_ipython().magic('time tfidf_matrix = tfidf_vectorizer.fit_transform(congress_dataframe["Text"])')


# In[31]:


print(tfidf_matrix.shape)


# In[ ]:


terms = tfidf_vectorizer.get_feature_names()


# In[ ]:


dist = 1 - cosine_similarity(tfidf_matrix)


# #### Set up K-Means clustering

# In[ ]:


#num_clusters = 4


# In[ ]:


#km = KMeans(n_clusters=num_clusters)


# In[ ]:


#%time km.fit(tfidf_matrix)


# In[ ]:


#clusters = km.labels_.tolist()


# #### Alternatively use K-Means++

# In[ ]:


num_clusters = 4


# In[ ]:


km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1, verbose=1)


# In[ ]:


get_ipython().magic('time km.fit(tfidf_matrix)')


# In[ ]:


clusters = km.labels_.tolist()


# #### Show the top 10 keywords for each cluster

# In[ ]:


print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d:" % i, end=" ")
    for ind in order_centroids[i, :10]:
        print(" %s" % terms[ind], end=" ")
    print()


# In[ ]:





# In[ ]:





# In[ ]:




