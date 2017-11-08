# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    c = []
    for i in movies.get('genres'):
        c.append(tokenize_string(i))
    movies['tokens'] = np.asarray(c) 
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    N = len(movies)
    vocab_list = []
    for mov in movies['tokens'].tolist():
        for m in mov:
            if m not in vocab_list:
                vocab_list.append(m)
    vocab_list.sort()
    vocab = defaultdict(lambda:0)
    for i in range(len(vocab_list)):
        vocab[vocab_list[i]] = i  
        
    df_frequency = {}
    max_freq = 0
    tf_frequency = {}
    for index, m in movies.iterrows():
        freq = defaultdict(lambda: 0)
        for f in m['tokens']:
            if f not in df_frequency:
                df_frequency[f] = 1
            else:
                df_frequency[f] = df_frequency[f] + 1
            if f not in freq:
                freq[f] = 1
            else:
                freq[f] = freq[f] + 1
            if freq[f] >max_freq:
                max_freq = freq[f]
        tf_frequency[m['movieId']] = (freq, max_freq)    
    matrix = []    
    for index, m in movies.iterrows():
        data = []
        col = []
        row = []
        for f in m['tokens']:       
            data.append(tf_frequency[m['movieId']][0][f]/ (tf_frequency[m['movieId']][1] * math.log10(N/df_frequency[f])))
            col.append(vocab[f])
            row.append(0)
        matrix.append(csr_matrix((data,(row,col)), shape=(1,len(vocab))))
    movies['features'] = matrix
    return movies,vocab
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    return np.dot(a,b.T)/((np.linalg.norm(a))* (np.linalg.norm(b)))  
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    train_rate = defaultdict(list)
    for test_Index, test_m in ratings_train.iterrows():
        train_rate[test_m.userId].append((test_m.movieId,test_m.rating))    

    movies_feature = defaultdict()
    for Index, m in movies.iterrows():
        movies_feature[m.movieId] = m.features

    predcited_rating = []
    for test_Index, test_m in ratings_test.iterrows():
        weighted_prod = []
        weighted_sum = []
        rating = []
        test = train_rate[test_m.userId] 
        for t in test:
            c = cosine_sim(movies_feature[t[0]].toarray(),movies_feature[test_m.movieId].toarray())
            weighted_prod.append(c*t[1])
            weighted_sum.append(c)
            rating.append(t[1])
        if np.sum(weighted_sum)>0 and np.sum(weighted_prod)>0:
            weighted_avg = np.sum(weighted_prod)/np.sum(weighted_sum)
            predcited_rating.append(weighted_avg)
        else:
            predcited_rating.append(np.mean(rating))
    return np.array(predcited_rating)
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
