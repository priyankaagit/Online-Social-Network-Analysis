from io import BytesIO, StringIO
from zipfile import ZipFile
import urllib.request
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from itertools import chain, combinations
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
# from sklearn.linear_model import LogisticRegression
import string
import tarfile

def download_data():
    url = urllib.request.urlopen('http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    # We'll focus on the smaller file that was manually labeled.
    # The larger file has 1.6M tweets "pseudo-labeled" using emoticons
    tweet_file = zipfile.open('testdata.manual.2009.06.14.csv')
    return tweet_file
    pass

def read_data(tweet_file):
    tweets = pd.read_csv(tweet_file,
                         header=None,
                         names=['polarity', 'id', 'date',
                                'query', 'user', 'text'])
    y = np.array(tweets['polarity'])
    return tweets['text'], y

def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    ###TODO
#     doc = re.sub(r'[0-9]+', ' ', doc)
    if keep_internal_punct:
        str_punt = ""
        for p in string.punctuation:
            if p != '_':
                str_punt = str_punt+p
            li = []
        for word in doc.lower().split():
            if(word.strip(str_punt)):
                li.append(word.strip(str_punt))
        return np.array(li)
    return np.array(re.sub('\W+', ' ', doc.lower()).split()) 
    
    pass

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    for x in tokens:
        key = "token=" + x
        if key not in feats:
            feats[key] = 1
        else:
            feats[key] = feats[key] + 1
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    for i in range(len(tokens)-(k-1)):
        for j in range(i,k+i):
            for l in range(j+1,k+i):
                x = "token_pair="+tokens[j]+"__"+tokens[l]
                if x not in feats:
                    feats[x] = 1
                else:
                    feats[x] = feats[x] + 1
    pass

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful','best','grand','happiest '])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats["pos_words"] = 0
    feats["neg_words"] = 0
    for tok in tokens:
        if tok.lower() in pos_words:
            if feats["pos_words"] == 0:
                    feats["pos_words"] = 1
            else:
                feats["pos_words"] = feats["pos_words"] + 1
        if tok.lower() in neg_words:
            if feats["neg_words"] == 0:
                feats["neg_words"] = 1
            else:
                feats["neg_words"] = feats["neg_words"] + 1
    pass

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats = defaultdict(lambda: 0)
    for function in feature_fns:
        function(tokens,feats)
    return sorted(feats.items(),key=lambda x:x[0])
    pass

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int32)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    data = []
    col = []
    row = [] 
    feat = {}

    for t in tokens_list:
        for f in featurize(t,feature_fns):
            if f[0] not in feat:
                feat[f[0]] = 1
            else:
                feat[f[0]] = feat[f[0]] + 1
                
    vocab_list = []
    if vocab is None:
        vocab = defaultdict(lambda:0)
        for key,value in feat.items():
                if  value >=min_freq:
                    vocab_list.append(key)
        vocab_list.sort()
        for i in range(len(vocab_list)):
            vocab[vocab_list[i]] = i
    
        
    for i in range(len(tokens_list)):
        sub_feat = featurize(tokens_list[i],feature_fns)
        for j in range(len(sub_feat)):
            if sub_feat[j][0] in vocab:               
                data.append(sub_feat[j][1])
                col.append(vocab[sub_feat[j][0]])
                row.append(i)
    new_data = np.array(data, dtype='int64')
    new_col = np.array(col,dtype='int64')
    new_row = np.array(row,dtype='int64')
    
    return csr_matrix((new_data, (new_row, new_col)), shape=(len(tokens_list), len(vocab))),vocab
    pass

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cv= KFold(len(labels),k)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind],labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind],predictions))        
    return np.mean(accuracies)
    
    pass

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    feature_combination = []
    feature_list = []
    for i in range(len(feature_fns)):
        for i in (list(combinations(feature_fns, i+1))):
            feature_combination.append(i)
    for p in punct_vals:
        tokens_list = [tokenize(d,p) for d in docs]
        for m in min_freqs:
            for f in feature_combination:
                X, vocab = vectorize(tokens_list,f,m)
                clf = LogisticRegression()
                n = cross_validation_accuracy(clf,X,labels,5)
                feature_list.append({'features':f,'punct':p,'min_freq':m,'accuracy':n})
    return sorted(feature_list, key=lambda x:x['accuracy'],reverse = True)   
    pass

def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    tokens_list = [tokenize(d,best_result['punct']) for d in docs]
    X, vocab = vectorize(tokens_list,best_result['features'],best_result['min_freq'])
    clf = LogisticRegression().fit(X,labels)
    
    return clf, vocab
    pass


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    test_tweets = pd.read_csv('testtweets.csv',
                     header=None,
                     names=['polarity', 'text'])
    test_docs = test_tweets['text']
    test_labels = np.array(test_tweets['polarity'])
    test_token_list = [tokenize(d,best_result['punct']) for d in test_docs]
    X_test, vocab = vectorize(test_token_list, best_result['features'],best_result['min_freq'],vocab)
    
    return test_docs, test_labels, X_test
    pass


def instance(predictions,test_docs):
    count_two = 0
    count_zero = 0
    count_four = 0
    for i in predictions:
        if i == 2:
            count_two = count_two+1
        if i == 0:
            count_zero = count_zero+1
        if i == 4:
            count_four = count_four+1

    file = open('classify.txt','w')
    file.write(str(count_four))
    file.write("\n")
    file.write(str(count_two))
    file.write("\n")
    file.write(str(count_zero))
    file.write("\n")
    file.write(test_docs[64])
    file.write("\n")
    file.write(test_docs[88])
    file.write("\n")
    file.write(test_docs[42])
    file.close()

def main():
    
    file = download_data()
    docs, labels = read_data(file)
    feature_fns = [token_features, token_pair_features, lexicon_features]
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    best_result = results[0]
    worst_result = results[-1]
    clf, vocab = fit_best_classifier(docs, labels, results[0])   
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)
    predictions = clf.predict(X_test)
    instance(predictions,test_docs)
    print('Classification complete, run the next script summarize.py')
    
if __name__ == '__main__':
    main()



