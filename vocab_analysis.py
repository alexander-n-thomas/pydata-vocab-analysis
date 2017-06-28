#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Alex Thomas"
__credits__ = ["Alex Thomas", "Annette Taberner-Miller"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alex Thomas"
__email__ = "althomas@indeed.com"

"""
This module contains utility code for the Vocabulary Analysis Workshop
"""

import codecs
from collections import Counter, defaultdict
import inspect
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
from wordcloud import WordCloud


def calculate_avg_tfidf(term_rows):
    """
    This function computes TF.IDF the most common manner.
    t := a term
    D := a corpus
    d := a document in D
    N := |D| the number of documents in D
    
    TF(t, d)     = number of times t occurs in d
    DF(t)        = |{d | t in d}|
    IDF(t)       = log_2(1 + N / DF(t))
    TF.IDF(t, d) = TF(t, d) * IDF(t)
    
    Parameters
    ----------
    tf : pandas.Series | numpy.ndarray | int | float
        1. (all d, all t) The sum of term occurrences for each term over all documents
        2. (all d, one t) The term occurrences for the term for all the documents
        3. (one d, all t) The term occurrences for all terms in the document
        4. (one d, one t) The term occurrences for the term for the document
    df : pandas.Series | numpy.ndarray | int | float
        1. (all d, all t) The document frequencies for each term
        2. (all d, one t) The document frequency for the term
        3. (one d, all t) The document frequencies for all terms in the document
        4. (one d, one t) The document frequency for the term
    n : int | float
        the number of documents
    ----------
    term_rows : pandas.Series[list[str]]
        Each row represents the terms in the document
    scheme : str
        the manner of calculating TF.IDF, default is "simple"
        see calculate_tfidf for currently supported schemes
    **kwargs
        additional arguments that are needed by some schemes
    Returns
    ----------
    pandas.DataFrame
        a dataframe with three columns, one for TF, IDF and TF.IDF
    """
    bags = term_rows.apply(Counter) # convert the documents to bags, this will calculate the TF per document per term
    sum_tf = Counter() # this will hold the sum of the TF per term
    df = Counter() # this will calculate the raw DF (n_t from above)
    for bag in bags:
        sum_tf.update(bag)
        df.update(bag.keys())
    sum_tf = pd.Series(sum_tf)
    df = pd.Series(df)
    idf = np.log2(1 + len(term_rows) / df)
    sum_tfidf = sum_tf * idf # this will calculate the sum TF.IDF per term
    avg_tfidf = sum_tfidf / len(term_rows)  # this will calculate the average TF.IDF per term over the documents
    return pd.DataFrame({'sum_tf': sum_tf, 'idf': idf, 'avg_tfidf': avg_tfidf})


def plot_tfidf_freqs(avg_tfidf_df, n, title='sum TF vs IDF', ax=None):
    """
    This function plots the TF vs the IDF as a scatterplot. The marks are colored from blue to red according to TF.IDF
    Parameters
    ----------
    avg_tfidf_df : pandas.DataFrame
        The output from calculate_avg_tfidf
    n : int
        The number of documents in the corpus
    title : str
        This is the title of the plot, default "TF vs IDF"
    ax : matplotlib.pyplot.Axes
        Optional
        This is the subplot on which to make the plot. If not provided, a new figure is created.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
    ax.scatter(avg_tfidf_df.sum_tf, avg_tfidf_df.idf, c=avg_tfidf_df.avg_tfidf, cmap=plt.cm.coolwarm)
    ax.set_xbound(0, avg_tfidf_df.sum_tf.max())
    ax.set_ybound(0, avg_tfidf_df.idf.max())
    ax.set_xlabel('sum TF')
    ax.set_ylabel('IDF')
    ax.set_title(title)
    

def wordcloud(wtd_words_s, title='TF.IDF Wordcloud', ax=None):
    """
    This function plots the TF vs the IDF as a scatterplot. The marks are colored from blue to red according to TF.IDF
    Parameters
    ----------
    wtd_words_s : pandas.Series
        Generally, should be the tfidf column from the output of calculate_avg_tfidf, but all that is needed is a series 
        with terms in the index, and weights as values
    title : str
        This is the title of the plot, default "TF.IDF Wordcloud"
    ax : matplotlib.pyplot.Axes
        Optional
        This is the subplot on which to make the plot. If not provided, a new figure is created.
    """
    word2wt = wtd_words_s.to_dict()
    wc = WordCloud(background_color='white', normalize_plurals=False, colormap="winter").generate_from_frequencies(word2wt)
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    ax.imshow(wc)
    ax.axis('off')
    ax.set_title(title)
        

def plot_learning_curve(train_sizes, train_scores, test_scores, title, ylim=None, ax=None):
    """
    Generate a simple plot of the test and training learning curve.
    modified from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Parameters
    ----------
    train_sizes_abs : numpy.array[int]
        Numbers of training examples that has been used to generate the learning curve. Note that the 
        number of ticks might be less than n_ticks because duplicate entries will be removed.
        shape = (n_unique_ticks,)
        (from sklearn.model_selection.learning_curve)
    train_scores : numpy.array[float]
        Scores on training sets. 
        shape = (n_ticks, n_cv_folds)
        (from sklearn.model_selection.learning_curve)
    test_scores : numpy.array[float]
        Scores on test set. 
         shape = (n_ticks, n_cv_folds)
        (from sklearn.model_selection.learning_curve)
    title : str
        Title for the chart.
    ylim : (float, float)
        Optional 
        This is to fix the minimum and maximum y-values in the plot
    ax : matplotlib.pyplot.Axes
        Optional
        This is the subplot on which to make the plot. If not provided, a new figure is created.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    ax.legend(loc="best")
    

def analyze(data, term_col, named_segment_masks=None):
    """
    This function exists to simplify analyzing the processed terms in documents.
    Abstractly, it does the following
    1. Process the values input_col in data
    2. Add the new column to data as output_col
    3. The values of output_col are expected to be lists of terms (one list per row)
    4. Calculate TF, IDF, TF.IDF for output_col according to the given scheme
    5. Plot TF vs IDF for the whole data set
    6. Create a wordcloud with TF.IDF as the weights
    7. If segments were provided, then for each segment do steps 4. and 7. for the segment
    Parameters
    ----------
    data : pandas.DataFrame
        This is the data to be analyzed
    term_col : str
        The name of the column containing the terms for which TF.IDF will be calculated
    named_segment_masks : list[pandas.Series[bool]]
        Optional
        A list of masks to segment the data
    """
    general_tfidf_df = calculate_avg_tfidf(data[term_col])
    
    minimal_subplots = 2
    num_subplots = minimal_subplots + len(named_segment_masks)
    fig = plt.figure(figsize=(10, num_subplots * 6))
    
    print('generating plot 1 / {}'.format(num_subplots))
    sys.stdout.flush()
    overall_tf_vs_idf_ax = fig.add_subplot(num_subplots, 1, 1)
    plot_tfidf_freqs(general_tfidf_df, len(data), ax=overall_tf_vs_idf_ax)
    
    print('generating plot 2 / {}'.format(num_subplots))
    sys.stdout.flush()
    overall_wordcloud_ax = fig.add_subplot(num_subplots, 1, 2)
    wordcloud(general_tfidf_df['avg_tfidf'], title='Overall TF.IDF Wordcloud', ax=overall_wordcloud_ax)
    
    for i, segment_mask in enumerate(named_segment_masks):
        i += minimal_subplots + 1
        print('generating plot {} / {}'.format(i, num_subplots))
        sys.stdout.flush()
        segment_tfidf_df = calculate_avg_tfidf(data[segment_mask][term_col])
        wordcloud_ax = fig.add_subplot(num_subplots, 1, i)
        wordcloud(segment_tfidf_df['avg_tfidf'], title='{} TF.IDF Wordcloud'.format(segment_mask.name), ax=wordcloud_ax)
    
    print('')
    print('Number of terms: ', len(general_tfidf_df))
    
    plt.show()
    

def build_indexes(term_col):
    """
    This function creates the indexes necessary for searching a corpus
    Parameters
    ----------
    term_col : str
        The name of the column containing the terms for which will be used for searching
    Returns
    ----------
    pd.Series[Counter]
        this serves as a mapping from document id to the associated bag-of-words
    pd.Series[str]
        this serves as a mapping from term to a set of document ids
    """
    doc_index = term_col.apply(Counter)
    inv_index = defaultdict(set)
    for ix, bag in doc_index.iteritems():
        for term in bag:
            inv_index[term].add(ix)
    inv_index = pd.Series(inv_index)
    return doc_index, inv_index


def search(query, docs, doc_index, inv_index, idf, processing, limit=10):
    """
    This function is used to search the given documents
    Parameters
    ----------
    query: str
        This is the query. example: "data scientist"
    docs: pd.Series[str]
        This is the text to be displayed
    doc_index: pd.Series[Counter[str]]
        this serves as a mapping from document id to the associated bag-of-words (see build_indexes)
    inv_index: pd.Series[list[str]]
        this serves as a mapping from term to a set of document ids (see build_indexes)
    idf: pd.Series[float]
        this is the IDF for the terms in the corpus
    processing : (str | list[str]) -> list[str]
        this is the processing done on the text of the corpus
    limit : int
        this is the number of result documents to display
    """
    terms = set(processing(query)) # always process your queries like you process your documents
    filter_set_ixs = set()
    term_idfs = idf[terms]
    for term in terms:
        filter_set_ixs |= inv_index.loc[term]
    # we should only return documents that contain at least one word from the query
    filter_set = doc_index.loc[filter_set_ixs]
    tf_df = pd.DataFrame({term: filter_set.apply(lambda bag: bag[term]) for term in terms})
    tfidf_df = tf_df * term_idfs
    score_df = tfidf_df.apply(np.sum, axis=1).sort_values(ascending=False)
    for doc_id, score in score_df[:limit].iteritems():
        print('=' * 80)
        print(doc_id)
        print('=' * 30)
        print(docs.loc[doc_id])
        print('=' * 80)

    
def save_fun(fun, imports=None, star_imports=None, **data):
    """
    This function will save a function as a .py file named my_<name of function> in current working directory
    Parameters
    ----------
    fun : callable
        This is the function to be saved. It is added to the __all__ of the .py file
    imports : list[str]
        This is a list of imports written as "import x"
    star_imports : list[str]
        This is a list of imports written as "from x import *"
    **data : Any
        Keyword arguments passed to save_fun will be pickled in the current working directory as 
        "my_<function name>.<keyword>.pickle". In the .py file code is written for unpickling the object. It 
        is also included in the__all__
    """
    with codecs.open('my_' + fun.func_name + '.py', 'w', encoding='UTF-8') as out:
        out.write('#!/usr/bin/env python\n')
        out.write('# -*- coding: UTF-8 -*-\n\n')
        imports = list(imports) if imports is not None else []
        imports.append('pickle')
        for imp in imports:
            out.write('import {}'.format(imp))
            out.write('\n')
        out.write('\n')
        if star_imports is not None:
            for imp in star_imports:
                out.write('from {} import *'.format(imp))
                out.write('\n')
            out.write('\n')
        for name, value in data.items():
            with open('my_{0}.{1}.pickle'.format(fun.func_name, name), 'wb') as fp:
                pickle.dump(value, fp)
            out.write('{1} = pickle.load(open(\'my_{0}.{1}.pickle\'))'.format(fun.func_name, name))
        out.write('\n\n')
        out.write(inspect.getsource(fun).decode('UTF-8'))
        out.write('\n\n')
        out.write('__all__ = [\'{}\']'.format("', '".join([fun.func_name] + data.keys())))


__all__ = ['calculate_avg_tfidf', 'plot_tfidf_freqs', 'wordcloud', 'analyze', 'build_indexes', 'search', 'save_fun', 
           'plot_learning_curve']
