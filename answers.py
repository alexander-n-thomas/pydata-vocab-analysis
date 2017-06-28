#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Alex Thomas"
__credits__ = ["Alex Thomas", "Annette Taberner-Miller"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alex Thomas"
__email__ = "althomas@indeed.com"

"""
This module contains the answers to the excersizes in the Vocabulary Analysis Workshop
"""

import nltk


def tokenize(job_description):
    """
    This function takes a job description and returns a list of tokens
    Parameters
    ----------
    job_description : str
        The text of the job description
    Returns
    ----------
    list[str]
        The list of tokens in the job description
    """
    return nltk.tokenize.regexp_tokenize(job_description, r'[$£€¥][\d,\.]+|\w+|[^\w\s]', gaps=False)


def lemmatize(tokens, mapping):
    """
    This function takes tokens and returns the lemma of the word.
    Parameters
    ----------
    tokens : list[str]
        the list of tokens to be stemmed
    mapping : dict[str, str]
        the mapping from word to lemma
    Returns
    ----------
    list[str]
        for each token, either the lemma or the token
    """
    lemmas = []
    for token in tokens:
        if not token.isalpha():
            continue
        lowered_token = token.lower()
        lemmas.append(mapping.get(lowered_token, lowered_token))
    return lemmas


additional_stopwords = set([
    'requirement',
    'must',
    'position',
    'work',
    'ability',
    'company',
    'job',
    'look',
    'able',
    'www',
    'example',
    'com',
    'may',
])


__all__ = ['tokenize', 'lemmatize', 'additional_stopwords']