import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
    result = []
    lemmatizer = WordNetLemmatizer() 

    # skip first/header line
    for line in inputfile.readlines()[1:]:
        sent_list = line.split()
        sent_list[2] = lemmatizer.lemmatize(sent_list[2].lower(), get_wordnet_pos(sent_list[3]))
        result.append(sent_list)

    return result

# POS-tags are in Penn Treebank format - lemmatizer uses wordnet format
# get POS-tag in wordnet format (which there are only four of)
# handle conversion of tags for adjectives, verbs and adverbs and fallback to noun-tag
def get_wordnet_pos(pt_pos_tag):
    if pt_pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pt_pos_tag.startswith('V'):
        return wordnet.VERB
    elif pt_pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data):
    instances = []

    for i, item in enumerate(data):
        neclass = item[4]

        if neclass != "O":
            neclass = neclass[2:5]

            # get indices to include words before and after the NE but not parts of the NE
            prev_start_index = i-5 if i-5 > 0 else 0
            next_start_index = i+1

            while data[prev_start_index+4][4].startswith("I") or data[prev_start_index+4][4].startswith("B"):
                prev_start_index = prev_start_index - 1

            while data[next_start_index][4].startswith("I"):
                next_start_index = next_start_index + 1

            # get previous and next 5 tokens
            prev_items = items_to_features(data[prev_start_index:prev_start_index+5] , item[1], "s") 
            next_items = items_to_features(data[next_start_index:next_start_index+5], item[1], "e")
            features = prev_items + next_items
            
            instances.append(Instance(neclass, features))

    return instances

def items_to_features(items, ne_index, position):
    # filter to contain only items from given context
    items = filter(lambda n_item: is_in_context(ne_index, n_item), items)

    # convert to list of words/tokens
    features = [x[2] for x in items]
    
    # add end/start tokens
    if len(features) < 5:
        token_i = 5
        while len(features) < 5:
            features.append(f'<{position}{token_i}>')
            token_i = token_i - 1

    return features

def is_in_context(index, item):
    return item[1] == index

# Code for part 3
def create_table(instances):
    word_list = set([f for x in instances for f in x.features])
    df_dict = {"class": [x.neclass for x in instances]}

    # construct dict for dataframe
    for i, word in enumerate(word_list):
        df_dict[i] = [inst.features.count(word) for inst in instances]

    df = pd.DataFrame(df_dict)
    return df

def ttsplit(bigdf):
    # select 80% of rows, get the remaining 20% by dropping train data from
    # complete dataframe, randomize with sample()
    df_train = bigdf.sample(frac=0.8).reset_index()
    df_test = bigdf.drop(df_train.index).sample(frac=1).reset_index()
    return df_train.drop('class', axis=1).to_numpy(), df_train['class'], df_test.drop('class', axis=1).to_numpy(), df_test['class']

# Code for part 5
def confusion_matrix(truth, predictions):
    actual = pd.Series(truth, name='Actual')
    predicted = pd.Series(predictions, name='Predicted')
    
    df = pd.crosstab(actual, predicted) # create matrix

    # add classes that have not been predicted with 0 values
    missing_predictions = list(set(truth) - set(predictions))
    
    for mp in missing_predictions:
        df[mp] = 0

    df = df.sort_index(axis=1) # sort columns

    # add totals of columns and rows
    df['Total'] = df.sum(axis=1)
    df.loc['Total']= df.sum()

    return df

# Code for bonus part B
def bonusb(filename):
    pass
