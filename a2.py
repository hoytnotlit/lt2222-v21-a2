import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

# TODO poista tää
def main():
    gmbfile = open('/scratch/lt2222-v21-resources/GMB_dataset.txt', "r")
    inputdata = preprocess(gmbfile)
    instances = create_instances(inputdata)

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
# To preprocess the text (lowercase and lemmatize; punctuation can be preserved as it gets its own rows).
# You can return the data in any indexable form you like. You can also choose to remove 
# infrequent or uninformative words to reduce the size of the feature space.  
def preprocess(inputfile):
    # NOTE gmbfile is opened and closed in notebook
    result = []
    lemmatizer = WordNetLemmatizer() 

    #https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
    #https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    #https://stackoverflow.com/questions/5364493/lemmatizing-pos-tagged-words-with-nltk
    #https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

    # TODO You can also choose to remove infrequent or uninformative words to reduce the size of the feature space. 

    # skip first/header line
    for line in inputfile.readlines()[1:]:
        # TODO should ne's be lemmatized..?
        sent_list = line.split()
        sent_list[2] = lemmatizer.lemmatize(sent_list[2].lower(), get_wordnet_pos(sent_list[3]))
        result.append(sent_list)

    return result

# POS-tags are in Penn Treebank format - lemmatizer uses wordnet format
# get POS-tag in wordnet format
def get_wordnet_pos(treebank_tag):
    # TODO make better/add more tags (default is now N)
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Code for part 2

# create instances from every from every identified named entity in the text with the type 
# of the NE as the class, and a surrounding context of five words on either side as the features. 

#You will create a collection of Instance objects. Remember to consider the case where the NE is 
#at the beginning of a sentence or at the end, or close to either (you can create a special start 
#token for that). You can also start counting from before the B end of the NE mention and after the 
#last I of the NE mention. That means that the instances should include things before and after the 
#named entity mention, but not the named entity text itself.
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

            # TODO You can also start counting from before the B end of the NE mention and after the 
            #last I of the NE mention. That means that the instances should include things before and after the 
            #named entity mention, but not the named entity text itself.

            # get previous and next 5 tokens
            # TODO else data[0:i] correct?
            prev_items = items_to_features(data[i-5:i] if i-5 > 0 else data[0:i], item[1], "s") 
            next_items = items_to_features(data[i+1:i+6], item[1], "e")
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
# create a data table with "document" vectors representing each instance and split the table 
#into training and testing sets and random with an 80%/20% train/test split.

def create_table(instances):
    word_list = set([f for x in instances for f in x.features])
    df_dict = {"class": [x.neclass for x in instances]}

    # construct dict for dataframe
    for i, word in enumerate(word_list):
        df_dict[i] = [inst.features.count(word) for inst in instances]

        # what if word is class? use numbers as col names instead of words?
        # if word != "classname":
        #     df_dict[word] = [inst.features.count(word) for inst in instances]
    
    df = pd.DataFrame(df_dict)
    return df

# def create_table(instances):
#     # df = pd.DataFrame()
#     word_list = set([f for x in instances for f in x.features])
    # word_counts = {}
    # # get word frequencies
    # for inst in instances:
    #     for feat in inst.features:
    #         if feat not in word_counts:
    #             word_counts[feat] = 0
    #         word_counts[feat] += 1

    # print(len(word_counts))
    # TODO top frequent words?
    # build data frame
    # df['class'] = [x.neclass for x in instances]
    #df['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(100)]

    # classnames = {"class": [x.neclass for x in instances]}
    # print(len(classnames["class"]))
    # word_dict = {}
    # # construct dict for dataframe
    # for word in word_list:
    #     word_dict[word] = [inst.features.count(word) for inst in instances]

    # df = pd.DataFrame({**classnames, **word_dict})
    # # each column represents a word and the value is the occurence of the word in the features
    # # for i in range(0, len(instances)):
    # #     for word in word_list:
    # #         df.loc[i, word] = instances[i].features.count(word) #/ word_counts[word]
    # #         # df[i] = npr.random(100)
    # return df

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
    df = pd.crosstab(actual, predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
    return df

# Code for bonus part B
def bonusb(filename):
    pass
