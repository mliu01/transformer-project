# + endofcell="--"
import os
from os.path import join
import json
import codecs
import argparse
import numpy as np
import io
import pickle
import string
from bs4 import BeautifulSoup
punctuations = string.punctuation
import pickle
import pandas as pd
CV_NUM = 3
# -
# --

# +
# # +
def load_isbns(directory):
    isbns = []
    soup = BeautifulSoup(open(join(directory), 'rt', encoding='utf-8').read(), "html.parser")
    for book in soup.findAll('book'):
        book_soup = BeautifulSoup(str(book), "html.parser")
        isbns.append(str(book_soup.find("isbn").string))
    return isbns


def load_data(directory, status):
    """
    Loads labels and blurbs of dataset
    """
    data = []
    soup = BeautifulSoup(open(join(directory), 'rt', encoding='utf-8').read(), "html.parser")
    for book in soup.findAll('book'):
        if status == 'train':
            categories = list()
            book_soup = BeautifulSoup(str(book), "html.parser")
            for t in book_soup.findAll('topic'):
                categories.append(str(t.string))
            data.append((str(book_soup.find("body").string), categories))
        elif status == 'test':
            book_soup = BeautifulSoup(str(book), "html.parser")
            data.append(str(book_soup.find("body").string))
    return data


def extract_hierarchies():
    """
    Returns dictionary with level and respective genres
    """
    hierarchies_inv = {}
    relations, singletons = read_relations()
    genres = set([relation[0] for relation in relations] +
    [relation[1] for relation in relations]) | singletons
    #genres, _= read_all_genres(language, max_h)
    for genre in genres:
        if not genre in hierarchies_inv:
            hierarchies_inv[genre] = 0
    for genre in genres:
        hierarchies_inv[genre], _ = get_level_genre(relations, genre)
    hierarchies = {}
    for key,value in hierarchies_inv.items():
        if not value in hierarchies:
            hierarchies[value] = [key]
        else:
            hierarchies[value].append(key)
    return [hierarchies,hierarchies_inv]

def read_relations():
    """
    Loads hierarchy file and returns set of relations
    """
    relations = set([])
    singeltons = set([])
    REL_FILE =  '../data/hierarchy.txt'
    with open(REL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            rel = line.split('\t')
            if len(rel) > 1:
                rel = (rel[0], rel[1][:-1])
            else:
                singeltons.add(rel[0][:-1])
                continue
            relations.add(rel)
    return [relations, singeltons]

def get_level_genre(relations, genre):
    """
    return hierarchy level of genre
    """
    height = 0
    curr_genre = genre
    last_genre = None
    while curr_genre != last_genre:
        last_genre = curr_genre
        for relation in relations:
            if relation[1] == curr_genre:
                height+=1
                curr_genre = relation[0]
                break
    return height, curr_genre

def adjust_hierarchy(output_b, binarizer):
    """
    Correction of nn predictions by either restrictive or transitive method
    """
    global ml
    print("Adjusting Hierarchy")
    relations,_ = read_relations()
    hierarchy, _ = extract_hierarchies()
    new_output = []
    outputs = binarizer.inverse_transform(output_b)
    for output in outputs:
        labels = set(list(output))
        if len(labels) >= 1:
            labels_cp = labels.copy()
            labels_hierarchy = {}
            for level in hierarchy:
                for label in labels:
                    if label in hierarchy[level]:
                        if level in labels_hierarchy:
                            labels_hierarchy[level].add(label)
                        else:
                            labels_hierarchy[level] = set([label])
            for level in labels_hierarchy:
                if level > 0:
                    for label in labels_hierarchy[level]:
                        all_parents = get_parents(label, relations)
                        missing = [parent for parent in all_parents if not parent in labels]
                        no_root = True
                        for element in missing:
                            if element in labels and get_genre_level(element, hierarchy) == 0:
                                labels = labels | all_parents
                                no_root = False

                        if len(missing) >= 1:
                            labels = labels | set(all_parents)
        new_output.append(tuple(list(labels)))
    return binarizer.transform(new_output)

def get_parents(child, relations):
    """
    Get the parent of a genre
    """
    parents = []
    current_parent = child
    last_parent = None
    while current_parent != last_parent:
        last_parent = current_parent
        for relation in relations:
            if relation[1] == current_parent:
                current_parent = relation[0]
                parents.append(current_parent)
    return parents


# -

train_file = "../data/blurbs_train"
dev_file = "../data/blurbs_dev"
test_file = "../data/blurbs_test"

try:
    with open(train_file+".txt") as f:
        f.close()
    with open(dev_file+".txt") as f:
        f.close()
    with open(test_file+".txt") as f:
        f.close()
except FileNotFoundError:
    print("Files not found")

train_data = load_data(train_file+".txt", 'train')
dev_data = load_data(dev_file+".txt", 'train')
test_data = load_data(test_file+".txt", 'train')

df_train = pd.DataFrame(train_data)
df_train.loc[:, 'label'] = df_train[1].map(lambda x: x[0]) #only first category
df_train.rename(columns={0: 'text'}, inplace=True)

df_dev = pd.DataFrame(dev_data)
df_dev.loc[:, 'label'] = df_dev[1].map(lambda x: x[0]) #only first category
df_dev.rename(columns={0: 'text'}, inplace=True)

df_test = pd.DataFrame(test_data)
df_test.loc[:, 'label'] = df_test[1].map(lambda x: x[0]) #only first category
df_test.rename(columns={0: 'text'}, inplace=True)

#export dataframe to json
df_train[['label', 'text']].to_json(train_file+".json", orient = "records", lines=True, force_ascii=False)
df_dev[['label', 'text']].to_json(dev_file+".json", orient = "records", lines=True, force_ascii=False)
df_test[['label', 'text']].to_json(test_file+".json", orient = "records", lines=True, force_ascii=False)

#check if all labels are from the first level
train_check = pd.read_json(train_file+".json", orient='records', lines=True)
dev_check = pd.read_json(dev_file+".json", orient='records', lines=True)
test_check = pd.read_json(test_file+".json", orient='records', lines=True)
# # +

# + endofcell="--"
h = extract_hierarchies()

for i in train_check['label']:
    if i not in h[0][0]:
        print('unkown labels in train')
if len(train_data) != len(train_check):
    print('something went wrong in train')

for i in dev_check['label']:
    if i not in h[0][0]:
        print('unkown labels in train')
if len(dev_data) != len(dev_check):
    print('something went wrong in train')
    
for i in train_check['label']:
    if i not in h[0][0]:
        print('unkown labels in test')           
if len(test_data) != len(test_check):
    print('something went wrong in test')
# -
# --

train_check[(train_check['label']=='Literatur & Unterhaltung') | (train_check['label']=='Sachbuch') | (train_check['label']=='Kinderbuch & Jugendbuch')].to_json('../data/blurbs3_train'+".json", orient = "records", lines=True, force_ascii=False)

dev_check[(dev_check['label']=='Literatur & Unterhaltung') | (dev_check['label']=='Sachbuch') | (dev_check['label']=='Kinderbuch & Jugendbuch')].to_json('../data/blurbs3_dev'+".json", orient = "records", lines=True, force_ascii=False)

test_check[(test_check['label']=='Literatur & Unterhaltung') | (test_check['label']=='Sachbuch') | (test_check['label']=='Kinderbuch & Jugendbuch')].to_json('../data/blurbs3_test'+".json", orient = "records", lines=True, force_ascii=False)
