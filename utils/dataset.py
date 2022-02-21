# %% endofcell="--"
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
from pathlib import Path
import logging

CV_NUM = 3
# -

# %%
# +
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
    REL_FILE =  Path('data/hierarchy.txt')
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
                            if element in labels and get_level_genre(element, hierarchy) == 0:
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


# %%
def main():
    logger = logging.getLogger(__name__)
    logger.info("Building datasets")

    folder_path = Path('data')
    dataset_name = 'blurbs'

    f_type = {
        'input': '.txt', 'output': '.json'
    }

    level = 3

    def train_file(t):
        return folder_path.joinpath('{}_train{}'.format(dataset_name, f_type[t]))

    def dev_file(t):
        return folder_path.joinpath('{}_dev{}'.format(dataset_name, f_type[t]))

    def test_file(t):
        return folder_path.joinpath('{}_test{}'.format(dataset_name, f_type[t]))


    # %%
    try:
        with open(train_file('input')) as f:
            f.close()
        with open(dev_file('input')) as f:
            f.close()
        with open(test_file('input')) as f:
            f.close()
    except FileNotFoundError:
        print("Files not found")

    # %%
    train_data = load_data(train_file('input'), 'train')
    dev_data = load_data(dev_file('input'), 'train')
    test_data = load_data(test_file('input'), 'train')

    # %%
    df_train = pd.DataFrame(train_data)
    df_dev = pd.DataFrame(dev_data)
    df_test = pd.DataFrame(test_data)

    splits = {'train': df_train, 'dev': df_dev, 'test': df_test}
    h = extract_hierarchies()


    # %%
    for key, df in splits.items():
        df.rename(columns={0: 'text'}, inplace=True)
        df.loc[:, 'path_list'] = df[1].map(lambda x: '>'.join(x[0:level])) #only categories going to stated hierarchy level
        df.loc[:, 'list'] = df[1].map(lambda x: x[0:level])
        df.loc[:, 'label'] = df['list'].map(lambda x: x[-1])

        df[['lvl{}'.format(i+1) for i in range(level)]] = pd.DataFrame(df.list.tolist(), index= df.index)
        
        df = df[['label', 'text', 'path_list'] + ['lvl{}'.format(i+1) for i in range(level)]]
        df = df.replace(to_replace='None', value=np.nan).dropna() # removes any empty rows
        
        # remove rows not in hierachy
        for lvl in range(level):
            for label in df['lvl{}'.format(lvl+1)]:
                if label not in h[0][lvl]:
                    df.drop(df[df['lvl{}'.format(lvl+1)] == label].index, inplace=True)
            if lvl+1 == 3 and key != 'train':
                for label in df['lvl{}'.format(lvl+1)]:
                    if label not in df_train['label'].values:
                        df.drop(df[df['lvl{}'.format(lvl+1)] == label].index, inplace=True)

                    
        df.to_json(folder_path.joinpath('{}_{}{}'.format(dataset_name, key, f_type['output'])), orient = "records", lines=True, force_ascii=False)

    # %%
    #check
    train_check = pd.read_json(train_file('output'), orient='records', lines=True)
    dev_check = pd.read_json(dev_file('output'), orient='records', lines=True)
    test_check = pd.read_json(test_file('output'), orient='records', lines=True)

    # %%
    ## using only labels present in all datasets
    relevant_labels = list(set.intersection(set(list(train_check['label'])), set(list(dev_check['label'])), set(list(test_check['label']))))
    splits = {'train': train_check, 'dev': dev_check, 'test': test_check}

    for key, df in splits.items():
        df = df[df['label'].isin(relevant_labels)]
        logger.info("Initial {} dataset length: {}".format(key, len(df)))
        df.to_json(folder_path.joinpath('{}_{}{}'.format(dataset_name, key, f_type['output'])), orient = "records", lines=True, force_ascii=False)

    train_check = pd.read_json(train_file('output'), orient='records', lines=True)
    dev_check = pd.read_json(dev_file('output'), orient='records', lines=True)
    test_check = pd.read_json(test_file('output'), orient='records', lines=True)
    
    ##
    splits = {'train': train_check, 'dev': dev_check, 'test': test_check}

    new_train = pd.concat([train_check, dev_check])
    new_train = new_train.groupby('label').filter(lambda x : len(x)>=30)

    for key, df in splits.items():
        df = df[df['label'].isin(new_train['label'])]
        logger.info("{} dataset length: {}".format(key, len(df)))
        df.to_json(folder_path.joinpath('{}_{}{}'.format(dataset_name, key, f_type['output'])), orient = "records", lines=True, force_ascii=False)

    new_train.to_json(folder_path.joinpath('{}_{}{}'.format(dataset_name, 'train2', f_type['output'])), orient = "records", lines=True, force_ascii=False)
    logger.info("Training with validation dataset length: {}".format(len(new_train)))



    # %%
    ## only 3 labels
    #train_check[(train_check['label']=='Literatur & Unterhaltung') | (train_check['label']=='Sachbuch') | (train_check['label']=='Kinderbuch & Jugendbuch')].to_json('../data/blurbs3_train'+".json", orient = "records", lines=True, force_ascii=False)

    # %%
    #dev_check[(dev_check['label']=='Literatur & Unterhaltung') | (dev_check['label']=='Sachbuch') | (dev_check['label']=='Kinderbuch & Jugendbuch')].to_json('../data/blurbs3_dev'+".json", orient = "records", lines=True, force_ascii=False)

    # %%
    #test_check[(test_check['label']=='Literatur & Unterhaltung') | (test_check['label']=='Sachbuch') | (test_check['label']=='Kinderbuch & Jugendbuch')].to_json('../data/blurbs3_test'+".json", orient = "records", lines=True, force_ascii=False)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    main()

# %%
