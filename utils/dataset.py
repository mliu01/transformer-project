# %% [markdown]
# # Preprocessing Dataset
# - `ds_name`: dataset name
# - `folder_path`: path to raw data and output path
# - `train_file`: training data file name
# - `dev_file`: validation data file name
# - `test_file`: test data file name
# - `hierarchy_file`: hierarchy data file name

# %% endofcell="--"
from os.path import join
import os
import numpy as np
import string
from bs4 import BeautifulSoup
punctuations = string.punctuation
import pandas as pd
from pathlib import Path
import logging
CV_NUM = 3

current_directory = os.getcwd()
print(current_directory)


# %%
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
# Paths   
ds_name = 'blurbs'

folder_path = Path(current_directory + '/data')
train_file = folder_path.joinpath(f'{ds_name}_train.txt')
dev_file = folder_path.joinpath(f'{ds_name}_dev.txt')
test_file = folder_path.joinpath(f'{ds_name}_test.txt')
hierarchy_file = folder_path.joinpath('hierarchy.txt')

assert (
    folder_path.exists() and train_file.exists() and dev_file.exists() and test_file.exists() and hierarchy_file.exists()
), "Directory or Files missing!" 

# %%
train_data = load_data(train_file, 'train')
dev_data = load_data(dev_file, 'train')
test_data = load_data(test_file, 'train')

# %%
df = pd.concat([pd.DataFrame(train_data), pd.DataFrame(dev_data), pd.DataFrame(test_data)])
#df

# %%
def format_df(df, level):
    df.rename(columns={0: 'text'}, inplace=True)
    df.loc[:, 'list'] = df[1].map(lambda x: x[0:level])
    df.loc[:, 'label'] = df['list'].map(lambda x: x[-1])

    lvl_list = [f'lvl{i+1}' for i in range(level)]
    df[lvl_list] = pd.DataFrame(df.list.tolist(), index= df.index)

    df = df[['label', 'text'] + lvl_list]
    df = df.replace(to_replace='None', value=np.nan).dropna() 
    return df
# %%
def full_dataset(level):
    logger = logging.getLogger(__name__)
    logger.info(f"Building datasets with hierarchy level: {level}")
    
    h = extract_hierarchies()
    
    df_full = format_df(df, level)
    # remove rows not in hierarchy
    for lvl in range(level):
        for label in df_full[f'lvl{lvl+1}']:
            if label not in h[0][lvl]:
                df_full.drop(df_full[df_full[f'lvl{lvl+1}'] == label].index, inplace=True)

    logger.info(f"Initial dataset length: {len(df_full)}") 
         
    df_full.to_json(folder_path.joinpath(f"{ds_name}_full.json"), orient = "records", lines=True, force_ascii=False)

# %%
def extra_processing(ds_name='blurbs', out='part-blurbs', minOcc=30, split=0.8):
    logger = logging.getLogger(__name__)
    logger.info(f"Building datasets with at least {minOcc} entries and adding underscore to labels")

    df = pd.read_json(folder_path.joinpath(f"{ds_name}_full.json"), orient='records', lines=True)
    df = df.groupby('label').filter(lambda x : len(x)>=minOcc)

    df['label'] = df['label'].apply(lambda x: x.replace(' ', '_'))

    logger.info(f"Finished dataset length: {len(df)}")

    train_data = df.sample(frac=split)
    test_data = df.drop(train_data.index)
    logger.info(f"Train dataset length: {len(train_data)}")
    logger.info(f"Test dataset length: {len(test_data)}")

    train_data.to_json(
        folder_path.joinpath(f"{out}_train.json"), orient = "records", lines=True, force_ascii=False)
    test_data.to_json(
        folder_path.joinpath(f"{out}_test.json"), orient = "records", lines=True, force_ascii=False)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    full_dataset(level=3) #complete dataset with hierarchy level 3
    
    extra_processing()

    train = folder_path.joinpath("part-blurbs_train.json")
    test = folder_path.joinpath("part-blurbs_test.json")

    for i in range(2):
        df_train = pd.read_json(train, orient='records', lines=True)
        df_train['label'] = df_train[f'lvl{i+1}'].apply(lambda x: x.replace(' ', '_'))
    
        df_test = pd.read_json(test, orient='records', lines=True)
        df_test['label'] = df_test[f'lvl{i+1}'].apply(lambda x: x.replace(' ', '_'))

        df_train.to_json(
            folder_path.joinpath(f"part-blurbs-lvl{i+1}_train.json"), orient = "records", lines=True, force_ascii=False)
        df_test.to_json(
            folder_path.joinpath(f"part-blurbs-lvl{i+1}_test.json"), orient = "records", lines=True, force_ascii=False)


# %%
