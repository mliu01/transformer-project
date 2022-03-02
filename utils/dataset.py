# %% endofcell="--"
from os.path import join
import numpy as np
import string
from bs4 import BeautifulSoup
punctuations = string.punctuation
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
# test if dataset files are there    
folder_path = Path('data')
dataset_name = 'blurbs'

f_type = {
    'input': '.txt', 'output': '.json'
}

level = 3

def train_file(name=dataset_name, t='input'):
    return folder_path.joinpath('{}_train{}'.format(name, f_type[t]))

def dev_file(name=dataset_name, t='input'):
    return folder_path.joinpath('{}_dev{}'.format(name, f_type[t]))

def test_file(name=dataset_name, t='input'):
    return folder_path.joinpath('{}_test{}'.format(name, f_type[t]))

try:
    with open(train_file()) as f:
        f.close()
    with open(dev_file()) as f:
        f.close()
    with open(test_file()) as f:
        f.close()
except FileNotFoundError:
    print("Files not found")

# %%
def main():
    logger = logging.getLogger(__name__)
    logger.info(f"Building datasets with hierarchy level: {level}")

    # %%
    train_data = load_data(train_file(), 'train')
    dev_data = load_data(dev_file(), 'train')
    test_data = load_data(test_file(), 'train')

    # %%
    df_train = pd.concat([pd.DataFrame(train_data), pd.DataFrame(dev_data)])
    df_test = pd.DataFrame(test_data)

    splits = {'train': df_train, 'test': df_test}
    h = extract_hierarchies()


    # %%
    for key, df in splits.items():
        df.rename(columns={0: 'text'}, inplace=True)
        df.loc[:, 'path_list'] = df[1].map(lambda x: '>'.join(x[0:level])) #only categories going to stated hierarchy level
        df.loc[:, 'list'] = df[1].map(lambda x: x[0:level])
        df.loc[:, 'label'] = df['list'].map(lambda x: x[-1])

        df[[f'lvl{i+1}' for i in range(level)]] = pd.DataFrame(df.list.tolist(), index= df.index)
        
        df = df[['label', 'text', 'path_list'] + [f'lvl{i+1}' for i in range(level)]]
        df = df.replace(to_replace='None', value=np.nan).dropna() # removes any empty rows
        
        # remove rows not in hierarchy
        for lvl in range(level):
            for label in df[f'lvl{lvl+1}']:
                if label not in h[0][lvl]:
                    df.drop(df[df[f'lvl{lvl+1}'] == label].index, inplace=True)
            if lvl+1 == 3 and key != 'train':
                for label in df[f'lvl{lvl+1}']:
                    if label not in df_train['label'].values:
                        df.drop(df[df[f'lvl{lvl+1}'] == label].index, inplace=True)

        logger.info(f"Initial {key} dataset length: {len(df)}")            
        df.to_json(folder_path.joinpath(f"{dataset_name}_{key}{f_type['output']}"), orient = "records", lines=True, force_ascii=False)

# %%
def extra_main(name, subcount):
    logger = logging.getLogger(__name__)
    logger.info(f"Building datasets with at least {subcount} entries and adding underscore to labels")

    train_check = pd.read_json(train_file(t='output'), orient='records', lines=True)
    test_check = pd.read_json(test_file(t='output'), orient='records', lines=True)

    ## using only labels present in all datasets
    train_check = train_check.groupby('label').filter(lambda x : len(x)>=subcount)
    relevant_labels = list(set.intersection(set(list(train_check['label'])), set(list(test_check['label']))))
    splits = {'train': train_check, 'test': test_check}

    for key, df in splits.items():
        df = df[df['label'].isin(relevant_labels)].copy()
        df['label'] = df['label'].apply(lambda x: x.replace(' ', '_'))
        logger.info(f"Finished {key} dataset length: {len(df)}")
        df.to_json(folder_path.joinpath(f"{name}_{key}{f_type['output']}"), orient = "records", lines=True, force_ascii=False)

# %%
def lowercase_main(ds, name):
    logger = logging.getLogger(__name__)
    logger.info("Building lowercased datasets")

    train_check = pd.read_json(train_file(name=ds, t='output'), orient='records', lines=True)
    test_check = pd.read_json(test_file(name=ds, t='output'), orient='records', lines=True)

    splits = {'train': train_check,'test': test_check}

    for key, df in splits.items():
        df = df.applymap(lambda s: s.lower() if type(s) == str else s)
        logger.info(f"(lowercase) {key} dataset length: {len(df)}")
        df.to_json(folder_path.joinpath(f'{name}_{key}.json'), orient = "records", lines=True, force_ascii=False)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    main()
    extra_main(name='part-blurbs', subcount=30)
    extra_main(name='part50-blurbs', subcount=50)
    extra_main(name='part80-blurbs', subcount=80)
    extra_main(name='part100-blurbs', subcount=100)
    lowercase_main(ds='part-blurbs', name='lowercase-blurbs')

# %%
