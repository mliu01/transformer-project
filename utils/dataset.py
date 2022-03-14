# %% [markdown]
# # Preprocessing Dataset
# Processes Dateset, Hierarchy Level is set to 3 (can be changed)
# - `ds_name`: inital dataset name
# - `folder_path`: path to raw data
# - `output_path`: output path
# - `train_file`: training data file name
# - `dev_file`: validation data file name
# - `test_file`: test data file name
# - `hierarchy_file`: hierarchy data file name (optional)

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
def load_data(directory, status):
    """
    Loads labels and blurbs of dataset TODO: BeautifulSoup is a very slow during parsing, alternatives?
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

# %%
# Paths   
ds_name = 'blurbs'

folder_path = Path(current_directory + '/data')
output_path = folder_path.joinpath(f"{ds_name}_dataset")
Path(output_path).mkdir(parents=True, exist_ok=True)

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


# %%
#df

# %%
def format_df(df, level=None):
    '''Renaming columns, adding underscores to all columns except text and if level is specified: create new column path_list'''

    ndf = df.copy()
    ndf.rename(columns={0: 'text'}, inplace=True)
    ndf.loc[:, 'list'] = ndf[1].map(lambda x: x[0:level])
    ndf.loc[:, 'label'] = ndf['list'].map(lambda x: x[-1])

    max_len = ndf["list"].map(lambda x: len(x)).max()

    # splits list in seperate columns (needed for later for building tree/graph)
    lvl_list = [f'lvl{i+1}' for i in range(max_len)]
    ndf[lvl_list] = pd.DataFrame(ndf.list.tolist(), index= ndf.index)
    
    # adds underscore to all but text
    replacing_columns = list(ndf.drop('text', axis=1).columns)
    ndf[replacing_columns] = ndf[replacing_columns].replace(' ', '_', regex=True)

    # adds path_list
    if level:
        assert level == max_len
        lvl_list = lvl_list[:level]

    ndf['path_list'] = ndf[lvl_list].values.tolist()
    # only relevant columns
    ndf = ndf[['label', 'text', 'path_list'] + lvl_list] 

    return ndf
# %%
def full_dataset(level=None):
    logger = logging.getLogger(__name__)
    if level:
        logger.info(f"Building datasets with hierarchy level: {level}")
        df_full_lvl = format_df(df, level)
        logger.info(f"Initial dataset with {level} hierarchy level/s length: {len(df_full_lvl)}") 

        new_ds_name = f"{ds_name}_lvl{level}_full.json"
        df_full_lvl.to_json(output_path.joinpath(new_ds_name), orient = "records", lines=True, force_ascii=False)
        return new_ds_name

    df_full = format_df(df)
    logger.info(f"Initial dataset length: {len(df_full)}")     
    new_ds_name = f"{ds_name}_full.json"
    df_full.to_json(output_path.joinpath(new_ds_name), orient = "records", lines=True, force_ascii=False)

    return new_ds_name #return new ds names 

# %%
def extra_processing(ds_file='blurbs_full.json', out_prefix='part-blurbs', minOcc=50, split=0.85):
    logger = logging.getLogger(__name__)
    logger.info(f"Building datasets with at least {minOcc} entries")

    df = pd.read_json(folder_path.joinpath(ds_file), orient='records', lines=True)
    df = df.dropna()

    # each path has to occur at least minOcc times
    df = df.groupby(df['path_list'].map(tuple)).filter(lambda x : len(x)>=minOcc)

    # alternative: each label in last hierarchy level has to occur at least minOcc times
    # df = df.groupby(df['label'].map(tuple)).filter(lambda x : len(x)>=minOcc)

    logger.info(f"Finished dataset length: {len(df)}")

    # split into train and test dataset
    train_data = df.sample(frac=split)
    test_data = df.drop(train_data.index)

    logger.info(f"Train dataset length: {len(train_data)}")
    logger.info(f"Test dataset length: {len(test_data)}")

    train_data.to_json(
        output_path.joinpath(f"{out_prefix}_train.json"), orient = "records", lines=True, force_ascii=False)
    test_data.to_json(
        output_path.joinpath(f"{out_prefix}_test.json"), orient = "records", lines=True, force_ascii=False)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    lvl_ds = full_dataset(level=3) # complete dataset with hierarchy level 3, returns new files name
    
    #extra_processing(ds_file=lvl_ds, out_prefix='blurbs_lvl3')
    #extra_processing(ds_file="./blurbs_dataset/blurbs_lvl3_full.json", out_prefix='blurbs_reduced', minOcc=200)


# %% [markdown]
# # Build Tree from Dataset
# Takes train and test dataset and builds a graph with every possible path from root to leaf node
# Is hard coded for 3 hierarchy levels (json_line, nodes dictionaries)
# - `name`: name of dataset to build tree from (prefix, i.e.: blurbs_train.json -> blurbs)

# %%
import json
import logging
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
current_directory = os.getcwd()
print(current_directory)

# %%
folder_path = Path(current_directory + '/data/blurbs_dataset')
output_path = folder_path
Path(output_path).mkdir(parents=True, exist_ok=True)


# %%
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)  
                  
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# %%
def main(name):
    G = nx.DiGraph()

    logger = logging.getLogger(__name__)
    logger.info("Start to build Graph")

    #Root
    root = 'Root'
    inserted_nodes = 0

    # Add root node
    dict_nodes = {inserted_nodes: root}

    dict_paths = {}

    nodes_lvl1 = dict()
    nodes_lvl2 = dict()
    nodes_lvl3 = dict()
    lvl_list = [nodes_lvl1, nodes_lvl2, nodes_lvl3]

    G.add_node(0, attribute=root, lvl=0, predecessor=None)
    inserted_nodes += 1

    files = ["{}/{}_train.json".format(str(folder_path), name), "{}/{}_test.json".format(str(folder_path), name)]
    
    for file in files:
        count = 0
        with open(file, encoding='utf-8') as fp:
            while True:
                count += 1
                line = fp.readline()

                if not line:
                    logger.info('Read {} lines from {}'.format(count, file))
                    logger.info('Added {} nodes'.format(inserted_nodes))
                    break

                json_line = json.loads(line)
                # Create labels
                    
                lvl1 = json_line['lvl1'].replace(" ","_")
                lvl2 = json_line['lvl2'].replace(" ","_")
                lvl3 = json_line['lvl3'].replace(" ","_")

                # Paths
                nodes = [lvl1, lvl2, lvl3]

                if nodes not in dict_paths.values():
                    dict_paths[inserted_nodes] = nodes
                    inserted_nodes += 1
                

    # create nodes based on each existing path
    inserted = 1
    for path in dict_paths.values():
        for i in range(3):
            if i == 0:
                predecessor = 0
            else:
                check = lvl_list[i-1]
                predecessor = [node[0] for node in check.items() if node[1] == path[i-1]][0]

            if path[i] not in lvl_list[i].values() or i == 2: #leaf nodes are allowed to have multiple nodes
                dict_nodes[inserted] = path[i]
                lvl_list[i][inserted] = path[i]
                G.add_node(inserted, attribute=dict_nodes[inserted], lvl=i+1, predecessor=predecessor)
                G.add_edge(predecessor, inserted)
                inserted += 1

    logger.info('Done: Total of {} nodes'.format(len(G.nodes(data=True))))
    plt.figure(figsize=(10,5))

    pos = hierarchy_pos(G,0)    
    nx.draw(G, pos=pos, with_labels=True, font_size = 8)
    plt.savefig(output_path.joinpath(f'hierarchy_{name}.png'))

    # Save tree
    with open(output_path.joinpath(f'tree_{name}.pkl'), "wb") as file:
        pickle.dump(G, file=file)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    #main(name='blurbs_lvl3')
    #main(name='blurbs_reduced')

# %%
