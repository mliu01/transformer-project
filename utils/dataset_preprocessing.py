# %% [markdown]
# # Preprocessing Blurbs Dataset
# (from https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc/)
#
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
# Setting paths   
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
#Loading dataset
train_data = load_data(train_file, 'train')
dev_data = load_data(dev_file, 'train')
test_data = load_data(test_file, 'train')

# %%
# all entries
df = pd.concat([pd.DataFrame(train_data), pd.DataFrame(dev_data), pd.DataFrame(test_data)])


# %%
def format_df(df):
    '''Formats dataset correctly. Renaming columns, adding underscores to all columns except text, concat labels and drops irrelevant columns.'''

    ndf = df.copy()
    ndf.rename(columns={0: 'text'}, inplace=True)
    ndf.loc[:, 'list'] = ndf[1].map(lambda x: x[0:3])

    # splits list in seperate columns (needed for later for building tree/graph)
    lvl_list = [f'lvl{i+1}' for i in range(3)]
    ndf[lvl_list] = pd.DataFrame(ndf.list.tolist(), index= ndf.index)

    #concatenate labels from each level, so there are no duplicates; label is always leaf node
    ndf['lvl2'] = ndf['lvl1'] + '/' + ndf['lvl2']
    ndf['lvl3'] = ndf['lvl2'] + '/' + ndf['lvl3']
    ndf['label'] = ndf['lvl3']
    
    # adds underscore to all but text
    replacing_columns = list(ndf.drop('text', axis=1).columns)
    ndf[replacing_columns] = ndf[replacing_columns].replace(' ', '_', regex=True)

    # only relevant columns
    ndf = ndf[['label', 'text'] + lvl_list] 
    ndf = ndf.dropna()

    return ndf
# %%
# saves full dataset as is
def full_dataset():
    logger = logging.getLogger(__name__)

    logger.info(f"Building datasets with hierarchy level 3")
    df_full_lvl = format_df(df)
    logger.info(f"Initial dataset with 3 hierarchy level/s length: {len(df_full_lvl)}") 

    new_ds_name = f"{ds_name}_full.json"
    df_full_lvl.to_json(output_path.joinpath(new_ds_name), orient = "records", lines=True, force_ascii=False)
    return new_ds_name

# %%
def extra_processing(ds_file='blurbs_full.json', out_prefix='part-blurbs', minOcc=100):
    logger = logging.getLogger(__name__)
    logger.info(f"Building datasets with at least {minOcc} entries")

    df = pd.read_json(output_path.joinpath(ds_file), orient='records', lines=True)

    # each path has to occur at least minOcc times
    #df = df.groupby(df['path_list'].map(tuple)).filter(lambda x : len(x)>=minOcc)

    # alternative: each label in last hierarchy level has to occur at least minOcc times
    df = df.groupby(df['label'].map(tuple)).filter(lambda x : len(x)>=minOcc)

    logger.info(f"Finished dataset length: {len(df)}")
    df.to_json(output_path.joinpath(f"{out_prefix}_full.json"), orient = "records", lines=True, force_ascii=False)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    lvl_ds = full_dataset() # complete dataset with hierarchy level 3, returns new files name
    extra_processing(ds_file=lvl_ds, out_prefix='blurbs_reduced', minOcc=100)


# %% [markdown]
# # Build Tree from Dataset
# Takes dataset and builds a graph with every possible path from root to leaf node.
# Specifically coded for 3 hierarchy levels (json_line, nodes dictionaries)
# - `name`: name of dataset to build tree from

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
# optional, used for visual represenation of graph
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
def main():
    G = nx.DiGraph()

    logger = logging.getLogger(__name__)
    logger.info("Start to build Graph")

    #Node Dict
    root = 'Root'
    inserted_nodes = 0

    # Add root node
    dict_nodes = {root: inserted_nodes}
    G.add_node(dict_nodes[root], name=root)
    inserted_nodes += 1

    count = 0
    files = "./data/blurbs_dataset/blurbs_reduced_full.json"

    #for file in files:
    new_nodes = []
    with open(files, encoding="utf8") as fp:
        while True:
            count += 1
            line = fp.readline()

            if not line:
                logger.info('Read {} lines'.format(count))
                logger.info('Added {} nodes'.format(inserted_nodes))
                print(G.nodes(data=True))
                break

            json_line = json.loads(line)

            # Create labels
            lvl1 = json_line['lvl1'].replace(" ","_")
            lvl2 = json_line['lvl2'].replace(" ","_")
            lvl3 = json_line['lvl3'].replace(" ","_")

            # Add labels to graph
            nodes = [lvl1, lvl2, lvl3]
            predecessors = [root, lvl1, lvl2]

            for node, predecessor in zip(nodes, predecessors):
                if node not in dict_nodes:
                    dict_nodes[node] = inserted_nodes
                    G.add_node(dict_nodes[node], name=node)
                    G.add_edge(dict_nodes[predecessor], dict_nodes[node])
                    new_nodes.append(node)
                    inserted_nodes += 1

    logger.info('Done: Total of {} nodes'.format(len(G.nodes(data=True))))
    plt.figure(figsize=(10,5))

    pos = hierarchy_pos(G,0)    
    nx.draw(G, pos=pos, with_labels=True, font_size = 8)
    plt.savefig(output_path.joinpath(f'hierarchy_blurbs_reduced_full.png'))

    # Save tree
    with open(output_path.joinpath(f'tree_blurbs_reduced_full.pkl'), "wb") as file:
        pickle.dump(G, file=file)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    main()

# %%
