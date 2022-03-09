# %%
import json
import logging
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import random

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

    #Node Dict
    root = 'Root'
    inserted_nodes = 0

    # Add root node
    dict_nodes = {root: inserted_nodes}
    G.add_node(dict_nodes[root], name=root)
    inserted_nodes += 1

    count = 0

    files = ["data/{}_train.json".format(name), "data/{}_test.json".format(name)]
    
    for file in files:
        new_nodes = []
        with open(file, encoding='utf-8') as fp:
            while True:
                count += 1
                line = fp.readline()

                if not line:
                    logger.info('Read {} lines from {}'.format(count, file))
                    logger.info('Added {} nodes'.format(inserted_nodes))
                    #print(G.nodes(data=True))
                    break

                json_line = json.loads(line)
                # Create labels

                ### for different hierarchy levels
                # check = len([x for x in list(json_line.values())[-3:] if bool(x)])
                # lvl = OrderedDict()
                # for i in range(1, check+1):
                #     level = 'lvl{}'.format(i)
                #     lvl[level] = json_line[level].replace(" ","_")
                # nodes = [i for i in lvl.values()]
                # predecessors = [root]
                # for i in nodes[0:-1]:
                #     predecessors.append(i)
                    
                lvl1 = json_line['lvl1'].replace(" ","_")
                lvl2 = json_line['lvl2'].replace(" ","_")
                lvl3 = json_line['lvl3'].replace(" ","_")

                # Add labels to graph
                nodes = [lvl1, lvl2, lvl3]
                predecessors = [root, lvl1, lvl2]

                for node, predecessor in zip(nodes, predecessors):
                    if node not in dict_nodes:
                        dict_nodes[node] = inserted_nodes
                        G.add_node(dict_nodes[node], attribute=node)
                        G.add_edge(dict_nodes[predecessor], dict_nodes[node])
                        new_nodes.append(node)
                        inserted_nodes += 1

    logger.info('Done: Total of {} nodes'.format(len(G.nodes(data=True))))
    plt.figure(figsize=(10,5))

    pos = hierarchy_pos(G,0)    
    nx.draw(G, pos=pos, with_labels=True, font_size = 8)
    plt.savefig('data/hierarchy_{}.png'.format(name))

    # Save tree
    with open("./data/tree_{}.pkl".format(name), "wb") as file:
        pickle.dump(G, file=file)

# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    #main(name = 'blurbs')
    main(name = 'part-blurbs')
    main(name='part-blurbs-lvl1')
    main(name='part-blurbs-lvl2')   

# %%
