from pathlib import Path
import pickle

class TreeUtils():

    def __init__(self, folder, name):

        self.tree = None
        self.load_tree(folder, name)
        
        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

    def load_tree(self, path, name):
        data_dir = Path(path)
        path_to_tree = data_dir.joinpath('tree_{}.pkl'.format(name))

        with open(path_to_tree, 'rb') as f:
            self.tree = pickle.load(f)

        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

    def determine_path_to_root(self, nodes):

        predecessors = list(self.tree.predecessors(nodes[-1]))
        predecessor = [k for k in predecessors][0]

        if predecessor == self.root:
            nodes.reverse()
            return nodes
        nodes.append(predecessor)
        return self.determine_path_to_root(nodes)

    def get_all_nodes_per_lvl(self, level):

        successors = list(self.tree.successors(self.root))
        while level > 0:
            next_lvl_succesors = []
            for successor in successors:
                next_lvl_succesors.extend(list(self.tree.successors(successor)))
            successors = next_lvl_succesors
            level -= 1

        return successors

    def normalize_path_from_root_per_level(self, path):
        """Normalize label values per level"""
        
        normalized_path = []
        for i in range(len(path)):
            counter = 0
            nodes_per_lvl = self.get_all_nodes_per_lvl(i)
            for node in list(nodes_per_lvl):
                counter += 1
                if node == path[i]:
                    normalized_path.append(counter)
                    break

        assert (len(path) == len(normalized_path))
        return normalized_path

    def get_sorted_leaf_nodes(self):
        leaf_nodes = []
        successors = [node for node in self.tree.successors(self.root)]
        while len(successors) > 0:
            successor = successors.pop()
            new_successors = [node for node in self.tree.successors(successor)]
            if len(new_successors) > 0:
                successors.extend(new_successors)
            else:
                leaf_nodes.append(successor)

        return leaf_nodes

    def get_number_of_nodes_lvl(self, max_hierarchy_level):
        num_labels_per_level = {}
        for i in range(max_hierarchy_level):
            nodes_per_lvl = [node for node in self.get_all_nodes_per_lvl(i)]
            num_labels_per_level[i] = len(nodes_per_lvl) + 1 # Plus 1 for ooc

        return num_labels_per_level

    def encode_node(self, name):
        decoder = dict(self.tree.nodes(data="attribute"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        return encoder[name]

    def encoding(self, only_decoder=False):
        """Builds encoder and decoder based on tree; Seperates labels between hierarchy levels {lvl > name > key}"""
        all_nodes = dict(self.tree.nodes())
        normalized_encoder = dict()

        decoder = dict()

        #decodes from orig. key to name
        for lvl in range(4):
            lvl_nodes = dict()
            for i in all_nodes.items():
                if i[1]['lvl'] == lvl:
                    lvl_nodes[i[0]] = {'name': i[1]['attribute'], 'predecessor': i[1]['predecessor']}
            decoder[lvl] = lvl_nodes

        #encodes from name to orig. key
        encoder = dict(
            (node[1]['lvl'], dict([(value[1]['attribute'], value[0]) for value in all_nodes.items() if value[1]['lvl'] == node[1]['lvl']]))
             for node in all_nodes.items())

        leaf_nodes = [key for key, val in decoder[3].items()]
        paths = [self.determine_path_to_root([node]) for node in leaf_nodes]
        normalized_paths = [self.normalize_path_from_root_per_level(path) for path in paths]

        # adding derived keys to decoder
        orig_key = dict([(i, [key[i] for key in paths]) for i in range(3)])
        derived_key = dict([(i, [key[i] for key in normalized_paths]) for i in range(3)])
        
        decoder[0][0] = {'name': 'Root', 'predecessor': None, 'derived_key': 0}
        for i in range(len(orig_key)):
            for key, d_key in zip(orig_key[i], derived_key[i]):
                for node in decoder[i+1].items():
                    if node[0] == key:
                        decoder[i+1][key] = {'name': node[1]['name'], 'predecessor': node[1]['predecessor'], 'derived_key': d_key}
                        break
    
        #for original labels only! encodes labels to normalized path
        for path, normalized_path in zip(paths, normalized_paths):
            normalized_encoder[tuple(path)] = {'derived_path': normalized_path}

        #decodes from derived key to original key
        normalized_decoder = dict([ (node[0], dict([ (value[1]['derived_key'], value[0] ) for value in node[1].items() ])) for node in decoder.items() ] )

        number_of_labels = len(leaf_nodes) + 1
        num_labels_per_lvl = {0: len(decoder[1])+1, 1: len(decoder[2])+1, 2: len(decoder[3])+1 }

        if only_decoder:
            return normalized_decoder

        return encoder, decoder, normalized_encoder, normalized_decoder, number_of_labels, num_labels_per_lvl