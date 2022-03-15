# %%
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from pathlib import Path
# %%
class CustomDS():
    def __init__(self, dataset):
        '''
        Expects Dataset object for dataset from huggingface transformer library.
        CustomDS class is responsible for saving the all possible paths for the given dataset.
        Also saves the encoders used to encode labels.

        Labels have to be normalized before feeding it to the classifier, because predictions for each level are seperate,
        therefore if hierarchy level 1 has N classes, all label for level 1 have to be integers between 0 and N-1.
        hierarchy level 2 has M classes, labels for level 2 need to be between 0 and M-1 and so on
        
        The predictions for each level will be the corresponding number. 
        To make sure that during that loss is calculated properly, one encoder for each level is used 
        and saved as a NumPy array file (serializable) that can be used troughout the project.
        '''
        self.dataset = dataset
        self.encoder = None
        self.path = Path(f"./data/encoding")
        self.path.mkdir(parents=True, exist_ok=True)

        self.save_paths()
        self.save_encoder()

    def save_paths(self):

        # used during compute loss to check hierarchy 
        all_paths = self.dataset['path_list']
        unique_paths = set(tuple(i) for i in all_paths)
        self.unique_paths = [list(i) for i in unique_paths]

        with open(self.path.joinpath('all_paths.pkl'), 'wb') as f:
            pickle.dump(unique_paths, f)

        

    def save_encoder(self):
        encoder = []
        for i in range(3):
            encoder.append(LabelEncoder().fit(self.dataset[f'lvl{i+1}']))
            np.save(self.path.joinpath(f'encoder_classes_lvl{i+1}'), encoder[i].classes_)
        self.encoder = encoder

    def get_encoded_dataset(self):
        transposed_labels = np.array(self.dataset['path_list']).transpose()
        new_labels = []
        for i in range(len(transposed_labels)):
           new_labels.append(self.encoder[i].transform(transposed_labels[i]))

        dataset = self.dataset.add_column('label', list(np.array(new_labels).transpose()))

        return dataset

# %%
path = Path(f"./data/encoding")
path.mkdir(parents=True, exist_ok=True)

def load_encoder():
    encoder = []
    for i in range(3):
        enc = LabelEncoder()
        enc.classes_ = np.load(path.joinpath(f'encoder_classes_lvl{i+1}.npy'))
        encoder.append(enc)
    return encoder

def load_paths():
    with open(path.joinpath('all_paths.pkl'), 'rb') as f:
        all_paths = pickle.load(f)

    return all_paths


# %%
#paths = [[level1, level2, level3] for level1 in my_dict.keys() for level2 in my_dict[level1].keys() for level3 in my_dict[level1][level2].keys()]
