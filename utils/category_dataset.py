import torch

class CategoryDatasetHierarchy(torch.utils.data.Dataset):
    def __init__(self, ds, encoder, decoder, normalized_encoder):
        # Preprocess encodings
        new_ds = {'input_ids': ds['input_ids'], 'attention_mask': ds['attention_mask']}
        self.encodings = new_ds
        # Preprocess labels
        self.labels = [self.encode_label(encoder, normalized_encoder, decoder, path) for path in ds['label']]


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(torch.int64) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(torch.int64)
        return item

    def __len__(self):
        return len(self.labels)

    def encode_label(self, encoder, normalized_encoder,  decoder, path):
        ''' encodes each path_list to original key (as noted in self.tree) and then normalizes it, i.e. [1,2,3] -> [1,1,1]
        '''
        path = [i.replace(' ', '_') for i in path]
        result = []
        for i in range(len(path)):

            if i+1 == len(path): # -> 3
                counter = []
                for item in decoder[i+1].items():
                    if item[1]['name'] == path[i]:
                        counter.append(item)
                if len(counter) > 1:
                    for item in counter:
                        if decoder[i][item[1]['predecessor']]['name'] == path[i-1]:
                            result.append(item[0])
                            break
                else:
                    result.append(counter[0][0])

            else:
                result.append(encoder[i+1][path[i]])

        # normalizes from orig. key to normalized version; remove this if you want to use the original keys
        result = normalized_encoder[tuple(result)]['derived_path']

        return result


class CategoryDatasetFlat(torch.utils.data.Dataset):
    def __init__(self, ds, encoder):
        # Preprocess encodings
        new_ds = {'input_ids': ds['input_ids'], 'attention_mask': ds['attention_mask']}
        self.encodings = new_ds
        # Preprocess labels
        self.labels = [encoder[x.replace(' ', '_')]['derived_key'] for x in ds['label']]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(torch.int64) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(torch.int64)
        return item

    def __len__(self):
        return len(self.labels)