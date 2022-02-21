import torch


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
