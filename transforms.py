from torch.utils.data import Dataset

class AddTransform(Dataset):
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        #print('pre x shape', x.shape)
        #print('pre x type', type(x))
        x = self.transform(x)
        #print('after x shape', x.shape)
        #print('after x type', type(x))
        
        return x, y