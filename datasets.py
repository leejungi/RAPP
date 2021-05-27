import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_data, test_data, normal=[0],  train=True):
        super(Dataset, self).__init__()
        self.data = []
        self.targets = []
        #Train true 
        if train == True:
            train_data, train_targets = train_data.data, train_data.targets

            indices = np.where(np.isin(train_targets,normal))
            self.data = train_data[indices]
            self.targets = train_targets[indices]
        else:
            train_data, train_targets = train_data.data, train_data.targets
            test_data, test_targets = test_data.data, test_data.targets
            abnormal = [i for i in range(10)]   
            abnormal = np.where(~np.isin(abnormal,normal))
            
            self.data = test_data
            self.targets = test_targets
            
            indices = np.where(np.isin(train_targets,abnormal))
            self.data = torch.cat([self.data, train_data[indices]])
            self.targets = torch.cat([self.targets,train_targets[indices]]) 
            
            self.targets = np.where(np.isin(self.targets, abnormal),1,0)
            self.targets = torch.Tensor(self.targets)
            
                    

        self.data = self.data/255.0
        self.data = self.data.view(-1,28*28).float()
        
        
        # print(self.data.size(), self.targets.size())
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.targets[index]
        return data, label 
