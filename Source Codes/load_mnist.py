import torch

from torchvision import datasets
from torchvision import transforms

def non_iid_split(dataset, nodes, samples_per_node, batch_size, shuffle, shuffle_digits=False):
    
    # assert(nodes>0 and nodes<=10)

    digits = torch.arange(10) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))

    
    digits_split=list()
    i=0
    
    # Assigning two digits for each client  
    for n in range(nodes, 0, -1):
        delta=int((10-i)/n)
        digits_split.append(digits[i:i+delta])
        i+=delta

    # loading and shuffling samples from the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=nodes * samples_per_node, shuffle=shuffle)
    
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = next(dataiter)
    

    data_splitted=list()
    for i in range(nodes):
        idx = torch.stack([train_y == labels_train_mnist for train_y in digits_split[i]]).sum(0).bool() 
        
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size, shuffle=shuffle))

    return data_splitted
    
    
def get_MNIST (type="non_iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True):
    train_data = datasets.MNIST(root="./data", train=True, download=True,transform=transforms.ToTensor())

    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    
    train = non_iid_split(train_data, n_clients, n_samples_train, batch_size, shuffle)
    test = non_iid_split(test_data, n_clients, n_samples_test, batch_size, shuffle)
    
    return train, test
