import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
#import matplotlib.pyplot as plt
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# print(device)
mnist_train_set = torchvision.datasets.MNIST('./data', train=True, download=True,transform=transform)
mnist_test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train_set ,
                                          batch_size=64,
                                          shuffle=True
                                         )
test_loader = torch.utils.data.DataLoader(mnist_test_set ,
                                          batch_size=64,
                                          shuffle=True
                                         )
# print(len(test_loader),len(test_loader.dataset)/5)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(500, 10)
        nn.init.xavier_uniform(self.fc2.weight)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        
        # self.weights_init()
        x = F.relu(self.fc1(x))
        
        # self.weights_init()
        x = F.softmax(self.fc2(x),dim=0)
        #print(x.shape)
        #x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    



model = Net()
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 5

for epoch in range(n_epochs):
    ##################
    ### TRAIN LOOP ###
    ##################
    # set the model to train mode
    model.train()
    train_loss = 0
    for train_data, train_target in train_loader:
        train_data, train_target = train_data, train_target
        #print(train_data.type(),train_target.type())
        train_target_onehot = torch.zeros(train_target.shape[0], 10)
        train_target_onehot.scatter_(1, train_target.unsqueeze(1), 1.0)
        optimizer.zero_grad()
        output = model(train_data)
        criterion = nn.BCELoss()
        loss = criterion(output, train_target_onehot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    #######################
    ### VALIDATION LOOP ###
    #######################
    # set the model to eval mode
    model.eval()
    valid_loss = 0
    # turn off gradients for validation
    with torch.no_grad():
        for test_data, test_target in test_loader:
            test_data, test_target = test_data, test_target
            test_target_onehot = torch.zeros(test_target.shape[0], 10)
            test_target_onehot.scatter_(1, test_target.unsqueeze(1), 1.0)
            output = model(test_data)
            loss = criterion(output, test_target_onehot) 
            valid_loss += loss.item()
            
    #########################
    ## PRINT EPOCH RESULTS ##
    #########################
    train_loss /= len(train_loader)
    valid_loss /= len(test_loader)
    if epoch <= 3 or epoch % 1 == 0:
        print(f'Epoch: {epoch+1}/{n_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
#################
### TEST LOOP ###
#################
# set the model to eval mode
model.eval()
test_loss = 0
correct = 0


# turn off gradients for validation
with torch.no_grad():
    for test_data, test_target in test_loader:
        test_data, test_target = test_data, test_target
        #print(test_data.shape)
        test_target_onehot = torch.zeros(test_target.shape[0], 10)
        test_target_onehot.scatter_(1, test_target.unsqueeze(1), 1.0)
        # forward pass
        output = model(test_data)
        # validation batch loss
        loss = criterion(output, test_target_onehot) 
        # accumulate the valid_loss
        test_loss += loss.item()
        # calculate the accuracy
        predicted = output.max(dim=1).indices
        correct += (test_target == predicted).sum().item()

########################
## TEST RESULTS ##
########################
test_loss /= len(test_loader)
accuracy = correct / len(test_loader)
print(f'Test loss: {test_loss}.. Test Accuracy(%): {accuracy}')     