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
cifar_train_set = torchvision.datasets.CIFAR10('./data2', train=True, download=True,transform=transform)
cifar_test_set = torchvision.datasets.CIFAR10('./data2', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(cifar_train_set ,
                                          batch_size=50,
                                          shuffle=True
                                         )
test_loader = torch.utils.data.DataLoader(cifar_test_set ,
                                          batch_size=50,
                                          shuffle=True
                                         )
# print(len(train_loader.dataset))
# print(len(test_loader.dataset))
# print(len(test_loader))

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3)
		self.conv2 = nn.Conv2d(32, 32, 3)
		self.fc1 = nn.Linear(14*14*32, 512)
		self.fc2 = nn.Linear(512, 10)
		self.dp1 = nn.Dropout(0.2)
		self.dp2 = nn.Dropout(0.5)

	def forward(self, x):
		# x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.relu(self.conv1(x))
		# x = F.embedding(x, weight= self.weight1(x), max_norm=3)
		# clipping_value = 1 # arbitrary value of your choosing
		nn.utils.clip_grad_norm(model.parameters(), 3)
		x = self.dp1(x)
		x = self.conv2(x)
		# x = F.embedding(x, weight = self.weight1(x), max_norm=3)
		nn.utils.clip_grad_norm(model.parameters(), 3)
		x = F.max_pool2d(F.relu(x), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = self.dp2(x)
		x = F.softmax(self.fc2(x),dim=0)
		return x

	# def weight1(self, x):
	# 	# print(1)
	# 	name, param = model.named_parameters(x)
	# 	print(param.data.shape)
	# 	return param.data

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


model = Net()
# print(model.weight1)

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
        # train_data, train_target = train_data, train_target
        # print(train_data.shape,train_target.shape)
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
    train_loss /= len(train_loader.dataset)
    valid_loss /= len(test_loader.dataset)
    if epoch <= 3 or epoch % 10 == 0:
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
test_loss /= len(test_loader.dataset)
accuracy = correct / len(test_loader.dataset)
print(f'Test loss: {test_loss}.. Test Accuracy(%): {accuracy}')     
