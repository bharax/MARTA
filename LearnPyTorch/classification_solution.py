import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times'] #['Times New Roman']

# ---
# Neural network implementation
# ---
class MLP(nn.Module):
    def __init__(self, in_size, hidden_units, out_size):
        super(MLP, self).__init__()
        
        self.hidden_units = hidden_units
        modules           = []
        
        # --- input layer
        modules.append(nn.Linear(in_size, hidden_units[0]))
        modules.append(nn.ReLU())
        
        # --- hidden layers
        for i in range(len(hidden_units)-1):
            modules.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            modules.append(nn.ReLU())
        
        # --- output layer
        modules.append(nn.Linear(hidden_units[-1], out_size))
        
        self.net = nn.Sequential(*modules)
        
    # --- forward pass
    def forward(self, x):
        out = self.net(x)
        return out

# ---
# Database
# ---
class MoonDatabase(torch.utils.data.IterableDataset):
    def __init__(self, X, Y, train=True):
        super(MoonDatabase).__init__()
        self.X = X
        self.Y = Y   
        
    def __iter__(self):
        return iter(torch.cat((self.X, self.Y), 1))

# ---
# Training loop
# ---
def training(hidden_layer,     # list of number of neurons/layers
             X_train, Y_train, # train data
             X_test, Y_test,   # test data
             nbEpochs,         # total number of epoch
             verbose=True,
             spacing=10
            ):
    
    # transfer to torch.Tensor for classification 
    if(not(torch.is_tensor(X_train))):
        X_train = torch.from_numpy(X_train).float()
    if(not(torch.is_tensor(X_test))):
        X_test  = torch.from_numpy(X_test).float()
    if(not(torch.is_tensor(Y_train))):
        Y_train = torch.from_numpy(Y_train).unsqueeze(1).float()
    if(not(torch.is_tensor(Y_test))):
        Y_test  = torch.from_numpy(Y_test).unsqueeze(1).float()

    # model, optimizer, loss function
    net       = MLP(X_train.shape[1], hidden_layer, 2)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # dataLoader
    trainData   = MoonDatabase(X_train, Y_train)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=32, num_workers=0, pin_memory=False)

    testData   = MoonDatabase(X_test, Y_test)
    testLoader = torch.utils.data.DataLoader(testData, batch_size=32, num_workers=0)

    # --- initialize array
    training_loss = np.zeros(nbEpochs)
    testing_loss  = np.zeros(nbEpochs)
    training_acc  = np.zeros(nbEpochs)
    testing_acc   = np.zeros(nbEpochs)
    
    # --- training loop
    for i in range(nbEpochs):
        net.train()
        
        # ----
        train_loss_epoch = 0
        train_acc_epoch  = 0
        nb               = 0
        for data in trainLoader:
            batch_sz = data.shape[0]
            x        = data[:,[0,1]]
            y        = data[:,[2]].squeeze().long()
            y_pred   = net(x)
            optimizer.zero_grad()
            batch_loss = criterion(y_pred,y)
            batch_loss.backward()
            optimizer.step()
            train_loss_epoch += batch_loss.detach()
            nb += 1
            # --- accuracy
            acc0_epoch = 1 - np.mean(np.abs(y.squeeze().long().detach().numpy() - np.where(y_pred[:,0].detach().numpy()<0, 1, 0)))
            acc1_epoch = 1 - np.mean(np.abs(y.squeeze().long().detach().numpy() - np.where(y_pred[:,1].detach().numpy()<0, 0, 1)))
            train_acc_epoch += (acc0_epoch+acc1_epoch)/2
            
        training_loss[i] = train_loss_epoch/nb
        training_acc[i]  = train_acc_epoch/nb
        
        # ----
        test_loss_epoch  = 0
        test_acc_epoch   = 0
        nb               = 0
        for data in testLoader:
            # loss
            batch_sz = data.shape[0]
            x        = data[:,[0,1]]
            y        = data[:,[2]].squeeze().long()
            y_pred   = net(x)
            batch_loss = criterion(y_pred,y)
            test_loss_epoch += batch_loss.detach()
            nb += 1
            
            # --- accuracy
            acc0_epoch = 1 - np.mean(np.abs(y.squeeze().long().detach().numpy() - np.where(y_pred[:,0].detach().numpy()<0, 1, 0)))
            acc1_epoch = 1 - np.mean(np.abs(y.squeeze().long().detach().numpy() - np.where(y_pred[:,1].detach().numpy()<0, 0, 1)))
            test_acc_epoch += (acc0_epoch+acc1_epoch)/2
            
        testing_loss[i] = test_loss_epoch/nb
        testing_acc[i]  = test_acc_epoch/nb
        # ----
        if(verbose and i%spacing==0):
            print("Epoch: %5d, train_loss=%.4e, test_loss=%.4e, train_acc=%.4e, test_acc=%.4e" %(i,training_loss[i],testing_loss[i],training_acc[i],testing_acc[i]))
        # ----
        
    return net, training_loss, testing_loss, training_acc, testing_acc

# ---
# Function to plot the decision boundary and data points of a model.
# Data points are colored based on their actual label.
# ---
def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - .5, X[:,0].max() + .5
    ymin, ymax = X[:,1].min() - .5, X[:,1].max() + .5
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    with torch.no_grad():
        labels = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))

    # Plot decision boundary in region of interest
    z = np.where(labels[:,0]<0,1,0).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model(torch.FloatTensor(X))
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), cmap=cmap, lw=0)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(15)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(15)

    ax.set_xlabel("$x$", fontsize=25)
    ax.set_ylabel("$y$", fontsize=25)
    
    return fig, ax

def plot_loss_accuracy(nbEpochs, X_test, Y_test, net,                       
                       training_loss, testing_loss,
                       training_acc, testing_acc):
    fig, ax = plt.subplots(1,2, constrained_layout=True)

    ax[0].loglog(np.arange(1,nbEpochs+1), training_loss, 'k.-', linewidth=2)
    ax[0].loglog(np.arange(1,nbEpochs+1), testing_loss, 'b.-', linewidth=2)

    ax[0].set_xlabel("Epoch", fontsize=25)
    ax[0].set_ylabel("Loss", fontsize=25)

    ax[1].semilogx(np.arange(1,nbEpochs+1), training_acc, 'k.-', linewidth=2)
    ax[1].semilogx(np.arange(1,nbEpochs+1), testing_acc, 'b.-', linewidth=2)


    ax[1].set_xlabel("Epoch", fontsize=25)
    ax[1].set_ylabel("Accuracy", fontsize=25)

    for i in range(2):
        ax[i].grid(True)

        for tick in ax[i].get_xticklabels():
            tick.set_fontsize(15)

        for tick in ax[i].get_yticklabels():
            tick.set_fontsize(15)

    fig.set_size_inches(14,4)

    fig, ax = plot_decision_boundary(X_test,Y_test, net, cmap = 'RdBu')
    fig.set_size_inches(7,4)
    
    plt.show()