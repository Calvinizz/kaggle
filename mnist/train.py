from conv import ConvNet
import torch
from os.path import join
from torch.utils.data import DataLoader
from data_loader import load_data_torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import numpy as np  

max_epochs = 10
learning_rate = 1e-1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
mode_dir = './model'


num_input = 784
num_class = 10
dropout = 0.75
momentum = 0.9 # SGD
display_step = 100  

#load data
train_set,test_set=load_data_torch()
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)

convNet = ConvNet()
convNet.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(convNet.parameters(), lr=learning_rate, momentum=momentum)

def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("acc(\%)", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost)
    plt.plot(iters, accs,color='green',label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()

def showTestResult(test_data):
    examples = enumerate(test_data)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = convNet(example_data)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def train():
    global convNet
    
    if os.path.exists(mode_dir):
        print("Use pre_mode")
        convNet.load_state_dict(torch.load(os.path.join(mode_dir,'model.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(mode_dir,'optimizer.pth')))
    else:
        os.makedirs(os.path.join(os.getcwd(),mode_dir))
        
    train_losses=[]
    train_acces=[]

    for epoch in range(max_epochs):
        convNet = convNet.train()
        for step, (img,label) in enumerate(train_data):
            img = Variable(img)
            img = img.to(device)
            label = Variable(label)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = convNet(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step() # 更新网络参数，执行一次优化步骤

            
            _,pred = torch.max(output.data, 1)
            num_correct = (pred == label).sum().item()
            train_acc = num_correct / img.shape[0]
            
            train_loss = loss.item()
            train_losses.append(train_loss )
            train_acces.append(train_acc*100 )

            if(step+1)%display_step ==0 or (step+1) == len(train_data):
                print('Epoch:{} [{}/{}], Loss:{:.4f}, Acc:{}'.format(epoch,step+1,len(train_data),train_loss,train_acc))

             # Save model
        torch.save(convNet.state_dict(), os.path.join(mode_dir,'model.pth'))
        torch.save(optimizer.state_dict(),os.path.join(mode_dir,'optimizer.pth'))

        print('Epoch {}: Train Loss: {} Train  Accuracy: {} '.format(epoch + 1, np.mean(train_losses), np.mean(train_acces)))
        # Set up validation set
        print("############## val this epoch ##############")
        test()
        
    # Draw loss and acc charts
    draw_train_process('training', range(len(train_losses)), train_losses, train_acces, 'training loss', 'training acc')

def test():
    convNet.eval()
    test_loss=[]
    test_acc=[]

    with torch.no_grad(): # The network does not update the gradient during evaluation
        for i,(img,label) in enumerate(test_data):
            output=convNet(img)
            test_loss.append(loss_fn(output, label))
            _,test_pred=torch.max(output.data,1) # test_pred = output.data.max(1)[1]
            num_correct = (test_pred == label).sum().item()
            test_acc.append(num_correct / img.shape[0])
            # test_acc += label.size(0)
            # test_acc.append ((test_pred == label).sum().item()) #test_acc.append(test_pred.eq(label.data.view_as(test_pred)).sum())

        print("avg loss：{}, avg acc：{}".format(np.mean(test_loss),np.mean(test_acc)))

    #Draw loss and acc charts
    draw_train_process("Test", range(len(test_loss)), test_loss, test_acc, "testing loss", "testing acc")
    showTestResult(test_data)

if __name__=="__main__":
    train()
