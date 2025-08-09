# -*- coding:utf-8 -*-
import os
import gzip
import numpy as np
from torchvision import transforms
from torchvision.datasets import mnist
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def load_data_minist(data_folder):
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

def load_data(data_folder,data_name,label_name):
    """
    :param data_folder: 文件目录
    :param data_name: 数据文件名
    :param label_name: 标签数据文件名
    :return:
    """
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        y = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y), 28, 28)

    return (x, y)

class DealDataSet():
    """
    """
    def __init__(self,folder,data_name,label_name,transform):
        (X_set,Y_set)=load_data(folder,data_name,label_name)# 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.X_set=X_set
        self.Y_set=Y_set
        self.transform=transform

    def __getitem__(self,index):
        img,target=self.X_set[index],int(self.Y_set[index])
        if self.transform is not None:
            img=self.transform(img)
        return img,target

    def __len__(self):
        return len(self.X_set)

def load_data_torch():
    # 预处理=>将各种预处理组合在一起
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    # 使用内置函数下载mnist数据集
    train_set = mnist.MNIST('./MNIST_data', train=True, transform=data_tf, download=True)
    test_set = mnist.MNIST('./MNIST_data', train=False, transform=data_tf, download=True)

    return (train_set,test_set)

def show_example_img():
    train_set, test_set = load_data_torch()
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(test_set, batch_size=64, shuffle=False)
    examples = enumerate(test_data)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_example_targets():
    train_set, test_set = load_data_torch()
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(test_set, batch_size=64, shuffle=False)
    examples = enumerate(test_data)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)


if __name__=="__main__":
    show_example_img()
    show_example_targets()
