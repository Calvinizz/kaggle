from torch import nn
 
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 128, 4,4
        
        self.fc = nn.Sequential(
            # nn.Linear(in_features, out_features, bias=True) 创建一个线性变换：$$y = xA^T + b$$
            nn.Linear(128*4*4, 1024), # 128*4*4是一步步池化计算出来的
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True), # inplace参数决定是否创建新张量。false创建，true复用
            nn.Linear(128, 10)
        )
        
    def forward(self,x):
        '''
         假设batch_size = 32
         x.shape  # torch.Size([32, 128, 4, 4])
         执行展平操作
         x = x.view(x.size(0), -1)
         x.shape  # torch.Size([32, 2048])
         # 其中 2048 = 128 × 4 × 4
        '''
        
        # 这个x不是一个参数，是个张量或者什么
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x