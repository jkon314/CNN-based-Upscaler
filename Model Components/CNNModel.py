import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a basic block (used in ResNet-18 and ResNet-34)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,device='cuda'):
        super(BasicBlock, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False,device='cuda')
        self.bn1 = nn.BatchNorm2d(out_channels,device='cuda')
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
        #                        stride=1, padding=1, bias=False,device='cuda')
        # self.bn2 = nn.BatchNorm2d(out_channels,device='cuda')
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out += identity
        # out = F.relu(out)
        return out


class CNNModel(nn.Module):
    def __init__(self, block, layers, outputRes=(135*240)):
        super(CNNModel, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False,device='cuda')
        self.bn1   = nn.BatchNorm2d(64,device='cuda')
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #fully connected layer should produce the final upscaled output as a flattened array
        self.fc = nn.Linear(256+90*160, outputRes,device='cuda') 

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False,device='cuda'),
                nn.BatchNorm2d(out_channels * block.expansion,device='cuda'),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        #copy input images as flattened array to provide context for FC layer
        
        x_copy = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.avgpool(x)
        
        x = self.layer2(x)

        

        x = self.avgpool(x)
        

        

        x = self.layer3(x)
        

        

        x = self.avgpool(x)


        
        x = torch.squeeze(x,(2,3))

        #concat the input image with the output of CNN to create a final output
        
        #go channel by channel to reduce memory usage???
      
        b = torch.cat((x,x_copy[:,0,:,:].reshape((x_copy.shape[0],x_copy.shape[2]*x_copy.shape[3]))),(1))
        g = torch.cat((x,x_copy[:,1,:,:].reshape((x_copy.shape[0],x_copy.shape[2]*x_copy.shape[3]))),(1))
        r = torch.cat((x,x_copy[:,2,:,:].reshape((x_copy.shape[0],x_copy.shape[2]*x_copy.shape[3]))),(1))
        
        
        r = self.fc(r)
        g = self.fc(g)
        b = self.fc(b)
        return b,g,r


