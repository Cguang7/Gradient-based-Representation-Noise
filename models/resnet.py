'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.configurable import configurable

from models.build import MODELS_REGISTRY


grads = {'cuda0': [], 'cuda1': [], 'cuda2': [], 'cuda3': []}

def save_grad(name):
    def hook(grad):
        grads[name].append(grad)
    return hook


def ini_noise(x):
    
    size = x.shape
    #zero = torch.cuda.FloatTensor(size, device=x.device).fill_(0) if torch.cuda.is_available() else torch.FloatTensor(size)
    #inoise = nn.Parameter(zero)
    one = torch.cuda.FloatTensor(size, device=x.device).fill_(1) if torch.cuda.is_available() else torch.FloatTensor(size)
    inoise = nn.Parameter(one)
    #x = torch.add(x, inoise)#= x+inoise
    x=torch.mul(x,inoise)
    keyname = str(inoise.device).replace(":", "")
    inoise.register_hook(save_grad(keyname))
    return x#, inoise
    """
    keyname = str(x.device).replace(":", "")
    x.register_hook(save_grad(keyname))
    """

def gen_noise(grad_eps, grad_n):
    
    device = grad_n.device
    size = grad_n.shape

    #mask = torch.cuda.FloatTensor(size, device=device).fill_(0) if torch.cuda.is_available() else torch.FloatTensor(size)
    #mask.bernoulli_(0.9)
    #print(mask.device)

    abs = torch.abs(grad_n).to(device)
    #print(size[-1])

    tmax,i = torch.kthvalue(abs,int(size[-1]*0.5)+1, dim=1, keepdim=True)#torch.topk(t, 3)#t.max(-1, keepdim=True)[0] dim=-1
    mask = abs.gt(tmax).to(device) 


    """
    fisher_value = torch.abs(grad_n)
    fisher_value_list = torch.flatten(fisher_value)
    
    keep_num =  int(len(fisher_value_list)*(1 - 0.5))
    
    _value, _index = torch.topk(fisher_value_list, keep_num)
    
    
    mask_list = torch.zeros_like(fisher_value_list).to(device)#torch.cuda.FloatTensor(size, device=x.device).fill_(1) if torch.cuda.is_available() else torch.FloatTensor(size)
    mask_list.scatter_(0, _index, torch.ones_like(_value).to(device))
    mask = mask_list.reshape(size).to(device)
    """
    
    #grad_norm = grad_n.norm(p=2).to(device)
    
    #scale = (grad_eps- grad_norm)/ (grad_norm + 1e-7)
    #grad_norm = grad_n.norm(p=2).to(device)
    #scale = grad_eps / (grad_norm + 1e-7)
    #noise = grad_n * scale
    one = torch.cuda.FloatTensor(size, device=device).fill_(1) if torch.cuda.is_available() else torch.FloatTensor(size)
    noise = torch.cuda.FloatTensor(size, device=device) if torch.cuda.is_available() else torch.FloatTensor(size)
    torch.nn.functional.normalize(grad_n*mask, p=2.0, dim=1, eps=1e-12, out=noise)
    dims = size[1]
    noise=one+grad_eps*(noise)#*mask)#/dims) #torch.mul(grad_eps, noise)
    #print(grad_eps)
    """
    
    ######

    grad_abs = torch.abs(grad_n)
    g_max=torch.norm(grad_abs,p=2)
    #std = (grad_eps/g_max)*grad_abs #(0.1/g_max)*grad_abs
    #print(grad_eps)
    std = (grad_eps/g_max)*(g_max-grad_abs)

    device = grad_n.device
    size = grad_n.shape
    noise = torch.cuda.FloatTensor(size, device=device) if torch.cuda.is_available() else torch.FloatTensor(size)
    mean = 0.0

    torch.normal(mean, std, out=noise)
    """

    return noise


class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True): #noisesize, noise_mode,
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
       
        self.noiseset = None
        self.grad_eps =1.0

    def set_bnnoise(self, noiseset='initial'):
        self.noiseset = noiseset

    def set_bneps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                #self.running_mean = exponential_average_factor * mean\
                    #+ (1 - exponential_average_factor) * self.running_mean
                self.running_mean.mul_(1 - exponential_average_factor).add_(exponential_average_factor * mean)
                # update running_var with unbiased var
                #self.running_var = exponential_average_factor * var * n / (n - 1)\
                    #+ (1 - exponential_average_factor) * self.running_var
                self.running_var.mul_(1 - exponential_average_factor).add_(exponential_average_factor * var * n / (n - 1))
        else:
            mean = self.running_mean
            var = self.running_var
        
        
        if self.training and self.noiseset == 'initial':
            input=ini_noise(input)

        elif self.training and self.noiseset == 'addnoise':
            keyname = str(input.device).replace(":", "")

            if grads[keyname]!= []:
                grad_n = grads[keyname].pop()
                noise = gen_noise(self.grad_eps, grad_n)
                input = input+noise
        
        
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        
        if self.affine:
            #with torch.no_grad():
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            

        return input



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.noiseset = None
        self.grad_eps = 1.0

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.bn1 = MyBatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.bn2 = MyBatchNorm2d(planes)#

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def set_blocknoise(self, noiseset='initial'):
        self.noiseset = noiseset

    def set_blockeps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

    def forward(self, x):
        out = self.conv1(x)
        """
        if self.training:
            self.bn1.set_bnnoise(self.noiseset)
            self.bn1.set_bneps(self.grad_eps)
        """
        out = self.bn1(out)
        out = F.relu(out)
        
        if self.training and self.noiseset ==None:
            size = out.shape
            noise = torch.cuda.FloatTensor(size, device=out.device) if torch.cuda.is_available() else torch.FloatTensor(size)
            mean = 0.0
            std = 0.1
            torch.normal(mean, std, size, out=noise)
            out = out+noise

        if self.training and self.noiseset == 'initial':
            out = ini_noise(out)

        elif self.training and self.noiseset == 'addnoise':
            out = ini_noise(out)
            keyname = str(out.device).replace(":", "")
            if grads[keyname]!= []:
                grad_n = grads[keyname].pop()
                #print(grad_n)
                noise = gen_noise(self.grad_eps, grad_n)
                #out = out+noise
                #out.add_(noise)
                out.mul_(noise)
        
        
        


        out = self.conv2(out)
        """
        if self.training:
            self.bn2.set_bnnoise(self.noiseset)
            self.bn2.set_bneps(self.grad_eps)
        """
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        

        
        if self.training and self.noiseset ==None:
            size = out.shape
            noise = torch.cuda.FloatTensor(size, device=out.device) if torch.cuda.is_available() else torch.FloatTensor(size)
            mean = 0.0
            std = 0.1
            torch.normal(mean, std, size, out=noise)
            out = out+noise

        if self.training and self.noiseset == 'initial':
            out = ini_noise(out)

        elif self.training and self.noiseset == 'addnoise':
            out = ini_noise(out)
            keyname = str(out.device).replace(":", "")
            if grads[keyname]!= []:
                grad_n = grads[keyname].pop()
                noise = gen_noise(self.grad_eps, grad_n)
                #out = out+noise
                #out.add_(noise)
                out.mul_(noise)
        
        
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.noiseset = None
        self.grad_eps = 1.0


        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.bn1 = MyBatchNorm2d(64) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)##########
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_gradeps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

        for i in self.layer1:
            i.set_blockeps(grad_eps)

        for i in self.layer2:
            i.set_blockeps(grad_eps)

        for i in self.layer3:
            i.set_blockeps(grad_eps)

        for i in self.layer4:
            i.set_blockeps(grad_eps)

    def set_noise(self, noiseset='initial'):
        self.noiseset = noiseset

        for i in self.layer1:
            i.set_blocknoise(noiseset)

        for i in self.layer2:
            i.set_blocknoise(noiseset)

        for i in self.layer3:
            i.set_blocknoise(noiseset)

        for i in self.layer4:
            i.set_blocknoise(noiseset)

    def forward(self, x,y=None):
        out = self.conv1(x)
        """
        if self.training:
            self.bn1.set_bnnoise(self.noiseset)
            self.bn1.set_bneps(self.grad_eps)
        """
        out = self.bn1(out)
        out = F.relu(out)
        
        if self.training and self.noiseset ==None:
            size = out.shape
            noise = torch.cuda.FloatTensor(size, device=out.device) if torch.cuda.is_available() else torch.FloatTensor(size)
            mean = 0.0
            std = 0.1
            torch.normal(mean, std, size, out=noise)
            out = out+noise

        if self.training and self.noiseset == 'initial':
            out = ini_noise(out)
            
        elif self.training and self.noiseset == 'addnoise':
            out = ini_noise(out)
            keyname = str(out.device).replace(":", "")
            if grads[keyname]!= []:
                grad_n = grads[keyname].pop()
                noise = gen_noise(self.grad_eps, grad_n)
                #out = out+noise
                #out.add_(noise)
                out.mul_(noise)
        
        #out = self.maxpool(out) ################
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if y is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(out,y.view(-1))
            return loss
        else:
            return out



def _cfg_to_resnet(args):
    return {
        "num_classes": args.n_classes,
        "imagenet": args.dataset[:8] == 'ImageNet'
    }

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet18(num_classes=10, imagenet=False):
    if imagenet: return torchvision.models.resnet18(pretrained=False)
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet34(num_classes=10, imagenet=False):
    if imagenet: return torchvision.models.resnet34(pretrained=False)
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet50(num_classes=10, imagenet=False):
    #if imagenet: return torchvision.models.resnet50(pretrained=False)
    print(num_classes)
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet101(num_classes=10, imagenet=False):
    if imagenet: return torchvision.models.resnet101(pretrained=False)
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes = num_classes)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet152(num_classes=10, imagenet=False):
    if imagenet: return torchvision.models.resnet152(pretrained=False)
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes = num_classes)


def test():
    net = resnet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
