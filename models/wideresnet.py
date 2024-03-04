import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.configurable import configurable

from models.build import MODELS_REGISTRY


grads = {'cuda0': [], 'cuda1': [], 'cuda2': [], 'cuda3': []}

def save_grad(name):
    def hook(grad):
        grads[name].append(grad)
    return hook


def ini_noise(x):
    
    size = x.shape
    zero = torch.cuda.FloatTensor(size, device=x.device).fill_(0) if torch.cuda.is_available() else torch.FloatTensor(size)
    inoise = nn.Parameter(zero)
    #one = torch.cuda.FloatTensor(size, device=x.device).fill_(1) if torch.cuda.is_available() else torch.FloatTensor(size)
    #inoise = nn.Parameter(one)
    #inoise=nn.Parameter(torch.zeros(size, dtype=torch.float)).to(x.device)
     #nn.Parameter(torch.ones(size, dtype=torch.float))
    x = torch.add(x, inoise)#= x+inoise
    #x=torch.mul(x,inoise)
    keyname = str(inoise.device).replace(":", "")
    inoise.register_hook(save_grad(keyname))
    return x
    """
    keyname = str(x.device).replace(":", "")
    x.register_hook(save_grad(keyname))
    """

def gen_noise(grad_eps, grad_n):
    
    device = grad_n.device
    size = grad_n.shape

    abs = torch.abs(grad_n).to(device)
    #print(size[-1])

    tmax,i = torch.kthvalue(abs,int(size[-1]*0.1)+1, dim=-1, keepdim=True)#torch.topk(t, 3)#t.max(-1, keepdim=True)[0]
    mask = abs.gt(tmax).to(device) 
    #grad_norm = grad_n.norm(p=2).to(device)
    
    #scale = (grad_eps- grad_norm)/ (grad_norm + 1e-7)
    #grad_norm = grad_n.norm(p=2).to(device)
    #scale = grad_eps / (grad_norm + 1e-7)
    #noise = grad_n * scale
    noise = torch.cuda.FloatTensor(size, device=device) if torch.cuda.is_available() else torch.FloatTensor(size)
    torch.nn.functional.normalize(grad_n, p=2.0, dim=1, eps=1e-12, out=noise)
    dims = size[1]
    noise=grad_eps*(noise*mask)#/dims) #torch.mul(grad_eps, noise)
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



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.noiseset = None
        self.grad_eps = 1.0

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
    def set_blocknoise(self, noiseset='initial'):
        self.noiseset = noiseset

    def set_blockeps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            if self.training and self.noiseset ==None:
                size = x.shape
                noise = torch.cuda.FloatTensor(size, device=x.device) if torch.cuda.is_available() else torch.FloatTensor(size)
                mean = 0.0
                std = 0.1
                torch.normal(mean, std, size, out=noise)
                x = x+noise

            if self.training and self.noiseset == 'initial':
                x = ini_noise(x)

            elif self.training and self.noiseset == 'addnoise':
                x = ini_noise(x)
                keyname = str(x.device).replace(":", "")
                if grads[keyname]!= []:
                    grad_n = grads[keyname].pop()
                    #print(grad_n)
                    noise = gen_noise(self.grad_eps, grad_n)
                    #out = out+noise
                    x.add_(noise)
                    #out.mul_(1.0+noise)
        else:
            out = self.relu1(self.bn1(x))
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
                    out.add_(noise)
                    #out.mul_(1.0+noise)
        

        
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

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
                out.add_(noise)
                #out.mul_(1.0+noise)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.noiseset = None
        self.grad_eps = 1.0

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def set_eps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

        for i in self.layer:
            i.set_blockeps(grad_eps)


    def set_n(self, noiseset='initial'):
        self.noiseset = noiseset

        for i in self.layer:
            i.set_blocknoise(noiseset)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.noiseset = None
        self.grad_eps = 1.0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def set_gradeps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

        self.block1.set_eps(grad_eps)
        self.block2.set_eps(grad_eps)
        self.block3.set_eps(grad_eps)


    def set_noise(self, noiseset='initial'):
        self.noiseset = noiseset

        self.block1.set_n(noiseset)
        self.block2.set_n(noiseset)
        self.block3.set_n(noiseset)


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        #######################################
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
                out.add_(noise)
                #out.mul_(1.0+noise)
        
        #######################################


        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def _cfg_to_resnet(args):
    return {
        "num_classes": args.n_classes,
    }


@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def wideresnet28x10(num_classes=10):
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10)


@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def wideresnet34x10(num_classes=10):
    return WideResNet(depth=34, num_classes=num_classes, widen_factor=10)


