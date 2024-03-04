import time
from collections import defaultdict
from typing import Iterable

import torch
import torch.distributed as dist
from utils.dist import is_dist_avail_and_initialized

noiselist = []

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce
    else:
        return x

def grad_norm(param_groups):
    shared_device = param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
    return norm

def ini_mask(size, grad_n):
    device=grad_n.device
    fisher_value = torch.square(grad_n).data
    fisher_value_list = torch.flatten(fisher_value)
        
    keep_num =  int(len(fisher_value_list)*(1 - 0.5))
    _value, _index = torch.topk(fisher_value_list, keep_num)
        
    mask_list = torch.zeros_like(fisher_value_list)
    mask_list.scatter_(0, _index, torch.ones_like(_value))
    mask = mask_list.reshape(size).to(device)

    return mask

def train_one_epoch(gradeps,
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, log_freq, use_closure
):
    model.train()

    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        def closure():
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()

        """
        output = model(images)
        loss = criterion(output, targets)
        
        optimizer.zero_grad()
        loss.backward()
        """

        images.requires_grad = True
        
        
        grad_eps=gradeps#0.01
        """
        if epoch <170:
            grad_eps=gradeps
        #elif epoch <150 and epoch>=100:
            #grad_eps=0.075
        elif epoch <200 and epoch>=170:
            grad_eps=0.08 #170 96.64 160 96.41
        """
        
        """
        elif epoch <40 and epoch >=20:
            grad_eps=gradeps+0.0125
            print(grad_eps)
        elif epoch <60 and epoch >=40:
            grad_eps=gradeps+0.025
        elif epoch <80 and epoch >=60:
            grad_eps=gradeps+0.0375
        elif epoch <100 and epoch >=80:
            grad_eps=gradeps+0.05
        elif epoch <120 and epoch >=100:
            grad_eps=gradeps+0.0625
        elif epoch <140 and epoch >=120:
            grad_eps=gradeps+0.075
        elif epoch <160 and epoch >=140:
            grad_eps=gradeps+0.0875
        elif epoch <180 and epoch >=160:
            grad_eps=gradeps+0.1
        elif epoch <200 and epoch >=180:
            grad_eps=gradeps+0.1125
        """

        model.module.set_gradeps(grad_eps)
        #model.module.set_noise("initial")
        model.module.set_noise("addnoise")

        output = model(images)
        loss = criterion(output, targets)
        
        optimizer.zero_grad()
        loss.backward()

        ######################################
        """
        noiselist=[]
        param_groups = optimizer.param_groups
        grad_n = grad_norm(param_groups)
        for group in param_groups:
            scale = 0.1 / (grad_n + 1e-7)
            for p in group["params"]:
                if p.requires_grad == True:
                    if p.grad is None: continue
                    #mask=ini_mask(p.grad.shape, p.grad)
                    inoise = p.grad * scale#*mask
                    noiselist.append(inoise)
                    p.data.add_(inoise)
        """
        ##############################################
        """
        if epoch==0 and batch_idx==0:
            output = model(images)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            
            param_groups = optimizer.param_groups
            for group in param_groups:
                for p in group["params"]:
                    if p.requires_grad == True:
                        size = p.shape
                        device = p.device
                        if p.grad is None: continue
                        noise = torch.cuda.FloatTensor(size, device=device) if torch.cuda.is_available() else torch.FloatTensor(size)
                        torch.nn.functional.normalize(p.grad, p=2.0, dim=1, eps=1e-12, out=noise)
                        noiselist.append(noise)
        else:
            param_groups = optimizer.param_groups
            for group in param_groups:
                for p in group["params"]:
                    if p.requires_grad == True:
                        if p.grad is None: continue
                        size = p.shape
                        device = p.device
                        wnoise = noiselist.pop(0)
                        p.data.mul_(1+wnoise)
                        
            
            output = model(images)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            
            param_groups = optimizer.param_groups
            for group in param_groups:
                for p in group["params"]:
                    if p.requires_grad == True:
                        if p.grad is None: continue
                        size = p.shape
                        device = p.device
                        noise = torch.cuda.FloatTensor(size, device=device) if torch.cuda.is_available() else torch.FloatTensor(size)
                        torch.nn.functional.normalize(p.grad, p=2.0, dim=1, eps=1e-12, out=noise)
                        noiselist.append(noise)
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
        grads[name] = param.grad
        """
        ##########################################

        if use_closure: 
            optimizer.step(
                closure, 
                epoch=epoch,
                step=epoch*len(train_loader)+batch_idx,
                batch_idx=batch_idx,
                model=model, 
                train_data=train_loader.dataset, 
                logger=logger,
            ) 
        else: 
            optimizer.step()
        
        ##################################
        """
        param_groups = optimizer.param_groups
        i=0
        for group in param_groups:
            for p in group["params"]:
                if p.requires_grad == True:
                    if p.grad is None: continue
                    inoise = noiselist[i]
                    i=i+1
                    p.data.sub_(inoise)
        """

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        
        _memory.update_meter('train_loss', loss.item(), n=batch_num)
        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        
        msg = ' '.join([
            'Epoch: {epoch}',
            '[{batch_id}/{batch_len}]',
            'lr:{lr:.6f}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Time:{batch_time:.3f}s'])
        if batch_idx % log_freq == 0:
            logger.log(
                msg.format(
                    epoch=epoch, 
                    batch_id=batch_idx, batch_len = len(train_loader),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=_memory.meters["train_loss"].global_avg,
                    train_acc1=_memory.meters["train_acc1"].global_avg,
                    train_acc5=_memory.meters["train_acc5"].global_avg,
                    batch_time=time.time() - batch_start,
                )
            )
    _memory.synchronize_between_processes()
    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: Iterable,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    _memory = MetricLogger()
    _memory.add_meter('test_loss', Metric())
    _memory.add_meter('test_acc1', Metric())
    _memory.add_meter('test_acc5', Metric())

    for images, targets in val_loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        batch_num = images.shape[0]
        _memory.update_meter('test_loss', loss.item(), n=batch_num)
        _memory.update_meter('test_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('test_acc5', acc5.item(), n=batch_num)
    _memory.synchronize_between_processes()
    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }

def accuracy(output, targets, topk=(1,)):
    # output: [b, n]
    # targets: [b]
    batch_size, n_classes = output.size()
    maxk = min(max(topk), n_classes)
    _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t() # pred: [b, maxk] -> [maxk, b]
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred)) # targets: [b] -> [1, b] -> [maxk, b]; correct(bool): [maxk, b]
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class Metric:
    def __init__(self) -> None:
        self.value = 0
        self.num = 0
    
    def update(self, value, n=1):
        self.num += n
        self.value += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        
        t = torch.tensor([self.num, self.value], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
    
        self.num = int(t[0])
        self.value = t[1]
    
    @property
    def global_avg(self):
        return self.value / self.num

class MetricLogger:
    def __init__(self) -> None:
        self.meters = defaultdict(Metric)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def update_meter(self, name, value, n):
        self.meters[name].update(value, n)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
