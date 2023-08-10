import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.optim import Optimizer
import time
from resnet import resnet20
import os


num_epochs = 30
batch_size = 400
learning_rate = 0.15
n_directions = 2

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ourMethod(Optimizer):
    def __init__(self, parameters, lr=1e-3, n_directions = 2, device = "cpu"):
        defaults = {"lr": lr, "device":device}
        super().__init__(parameters, defaults)
        self.state['step'] = 0
        self.n_directions = n_directions
        self.ro = 0.02
        self.device = device
        for p in self.get_params():
            p.update_value = 0
      
    
    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)
 
        
    @torch.no_grad()
    def update_subspace(self,old_subSpace,new_gradient):
        if self.state['step'] < self.n_directions:
            new_subSpace = torch.cat([old_subSpace,new_gradient],1)
        else:
            new_subSpace = torch.cat([old_subSpace[:,1:],new_gradient],1)
        return new_subSpace
        
       
    
    def Hessian_v(self,grad,vec,p):                
        Hv = torch.autograd.grad(
        grad,
        p,
        grad_outputs=vec,
        only_inputs=True,
        retain_graph=True)
        
        Hv_flaten = []
        for i in range(len(Hv)):
            Hv_flaten.append(Hv[i].reshape(-1))
        return torch.cat(Hv_flaten,0).view(-1,1)
    
    
    
    def Hessian_M(self,grad,matrix,p):
        H_M = []
        for i in range(matrix.shape[1]):
            H_M.append(self.Hessian_v(grad,matrix[:,i].view(-1,1),p))
        return torch.cat(H_M,1)  
        
        
    @torch.no_grad()    
    def correction_Hessian(self,H):
        eig, U = torch.linalg.eigh(H)
        alfa = 0
        if eig.min() < self.ro:
            alfa = self.ro - eig.min()            
        return U, eig, alfa           
        
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
                      
                

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.state['step'] == 0:
                    d_p = p.grad
                    with torch.no_grad():
                        p.add_(d_p, alpha=-group['lr'])
                    p.subSpace = p.grad.data.view(-1,1)
                else:
                    flat_grad = p.grad.view(-1,1)
                    p.subSpace = self.update_subspace(p.subSpace,flat_grad.data)
                    Q, _ = torch.linalg.qr(p.subSpace.data)
                    HQ = self.Hessian_M(flat_grad,Q,p)
                    U, eig, alfa = self.correction_Hessian(Q.T@HQ)
                    y = U@torch.diag(1/(eig+alfa))@U.T@(Q.T@flat_grad)
                    d_p = Q@y
                    p.update_value = d_p.view_as(p.data)
                    
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    p.add_(p.update_value, alpha=-group['lr'])
                        
                        
        
        self.state['step'] += 1        
        return 




def evaluate():
    evaluation = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        evaluation.append(acc)
        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
            evaluation.append(acc)
    return evaluation
    


def save_progress():
	evaluation = np.array(acc)
	with open('evaluation_ourOptimizer2dir015LR.npy', 'wb') as f:
		np.save(f, evaluation)

	losses_arr = np.array(losses)
	with open('losses_ourOptimizer2dir015LR.npy', 'wb') as f:
		np.save(f, losses_arr)
		
	torch.save(model.state_dict(), 'model_ownOpt2dir015LR.pt')


device = torch.device("cuda")
print(device)

def force_cudnn_initialization():
    s = 32
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))
force_cudnn_initialization()

model = resnet20()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = ourMethod(model.parameters(), lr=learning_rate, n_directions=n_directions, device=device)


n_total_steps = len(train_loader)
iteration = 0
losses = []
acc = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
       
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        if iteration == 0:
            print(f'Initial loss = {loss.item():.4f}\n\n')

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(create_graph=True)
        optimizer.step()

        losses.append([iteration,loss.item()])
        iteration += 1

        if (i+1) % 1 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    acc.append(evaluate())
    save_progress()

