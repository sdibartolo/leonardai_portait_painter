try:
 import torch, torch.nn as nn
except Exception: torch=None; nn=None
if nn is not None:
 class StrokePolicy(nn.Module):
  def __init__(self,action_dim=14):
   super().__init__();self.net=nn.Sequential(nn.Conv2d(9,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Conv2d(64,96,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(96,128),nn.ReLU(),nn.Linear(128,action_dim))
  def forward(self,x):return self.net(x)
