import torch
from torchviz import make_dot
import torch.nn as nn
import torch.nn.functional as F


# Load your PyTorch model
class SignNet(nn.Module):
  def __init__(self):
    super(SignNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5) # (224 - 5 + 2*(0)) / 1 + 1 =220 -> 6, 220, 220
    self.pool1 = nn.MaxPool2d(2,2) #220 / 2 = 110
    self.conv2 = nn.Conv2d(6, 16, 5)  # (110 - 5 + 2*(0)) / 1 + 1 = 106 -> 16, 106, 106
    self.pool2 = nn.MaxPool2d(2,2) #106 / 2 = 53 - > 16, 53, 53 
    self.fc1 = nn.Linear(16*53*53, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 11) # 11 classes

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = x.view(-1, 16*53*53) #Flatten
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return x


model = SignNet()
## Generate a visual representation of the model - Detailed:
# vis_plot = make_dot(model(torch.rand(1, 3, 224, 224)), params=dict(model.named_parameters()))
## Generate a visual representation of the model - Simplified:
vis_plot = make_dot(model(torch.rand(1, 3, 224, 224)), params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
# Save the visualization to a file
vis_plot.render("SignNet", format="jpg")