# NN.py
import torch
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim=2):  # 更改output_dim为2，表示预测区间的上下界
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output_dim)  # 输出层现在有两个单元

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# import torch
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 256) # Increased complexity
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(256, output_dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.fc3(x)
#         return x