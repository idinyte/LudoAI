import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        #self.fc5 = nn.Linear(128, 128)
        #self.fc6 = nn.Linear(128, 128)
        #self.fc7 = nn.Linear(128, 128)
        #self.fc8 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        #x = torch.nn.functional.relu(self.fc5(x))
        #x = torch.nn.functional.relu(self.fc6(x))
        #x = torch.nn.functional.relu(self.fc7(x))
        #x = torch.nn.functional.relu(self.fc8(x))
        x = self.fc5(x)

        return x