from Direction import Direction
import torch
import numpy as np


hidden_layer_size = 200
lr = 1e-4


class Brain(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Brain, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        #how to init? xavier?

    def forward(self, x):
        a1 = self.linear1(x).clamp(min=0)
        z2 = self.linear2(a1)
        a2 = torch.sigmoid(z2)
        return a2

# from Karpathy's post, preprocessing function
def prepro(I):
    I = I[::4, ::4, 0]  # downsample by factor of 2.
    I[I != 0] = 1
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

# from Karpathy's post, discounting rewards function
def discount_rewards(r):
    pass

class PlayerDeep:
    def __init__(self):
        self.model = Brain(100*100, hidden_layer_size, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_move(self, state):
        a = prepro(state.image_data)
        return Direction.STAY
