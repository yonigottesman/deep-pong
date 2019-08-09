import gym
import numpy as np
import torch
import torch.nn.functional as F

class Brain(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Brain, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

        # add regularization?
        # add batch norm
        # try more layers
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        a1 = self.linear1(x).clamp(min=0)
        z2 = self.linear2(a1)
        a2 = torch.sigmoid(z2)
        return a2


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    gamma = 0.99
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


prev_x = None  # used in computing the difference frame
fake_lables, drs, p_ups = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
env = gym.make('Pong-v0')
observation = env.reset()
GAMMA = 0.99
# hyperparam
hidden_layer_size = 200
lr = 3e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 10
weight_decay = 0
model = Brain(80 * 80, hidden_layer_size, 1)
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_computator = torch.nn.BCELoss(reduction='none')
optimizer.zero_grad()

while True:
    # env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
    prev_x = cur_x

    # make tensor
    x = torch.from_numpy(x).float().to(device)
    aprob = model.forward(x)
    p_ups.append(aprob)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    y = 1.0 if action == 2 else 0.0  # a "fake label"
    fake_lables.append(y)

    observation, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished (player reached 21)
        episode_number += 1
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epr = np.vstack(drs)

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        Y_hat = torch.stack(p_ups).float().to(device).squeeze(1)
        Y = torch.tensor(fake_lables).float().to(device)

        # First create tensor from discounter_epr
        t_discounted_epr = torch.from_numpy(discounted_epr).squeeze(1).float().to(device)

        # losses = loss_computator(Y_hat, Y)

        # # Multiply each log(yi|xi) with Ai (Advantage)
        #losses *= t_discounted_epr
        # loss = torch.mean(losses)


        losses2 = (1-Y)*torch.log(1-Y_hat)+Y*torch.log(Y_hat)
        losses2 *= t_discounted_epr
        loss2 = torch.sum(losses2) / batch_size

        # loss = loss / batch_size # Normalize loss because of accumulated gradients
        loss2.backward()
        if episode_number % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode {} reward total was {}. running mean: {}'.format(episode_number,
                                                                                       reward_sum, running_reward))

        if episode_number % 100 == 0: torch.save(model, './model.torch')
        fake_lables, drs, p_ups = [], [], []  # reset array memory
        observation = env.reset()  # reset env
        prev_x = None
        reward_sum = 0

env.close()


# RMSprop 3e-3 628 - 16.16 mean


# try 1e-1 or 1e-2 as wd in adam
# add another layer to network
# conv network
