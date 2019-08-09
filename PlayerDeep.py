from Direction import Direction
import torch
import numpy as np

hidden_layer_size = 200
lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Brain(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Brain, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        # how to init? xavier?
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        a1 = self.linear1(x).clamp(min=0)
        z2 = self.linear2(a1)
        a2 = torch.sigmoid(z2)
        return a2


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

# from Karpathy's post, discounting rewards function

def discount_rewards(r):
    gamma = 0.99
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # this function may run once every several episodes, each episode ends with a reward != 0
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def normalize_rewards(r):
    epr = np.vstack(r)

    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    return discounted_epr


class PlayerDeep:
    EPISODE_LEN = 21

    def __init__(self):
        self.model = Brain(80 * 80, hidden_layer_size, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_computator = torch.nn.BCELoss(reduction='none')
        self.episode_len = self.EPISODE_LEN
        self.prev_x = None
        self.episode_number = 0

        self.batch_size = 10  # every how many episodes to do a param update? (episode is a single game)

        self.fake_labels = []
        self.p_ups = []
        self.rewards = []

        self.running_reward = None

        # Use this to count episodes
        self.wins = 0
        self.loses = 0

    def get_move(self, state):

        if len(self.p_ups) > 0:
            # save reward now because its after previous move.
            # only save after first move
            self.rewards.append(float(state.reward))

        if state.reward != 0:
            # won or lost the game
            self.episode_number += 1

            discounted_epr = normalize_rewards(self.rewards)

            Y_hat = torch.stack(self.p_ups).float().to(device).squeeze(1)
            Y = torch.tensor(self.fake_labels).float().to(device)

            # Accumulated Gradients
            losses = self.loss_computator(Y_hat, Y)

            # First create tensor from discounter_epr
            t_discounted_epr = torch.from_numpy(discounted_epr).squeeze(1).float().to(device)

            # Multiply each log(yi|xi) with Ai (Advantage)
            losses *= t_discounted_epr
            loss = torch.mean(losses)

            loss.backward(torch.tensor(1.0 / self.batch_size).to(device))

            if self.episode_number % self.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # reset episode state
            self.fake_labels, self.p_ups, self.rewards = [], [], []
            self.prev_x = None

        cur_x = prepro(state.image_data)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros_like(cur_x)
        self.prev_x = cur_x
        x = torch.from_numpy(x).float().to(device)
        p_up = self.model(x)  # probability of going up
        action = Direction.UP if np.random.uniform() < p_up.data[0] else Direction.DOWN
        y = 1.0 if action == Direction.UP else 0.0  # fake label

        self.p_ups.append(p_up)
        self.fake_labels.append(y)

        return action
