import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from collections import namedtuple
import random
import matplotlib.pyplot as plt
from matplotlib import style
from itertools import count
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Player:
    def __init__(self, x, y, height, width, vel, up_key, down_key, num_actions, state_size, device, current_step, *args):
        self.current_step = current_step
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.vel = vel
        self.up_key = up_key
        self.down_key = down_key
        self.num_actions = num_actions
        self.policy_net = DQN(*args).to(device)
        self.target_net = DQN(*args).to(device)
        self.memory = ReplayBuffer(1000000)
        self.strategy = EpsilonGreedyStrategy(start=1, end=0.003, decay=0.0001)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=0.001)
        self.device = device
        self.hidden_layers = args
    def draw(self, win):
        pygame.draw.rect(win, (255, 255, 255), (self.x, self.y, self.width, self.height))
    def key_movement(self, keys):
        if keys[getattr(pygame, self.up_key)]:
            self.moveup()
        if keys[getattr(pygame, self.down_key)]:
            self.movedown()
    def movement(self, action):
        if action == 0:
            self.moveup()
        if action == 1:
            self.movedown()
    def moveup(self):
        if self.y <= 0:
            self.y -= 0
        else:
            self.y -= self.vel
    def movedown(self):
        if self.y + self.height >= screen_height:
            self.y += 0
        else:
            self.y += self.vel
    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(self.current_step)
        # if self.memory.can_provide_sample(self.memory.capacity):
        self.current_step += 1
        if rate > random.random():
            return torch.tensor(random.randrange(self.num_actions)).to(self.device) # explore
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1) # exploit
    def get_current(self, states, actions):
        return self.policy_net(states).gather(dim=1, index=actions)
    def get_next(self, next_states):
        return self.target_net(next_states).max(dim=1, keepdim=True).values
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def follow(self):
        if ball.y > self.y + self.height/2:
            return 1
        if ball.y < self.y + self.height/2:
            return 0
    def hardcodedai(self):
        return self.follow()
class Ball:
    def __init__ (self, x, y, r, velx, vely):
        self.x = x
        self.y = y
        self.r = r
        self.velx = velx
        self.vely = vely

    def draw(self, win):
        pygame.draw.circle(win, (0, 0, 0), (self.x, self.y), self.r)

    def move(self):
        self.x += self.velx
        self.y += self.vely

    def bounce_x_increase(self, maxvel):
        if abs(ball.velx) > maxvel:
            if ball.velx > 0:
                ball.velx = (ball.velx) * -1
            else:
                ball.velx = (ball.velx) * -1
        else:
            if ball.velx > 0:
                ball.velx = (ball.velx + 1) * -1
            else:
                ball.velx = (ball.velx - 1) * -1
        return ball.velx

    def bounce(self):
        bounce, x, _, _ = self.collision()
        if bounce:
            if x == 1:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely - 5
            if x == 2:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely - 1
            if x == 3:
                self.bounce_x_increase(maxvel)
            if x == 4:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely + 1
            if x == 5:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely + 5

        if self.groundcollision():
            self.vely = (self.vely) * -1

    def collision(self):
        collision = False
        P1 = False
        P2 = False
        x = 0
        reward1 = 0
        reward2 = 0
        if ball.x + ball.r >= player1.x and ball.x - ball.r <= player1.x + player1.width:
            if ball.y + ball.r >= player1.y and ball.y + ball.r <= player1.y + player1.height:
                collision = True
                P1 = True
        if ball.x - ball.r <= player2.x + player2.width and ball.x + ball.r >= player2.x:
            if ball.y + ball.r >= player2.y and ball.y + ball.r <= player2.y + player2.height:
                collision = True
                P2 = True

        if collision & P1:
            ball.x = player1.x - ball.r
            if ball.y + ball.r > player1.y and ball.y <= math.floor(player1.y + player1.height/8):
                x = 1
            if ball.y > math.floor(player1.y + player1.height/8) and ball.y <=  math.floor(player1.y + 3*player1.height/8):
                x = 2
            if ball.y > math.floor(player1.y + 3*player1.height/8) and ball.y <= math.floor(player1.y + 5*player1.height/8):
                x = 3
            if ball.y > math.floor(player1.y + 5*player1.height/8) and ball.y <= math.floor(player1.y + 7*player1.height/8):
                x = 4
            if ball.y > math.floor(player1.y + 7*player1.height/8) and ball.y - ball.r <= math.floor(player1.y + player1.height):
                x = 5
            reward1 = 0
        if collision & P2:
            ball.x = player2.x + player2.width + ball.r
            if ball.y + ball.r > player2.y and ball.y <= math.floor(player2.y + player2.height/8):
                x = 1
            if ball.y > math.floor(player2.y + player2.height/8) and ball.y <=  math.floor(player2.y + 3*player2.height/8):
                x = 2
            if ball.y > math.floor(player2.y + 3*player2.height/8) and ball.y <= math.floor(player2.y + 5*player2.height/8):
                x = 3
            if ball.y > math.floor(player2.y + 5*player2.height/8) and ball.y <= math.floor(player2.y + 7*player2.height/8):
                x = 4
            if ball.y > math.floor(player2.y + 7*player2.height/8) and ball.y - ball.r <= math.floor(player2.y + player2.height):
                x = 5
            reward2 = 0
        return collision, x, reward1, reward2

    def groundcollision(self):
        groundcollision = False
        if ball.y + ball.r >= screen_height:
            groundcollision = True
        if ball.y - ball.r <= 0:
            groundcollision = True
        return groundcollision
class DQN(nn.Module):
    def __init__(self, *args, input_size = 6, output_size = 3):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, args[0])
        self.fc2 = nn.Linear(args[0], args[1])
        self.fc3 = nn.Linear(args[1], args[2])
        self.out = nn.Linear(args[2], output_size)


    def forward(self, t):
        t = torch.tanh(self.fc1(t))
        t = torch.tanh(self.fc2(t))
        t = torch.tanh(self.fc3(t))
        t = self.out(t)
        return t
class Pong_env():
    def __init__(self):
        self.num_actions = 3
        self.state_size = 6

    def step(self, actions, placar, timestep):
        player1.movement(actions[0])
        player2.movement(actions[1])
        ball.bounce()
        ball.move()
        state = self.get_state()
        _, _, _, reward_collision2 = ball.collision()
        done, reward1, reward2 = self.score(player1_x, player1_y, player2_x, player2_y, placar, timestep)
        # reward -= 1
        # reward += reward_collision2
        return state, reward1, reward2, done

    def reset(self):
        ball.x = math.floor(screen_width/2)
        ball.y = math.floor(screen_height/2)
        ball.velx = 2 * random.choice(r)
        ball.vely = random.choice(r)
        player1.x = player1_x
        player1.y = player1_y
        player2.x = player2_x
        player2.y = player2_y
        state = np.array([player1.y/500, player2.y/500, ball.x/800, ball.y/500, ball.velx/10, ball.vely/10])

        return state

    def score(self, player1_x, player1_y, player2_x, player2_y, placar, timestep):
        point = False
        reward1 = 0
        reward2 = 0

        if timestep >= 2000:
            reward1 = -1
            reward2 = -1
            point = True
            print('interrupted')

        if ball.x + ball.r >= screen_width:
            reward1 = -1
            reward2 = 1
            point = True
            placar +=  np.array([1, 0])
            print(placar)

        if ball.x - ball.r <= 0:
            reward1 = 1
            reward2 = -1
            point = True
            placar +=  np.array([0, 1])
            print(placar)

        if point:
            ball.x = math.floor(screen_width/2)
            ball.y = math.floor(screen_height/2)
            ball.velx = 2 * random.choice(r)
            ball.vely = random.choice(r)
            player1.x = player1_x
            player1.y = player1_y
            player2.x = player2_x
            player2.y = player2_y

        return point, reward1, reward2

    def get_state(self):
        state = np.array([player1.y/500, player2.y/500, ball.x/800, ball.y/500, ball.velx/10, ball.vely/10])
        return state

    def get_state_reverse(self):
        state = np.array([player2.y, player1.y, screen_width - ball.x, ball.y, -ball.velx, ball.vely])
        return state

    def render(self, win):
        win.fill((0,100,100))
        player1.draw(win)
        player2.draw(win)
        ball.draw(win)
        pygame.display.update()
        clock.tick(90)
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return  self.end + (self.start - self.end) * \
            math.exp(-1 * current_step * self.decay)
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
class Tools():
    def np_to_torch(state):
        state = torch.tensor(state, dtype = torch.float32)
        state = torch.reshape(state, (1,len(state)))
        state = state.to(device)
        return state
    def extract_tensors(experiences):

        batch = Experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action2)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)
        t5 = batch.done
        return (t1, t2, t3, t4, t5)
    def preprocess(action):
        action = torch.tensor(action)
        action = torch.reshape(action, (1, 1))
        return action
    def invert(state):
        p1_y = state[0]
        p2_y = state[1]
        bx = state[2]
        by = state[3]
        bvx = state[4]
        bvy = state[5]
        return np.array([p2_y, p1_y, screen_width/800 - bx, by, -bvx, bvy])

def createaxes():
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)
    return fig, ax1, ax2, ax3
def plotaxes(fig, ax1, ax2, ax3, values, limit, losses, times, avgscore):

    ax1.cla()
    ax2.cla()
    ax3.cla()

    ax1.set_title('Pong')
    ax2.set_title('Loss')
    ax3.set_title('Avg. Episode duration')

    ax1.set_xlabel('Games')
    ax2.set_xlabel('Episodes')
    ax3.set_xlabel('Episodes (x500)')

    ax3.set_ylabel('Time (s)')

    ax1.set_ylim(-21, 21)
    ax1.plot(values, 'b')
    ax1.plot(limit, 'g--')
    ax1.plot(avgscore, color='#f59542', linewidth = 2.0)
    ax2.plot(losses, 'r')
    ax3.plot(times, 'y')

    plt.tight_layout()
    plt.show()
def plot(values, valor):
    plt.figure(1)
    plt.clf()
    plt.title('Pong')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(np.arange(0, len(values)), values)
    plt.plot(np.arange(0, len(values)), (valor*np.ones(len(values))), 'r', '--')
def plottime(values):
    plt.figure(4)
    plt.clf()
    plt.title('Average time per episode in seconds')
    plt.ylabel('time(s)')
    plt.plot(values)
def plotscores(values):
    plt.figure(3)
    plt.clf()
    plt.title('Pong')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(values)
def plotloss(values):
    plt.figure(2)
    plt.clf()
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(values)

    plt.pause(0.0001)
def load_model(model, current_step = 0):
    checkpoint = torch.load(model+'-checkpoint.tar')
    player2.policy_net.load_state_dict(checkpoint['policy_net_p2'])
    player2.optimizer.load_state_dict(checkpoint['optimizer_p2'])
    player2.memory.memory = checkpoint['memory2']
    player2.current_step = current_step
    player2.policy_net.train()
    player2.update_target_net()
    point = torch.load(model + '.tar')
    player1.policy_net.load_state_dict(checkpoint['policy_net_p2'])
    player1.policy_net.eval()
    print("Model loaded successfully")
def save_model(model, player, checkpoint = False):
    if checkpoint:
        torch.save({
        'policy_net_p2': player2.policy_net.state_dict(),
        'optimizer_p2': player2.optimizer.state_dict(),
        'memory2': player2.memory.memory
        }, model+'-checkpoint.tar')
    else:
        torch.save({
        'policy_net_p2': player2.policy_net.state_dict(),
        'optimizer_p2': player2.optimizer.state_dict(),
        }, model+'.tar')
def save_placar(placar, avgscore, points):
    valor = placar[0] - placar[1]
    points = np.append(points, valor)
    np.save('points.npy', points)
    avgscore = np.append(avgscore, np.sum(points[-10:])/10)
    np.save('avgscore.npy', avgscore)
    placar = np.array([0,0], dtype = int)
    return valor, points, avgscore, placar
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

model = 'model100x3'
# model2 = 'modelcalm.tar'
hidden_layers2 = [100, 100, 100]
hidden_layers1 = [100, 100, 100]
k = 4
current_step = 0
env = Pong_env()
Experience = namedtuple('Experience', ('state', 'action2',
    'next_state', 'reward', 'done'))
screen_width = 800
screen_height = 500
run = True
maxvel = 10
player_speed = 5
height = 56
width = 10
player1_x = screen_width - 50 - width
player1_y = screen_height/2 - height/2
player2_x = 50
player2_y = screen_height/2 - height/2
r = [-3, -2, -1, 1, 2, 3]
ball_speed_x = 2 * random.choice(r)
ball_speed_y = 2 * random.choice(r)

player2 = Player(player2_x, player2_y, height, width, player_speed, "K_w", "K_s", env.num_actions, env.state_size, device, current_step, *hidden_layers2)
player2.update_target_net()
player1 = Player(player1_x, player1_y, height, width, player_speed, "K_UP", "K_DOWN", env.num_actions, env.state_size, device, current_step, *hidden_layers1)
player1.update_target_net()
ball = Ball(math.floor(screen_width/2), math.floor(screen_width/2), 6, ball_speed_x, ball_speed_y)
clock = pygame.time.Clock()
points = []
game = []
loss = 0
loss_p2 = 0
losses = []
scores = []
game_n = 0
placar = np.array([0,0], dtype = int)
ep = 0
avgtime = []
win = 0
style.use('ggplot')
avgscore = np.array([])
