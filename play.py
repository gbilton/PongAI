from classes import *
from itertools import count
from multiprocessing import Pool
import time

# device = torch.device('cpu')
# player2 = Player(player2_x, player2_y, height, width, player_speed, "K_w", "K_s", env.num_actions, device, current_step)

def self_play():
    run = True
    pygame.init()
    win = pygame.display.set_mode((screen_width, screen_height))
    while run:
        state = env.reset()
        inverted_state = Tools.invert(state)
        state = Tools.np_to_torch(state)
        inverted_state = Tools.np_to_torch(inverted_state)
        for timestep in count():
            action2 = player2.policy_net(state).argmax(dim=1).item()
            action1 = player2.policy_net(inverted_state).argmax(dim=1).item()
            for _ in range(k):
                # keys = pygame.key.get_pressed()
                # action1 = player1.key_movement(keys)
                actions = np.array([action1, action2])
                state, _, _, done = env.step(actions, placar, timestep)
                inverted_state = Tools.invert(state)

                if render:
                    env.render(win)
                if done:
                    break
            state = Tools.np_to_torch(state)
            inverted_state = Tools.np_to_torch(inverted_state)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    done = True
                break
            if done:
                break
def play(numbers):
    games = 0
    state = env.reset()
    inverted_state = Tools.invert(state)
    state = Tools.np_to_torch(state)
    inverted_state = Tools.np_to_torch(inverted_state)
    for timestep in count():
        action2 = player2.policy_net(state).argmax(dim=1).item()
        action1 = player1.policy_net(inverted_state).argmax(dim=1).item()
        for _ in range(k):
            # keys = pygame.key.get_pressed()
            # action1 = player1.key_movement(keys)
            actions = np.array([action1, action2])
            state, _, _, done = env.step(actions, placar, timestep)
            inverted_state = Tools.invert(state)
            if render:
                env.render(win)
            if done:
                break
        state = Tools.np_to_torch(state)
        inverted_state = Tools.np_to_torch(inverted_state)
        if done:
            games+=1
        if games >= 80:
            break
    return placar
def serial(numbers):
    t2 = time.time()
    for _ in numbers:
        play(numbers)
        print("Serial took: ", time.time() - t2)
def parallel(numbers):
    vitorias = 0
    derrotas = 0
    placares = np.array([])
    t1 = time.time()
    p = Pool()
    placar = p.map(play, numbers)
    placares = np.append(placares, placar)
    p.close()
    p.join()
    print("Pool took: ", time.time()-t1)

    for i in range(len(placares)):
        if i%2 == 0:
            vitorias += placares[i]
        else:
            derrotas += placares[i]
    print("Total score: ", vitorias, derrotas)
def challenge():
    run = True
    pygame.init()
    win = pygame.display.set_mode((screen_width, screen_height))
    while run:
        state = env.reset()
        state = Tools.np_to_torch(state)
        for timestep in count():
            action2 = player2.policy_net(state).argmax(dim=1).item()
            for _ in range(k):
                keys = pygame.key.get_pressed()
                action1 = player1.key_movement(keys)
                actions = np.array([action1, action2])
                state, _, _, done = env.step(actions, placar, timestep)
                env.render(win)
                if done:
                    break
            state = Tools.np_to_torch(state)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    done = True
                break
            if done:
                break
def two_ai():
    run = True
    pygame.init()
    win = pygame.display.set_mode((screen_width, screen_height))
    while run:
        state = env.reset()
        inverted_state = Tools.invert(state)
        state = Tools.np_to_torch(state)
        inverted_state = Tools.np_to_torch(inverted_state)
        for timestep in count():
            action2 = player2.policy_net(state).argmax(dim=1).item()
            action1 = player1.policy_net(inverted_state).argmax(dim=1).item()
            for _ in range(k):
                # keys = pygame.key.get_pressed()
                # action1 = player1.key_movement(keys)
                actions = np.array([action1, action2])
                state, _, _, done = env.step(actions, placar, timestep)
                inverted_state = Tools.invert(state)

                if render:
                    env.render(win)
                if done:
                    break
            state = Tools.np_to_torch(state)
            inverted_state = Tools.np_to_torch(inverted_state)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    done = True
                break
            if done:
                break
def robot():
    run = True
    pygame.init()
    win = pygame.display.set_mode((screen_width, screen_height))
    while run:
        state = env.reset()
        state = Tools.np_to_torch(state)
        for timestep in count():
            # print(timestep)
            action2 = player2.policy_net(state).argmax(dim=1).item()
            for _ in range(k):
                action1 = player1.follow()
                actions = np.array([action1, action2])
                state, _, _, done = env.step(actions, placar, timestep)
                if render:
                    env.render(win)
                if done:
                    break
            state = Tools.np_to_torch(state)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    done = True
                break
            if done:
                break
def setup():
    checkpoint = torch.load(model + '.tar')
    # player2.hidden_layers = checkpoint['hidden_layers2']
    player2.policy_net.load_state_dict(checkpoint['policy_net_p2'])
    player2.policy_net.eval()

    # hidden_layers1 = player1.hidden_layers
    checkpoint = torch.load('model100x3-3.tar')
    player1.policy_net.load_state_dict(checkpoint['policy_net_p2'])
    player1.policy_net.eval()

    numbers = range(12)
    return numbers


if __name__ == '__main__':
    render = False
    numbers = setup()
    # placares = parallel(numbers)
    # serial(numbers)
    # challenge()
    # self_play()
    # two_ai()
    robot()
