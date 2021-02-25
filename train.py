from classes import *

# Hyper-parameters
batch_size = 64
gamma = 0.99
target_update = 100
num_episodes = 100000
save_checkpoint = 1000
current_step = 1000000
limite = 10

load_model(model, current_step)

t0 = time.time()
for episode in range(num_episodes):
        ep+=1
        score = 0
        state = env.get_state()
        inverted_state = Tools.invert(state)
        state = Tools.np_to_torch(state)
        inverted_state = Tools.np_to_torch(inverted_state)
        for timestep in count():
            action1 = torch.reshape(player1.policy_net(inverted_state).argmax(dim=1), (1,1))
            action2 = torch.reshape(player2.select_action(state), (1,1))
            # action1 = torch.reshape(player1.select_action(inverted_state), (1,1))
            for _ in range(k):
                actions = np.array([action1, action2.item()])
                next_state, reward1, reward2, done = env.step(actions, placar, timestep)
                inverted_next_state = Tools.invert(next_state)
                if done:
                    break

            reward2 = Tools.preprocess(reward2)
            reward1 = Tools.preprocess(reward1)
            score += reward2

            if not done:
                next_state = Tools.np_to_torch(next_state)
                inverted_next_state = Tools.np_to_torch(inverted_next_state)
            else:
                next_state = Tools.np_to_torch(np.zeros(len(next_state)))
                inverted_next_state = Tools.np_to_torch(np.zeros(len(inverted_next_state)))

            player2.memory.push(Experience(state, action2, next_state, reward2, done))
            state = next_state
            inverted_state = inverted_next_state

            if player2.memory.can_provide_sample(batch_size):
                experiences = player2.memory.sample(batch_size)

                states, actions2, rewards2, next_states, dones = Tools.extract_tensors(experiences)
                rewards2 = rewards2.to(device)
                current_q_values_p2 = player2.get_current(states, actions2)
                next_q_values_p2 = player2.get_next(next_states)

                for idx, next_state in enumerate(next_states):
                    if dones[idx] == True:
                        next_q_values_p2[idx] = 0

                target_q_values_p2 = (next_q_values_p2 * gamma) + rewards2

                loss_p2 = F.mse_loss(current_q_values_p2, target_q_values_p2).to(device)
                player2.optimizer.zero_grad()
                loss_p2.backward()
                player2.optimizer.step()

            if done:
                losses.append(loss_p2.item())
                scores.append(score)
                if episode % 1000 == 0:
                    np.save('losses.npy', losses)
                break

        if (episode+1) % save_checkpoint == 0:
            a = (time.time()-t0)/ep
            print(f'average episode time = {(a):.2f} s.')
            t0 = time.time()
            ep = 0
            avgtime.append(a)
            np.save('avgtime.npy', avgtime)
            save_model(model, player2, checkpoint = True)
            print("saved new checkpoint")
            # plottime(avgtime)


        if (episode+1) % target_update == 0:
            player2.update_target_net()
            print("target net updated")

        if placar[0] == 21:
            game_n += 1
            game = np.append(game, game_n)
            print('VITORIAAAAAA!!!')
            valor, points, avgscore, placar = save_placar(placar, avgscore, points)
            if valor >= limite:
                win += 1
            else:
                win = 0
            if win == 5:
                player1.policy_net.load_state_dict(player2.policy_net.state_dict())
                player1.optimizer.load_state_dict(player2.optimizer.state_dict())
                player1.policy_net.eval()
                print('Player 1 updated')
                player2.memory.memory = []
                win = 0
                save_model(model, player2)
                save_model(model, player2, checkpoint = True)
            print(episode)
            print(placar)
        if placar[1] == 21:
            win = 0
            game_n += 1
            game = np.append(game, game_n)
            print('Derrota')
            valor, points, avgscore, placar = save_placar(placar, avgscore, points)
            print(episode)
            print(placar)
