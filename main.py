#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gym
import gym_platform
from gym.wrappers import Monitor
from wrappers import Normalize_state,Flatten_Action_Space, Normalise_actions
import numpy as np
from RPDQN_agent import Agent_PDQN


# ----------------------
# Set Environment Parameters
# ----------------------
experiment_number=1
random_seed=123# Random seed--fix it
evaluation_episodes=1000  # the number of episodes used for final evaluation after training is complete
episodes=20000 # this variable expresses the number of epsiodes used for training the agent
batch_size=128  # this is the batch size
gamma=0.9     # this is the Discount factor
memory_trigger=500# Number of items required before the agents starts to learn
replay_memory_size=10000 # replay memory size 
epsilon_steps=1000# Number of episodes over which epsilon decreases linearly 
epsilon_final=0.01# Final smallest epsilon value once "epsilon_steps" pass 
tau_Q=0.1#   soft update parameter for target Q_network update
tau_actor_param=0.001# soft update parameter for actor-parameter network update 
Q_learning_rate=0.001 # learning rate of the Q network
actor_param_learning_rate=0.0001# learning rate of tha actor-parameter network
clip_grad=10.#  gradient clipping limit
layers=[128,]   #, help='Duplicate action-parameter inputs.')
save_dir="results"  #, help='Output directory.', type=str)
render_freq=100 # How often to render the environment
visualise=True  # visualise game states
update_rule="adam" # or SGD or other update rules can be integrated into the code for further testing
activation="relu" # the activation function for neural networks

if __name__ == "__main__":
    # ----------------------
    # Instantiate environment
    # ----------------------

    env = gym.make('Platform-v0')

    #-----definition of features in the observation space
    # basic features
    #features #1: player position
    #features #2: player velocity
    #features #3: enemy position
    #features #4: enemy velocity
    # platform features (implicit features which are deduced from basic features)
    #features #5: wd1: width of the platform on whih the player is standing
    #features #6: wd2: width of the next platform on whih the player will jump (equal to zero if the player is on platform3)
    #features #7: gap: width of the gap right in front of the player (equal to zero if player is on platform 3)
    #features #8: pos: can take three values: val1=0 if the player haven't reached platform 2 yet; or val2=Width_platform1+width_gap1 if the player hasn't reached platform 3 yet; or val3=total width if the player reached platform 3 
    #features #9: diff: difference in height ---typically zero

    #-----definition of action space
    # The agent can choose one of three actions: run or hop or leap
    # and it can perform the selected action with a specific intensity--that we call parameter
    # parameter 1: expresses running for a  given power/velocity 
    # parameter 2: expresses the power of jumping high to a position.
    # parameter 3: expresses the power of jumping low (over a gap) to a specific position.


    print('*******observation space*************')
    print('number of elements in the observation space',env.observation_space.spaces[0].shape)
    print('highest values of observation space',env.observation_space.spaces[0].high)
    print('lowest values of observation space',env.observation_space.spaces[0].low)

    print('*******Action space*************')
    print('number of actions',env.action_space.spaces[0].n)
    print('highest values of parameter actions',env.action_space.spaces[1].spaces[0].high,env.action_space.spaces[1].spaces[1].high,env.action_space.spaces[1].spaces[2].high)
    print('lowest values of parameter actions',env.action_space.spaces[1].spaces[0].low,env.action_space.spaces[1].spaces[1].low,env.action_space.spaces[1].spaces[2].low)

    # --------------------------------------------------------------
    # # Pre-processing of observation and action spaces 
    # --------------------------------------------------------------
    # modify an existing environment without having to alter the underlying code directly. 
    env = Normalize_state(env) # rescale the state space to [-1,1] rather than [0,1]
    # flatten the actions (instead of tuple(xx,tuple(xxx,xxx,xxx)) , one tuple only--which groups all elements)
    #env = PlatformFlattenedActionWrapper(env) 
    env =Flatten_Action_Space(env)
    # normalize actions to the interval [-1,1]
    env = Normalise_actions(env)

    # write information about the agent’s performance in a file with optional video recording of the agent in action.         
    env = Monitor(env, directory=save_dir, video_callable=False, write_upon_reset=False, force=True)
    # initialize random seed
    env.seed(random_seed)
    np.random.seed(random_seed)

    # environment action and state spaces
    state_space=env.observation_space.spaces[0]
    action_space=env.action_space

    # --- Instantiate agent ---    
    agent = Agent_PDQN(state_space,action_space , batch_size=batch_size, 
                       Q_learning_rate=Q_learning_rate, actor_param_learning_rate=actor_param_learning_rate,
                           epsilon_steps=epsilon_steps, gamma=gamma,
                           tau_Q=tau_Q, tau_actor_param=tau_actor_param,
                           clip_grad=clip_grad, memory_trigger=memory_trigger,
                           replay_memory_size=replay_memory_size, epsilon_final=epsilon_final,
                           layer_neurons= layers,initial_std= 0.0001,
                           random_seed=random_seed, activation=activation, update_rule=update_rule)

    print("############################ Agent architecture ############################")
    print("########################################################################")
    print("########################################################################")

    print(agent.Q.layers)
    print(agent.actor_param.layers)

    max_steps = 200 # see _init_ of platform environment
    total_reward = 0.
    returns = []

    print("############################ Training starts ############################")
    print("########################################################################")
    print("########################################################################")

    for i in range(episodes):
        # reset the environment at the beginning of each episode
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        if visualise and i % render_freq == 0:# rendering the environment
            env.render()
        #choose the first action 
        act, act_param, all_action_parameters = agent.chooseAction(state) 
        action = agent.reshape_action(act, act_param)

        episode_reward = 0.
        for j in range(max_steps):
            #apply the action
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            # choose a new action given the new state
            next_act, next_act_param, next_all_action_parameters = agent.chooseAction(next_state)
            next_action = agent.reshape_action(next_act, next_act_param)
            # store transition--(s,a,r,s',a',done) and do one training step by updating the weights of NNs
            agent.step(state, (act, all_action_parameters), reward, next_state,
                           (next_act, next_all_action_parameters), terminal)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            # action variable receives next action variable and state variable receives next state variable
            action = next_action
            state = next_state
            
            #increment the episode reward
            episode_reward += reward
            #render the env
            if visualise and i % render_freq == 0:
                env.render()
            #if we reach the end of an episode stop and start a new episode
            if terminal:
                break
        #update epsilon value (for exploration vs exploitation)
        agent.update_epsilon()
        # add the episode reward to the returns list
        returns.append(episode_reward) 
        # print result every 100 episodes
        if i % 100 == 0:
            print("episode: {0:.5s}/{1:.5s}, score: {2:.4f}  || average score(last 100 episodes): {3:.5f} ".format(str(i), str(episodes), episode_reward, np.array(returns[-100:]).mean()))
    #close the environment when all episodes end
    env.close()

    print("############################ Training stops ############################")
    print("########################################################################")
    print("########################################################################")

    print("Average score over all episodes =", sum(returns) / len(returns))
    print("Average score (last 100 episodes) =", sum(returns[-100:]) / 100.)
    #save the history of training (i.e., the obtained rewards per episode) 
    np.save(save_dir+ "/"+"training_experiment"+"{}".format(str(experiment_number)),returns)

    print("############################ Evaluation begins ############################")
    print("########################################################################")
    print("########################################################################")
    
    def final_evaluation(env, agent, episodes):
        # this function returns a list of rewards obtained over a given number of episodes (default value is 1000 episodes) 
        list_rewards = []
        for i in range(episodes):
            state, other = env.reset()
            terminal = False
            total_reward = 0.
            while not terminal:
                state = np.array(state, dtype=np.float32, copy=False)
                act, act_param, all_action_parameters = agent.chooseAction(state)
                params = [np.zeros((1,)), np.zeros((1,)), np.zeros((1,))]
                params[act][:] = act_param        
                action = (act,params)
                (state, other1), reward, terminal, other2 = env.step(action)
                total_reward += reward # # accumulate reward
            list_rewards.append(total_reward)
        return np.array(list_rewards)
    # do a performance evaluation (with no exploration) over 1000 episodes
    print("Perform Evaluation over {} episodes".format(evaluation_episodes))
    agent.epsilon_final = 0. # to anneal exploration --exploitation only
    agent.epsilon = 0. # to anneal exploration --exploitation only
    evaluation_score = final_evaluation(env, agent, evaluation_episodes)
    print("Average score:::", sum(evaluation_score) / len(evaluation_score))
    #save the evaluation results as an array in "results" directory
    np.save(save_dir+ "/evaluation_score_experiment" "{}".format(str(experiment_number)),evaluation_score)


# In[ ]:




