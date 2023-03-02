import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy


class CircularBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data[:] = 0  

class Memory(object):
    
    """
    Implement a typical memory buffer class to add elements and retrieve batches of data from the replay buffer
    Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
    """
    def __init__(self, limit, observation_shape, action_shape, next_actions=False):
        self.limit = limit

        self.states = CircularBuffer(limit, shape=observation_shape)
        self.actions = CircularBuffer(limit, shape=action_shape)
        self.rewards = CircularBuffer(limit, shape=(1,))
        self.next_states = CircularBuffer(limit, shape=observation_shape)
        self.next_actions = CircularBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = CircularBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries-1, size=batch_size)
        #sample a batch from the replay buffer
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)

        if next_actions is not None:
            return states_batch, actions_batch, rewards_batch, next_states_batch, next_actions, terminals_batch
        else:
            return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, training=True):
        #inserts a new element into the replay buffer
        if not training:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_actions.clear()
        self.terminals.clear()

    @property
    def nb_entries(self):
        return len(self.states)

class buildQnetwork(nn.Module):
    """
    Deep recurrent Q network using Pytorch
    
    Input
    -----------
    inputDimensions : input dimensions of the state space
    action_size: action space size
    action_parameter_size: action parameter size
    layer_neurons: number of neurons 
    initial_std: initial standard deviation for initializing layers
    activation: activation function
    update_rule: choice of the optimizer 
    -----------
    """

    def __init__(self, inputDimensions, action_size, action_parameter_size, layer_neurons=[128,], initial_std=0.0001, activation="relu",update_rule="adam"):
        super(buildQnetwork, self).__init__()
        
        self.inputDimensions = inputDimensions
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation
        
        # create NN layers: 
        self.layers = nn.ModuleList()
        inputSize = self.inputDimensions + self.action_parameter_size # input size
            
        self.layers.append(nn.Linear(inputSize, layer_neurons[0]))    
        self.layers.append(nn.LSTM(input_size=layer_neurons[0], hidden_size=layer_neurons[0], num_layers=1))
        self.layers.append(nn.Linear(layer_neurons[0], layer_neurons[0]))           
        self.layers.append(nn.Linear(layer_neurons[0], self.action_size))

        #  initialize neural network parameters
        # the first layer only
        nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity=activation) #Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a normal distribution
        nn.init.zeros_(self.layers[0].bias) # initialized bias of first layer to zero 
            
        nn.init.normal_(self.layers[-1].weight, mean=0., std=initial_std) # initialize the last layer with a normal distribution
        nn.init.zeros_(self.layers[-1].bias) # initialize the bias of the last layer to zero
            
    def forward(self, state, action_parameters):
        # implement forward
        x = torch.cat((state, action_parameters), dim=1) # combine state and action parameters
        x = F.relu(self.layers[0](x))
        
        h0 = torch.zeros(1, x.size(1)).requires_grad_() # hidden value for LSTM
        c0 = torch.zeros(1,x.size(1)).requires_grad_() # cell state for LSTM

        x, (hn, cn) = self.layers[1](x, (h0.detach(), c0.detach())) # this is the lstm layer
        
        x = F.relu(self.layers[2](x))
        
        Q = self.layers[-1](x) # output layer
        return Q


class buildParamActorNetwork(nn.Module):
    """
    Deep Parameter actor network using Pytorch
    
    Input
    -----------
    inputDimensions : input dimensions of the state space
    action_size: action space size
    action_parameter_size: action parameter size
    layer_neurons: number of neurons 
    initial_std: initial standard deviation for initializing layers
    activation: activation function
    update_rule: choice of the optimizer 
    -----------
    """


    def __init__(self, inputDimensions, action_size, action_parameter_size, layer_neurons,
                 initial_std=0.0001, activation="relu",update_rule="adam"):
        super(buildParamActorNetwork, self).__init__()

        self.inputDimensions = inputDimensions
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.inputDimensions
        self.layers.append(nn.Linear(inputSize, layer_neurons[0]))
#        self.layers.append(nn.LSTM(input_size=layer_neurons[0], hidden_size=layer_neurons[0], num_layers=1))
        self.layers.append(nn.Linear(layer_neurons[0], self.action_parameter_size)) # output layer

        # initialise neural network parameters of the first layer
        nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity=activation) #one can also try nn.init.normal_(self.layers[i].weight, std=init_std)
        nn.init.zeros_(self.layers[0].bias) # put bias to zero

        # initialise neural network parameters of the last layer
        nn.init.normal_(self.layers[-1].weight, std=initial_std) # noraml distribution
        nn.init.zeros_(self.layers[-1].bias)


    def forward(self, state):
        x = state
        x = F.relu(self.layers[0](x))
        
#        h1 = torch.zeros(1,128).requires_grad_() # hidden value for LSTM
#        c1 = torch.zeros(1,128).requires_grad_() # cell state for LSTM
#        x, (hn, cn) = self.layers[1](x, (h1.detach(), c1.detach())) # this is the lstm layer

        action_params = self.layers[-1](x)
        
        return action_params


class Agent_PDQN:
    """
    Agent for parameterised action spaces
    Paper: "Parametrized Deep Q-Networks Learning: Reinforcement
Learning with Discrete-Continuous Hybrid Action Space"
https://arxiv.org/pdf/1810.06394v1.pdf
    -----------
    state_space: the observation space
    action_space: the action space
    QNN: indicates which class to use for building the actor parameter network
    actor_paramNN: indicates which class to use for building the actor parameter network
    epsilon_initial: initial epsiolon value for exploration
    epsilon_final: final epsilon value
    epsilon_steps: maximum epsilon step
    batch_size: batch size
    gamma: discount factor gamma
    tau_Q: parameter tau for soft update of the Q network
    tau_actor_param: parameter tau for soft update of the actor parameter network
    replay_memory_size: replay buffer size
    Q_learning_rate: learning rate for the Q network
    actor_param_learning_rate: learning rate for the actor parameter network
    memory_trigger: trigger value before starting the training 
    clip_grad: gradient clip value
    layer_neurons: number of neurons
    initial_std: initial standard deviation for initializing weights
    update_rule="adam": indicate optimizer
    activation : activation functions for layers
    device="cuda" if torch.cuda.is_available() else "cpu": simulations were run using a CPU
    random_seed: random seed value

    """
        
    def __init__(self,state_space=None,action_space=None,QNN=buildQnetwork,actor_paramNN=buildParamActorNetwork, epsilon_initial=1.0, epsilon_final=0.01,epsilon_steps=10000,batch_size=64,gamma=0.9,tau_Q=0.01,tau_actor_param=0.001, replay_memory_size=1000000,Q_learning_rate=0.0001,actor_param_learning_rate=0.00001,memory_trigger=0,clip_grad=10,layer_neurons=[128], initial_std=0.0001,update_rule="adam",activation="relu",device="cuda" if torch.cuda.is_available() else "cpu",random_seed=None,memory=Memory):
        
#        super(Agent_PDQN, self).__init__(state_space, action_space)
        
        # typical initialization
        self.device = torch.device(device)
        self.action_space=action_space
        self.state_space=state_space
        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1,self.num_actions+1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.update_rule=update_rule
        self.activation=activation

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.memory_trigger = memory_trigger
        self.Q_learning_rate = Q_learning_rate
        self.actor_param_learning_rate = actor_param_learning_rate
        self.tau_Q = tau_Q
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self.episodes = 0
        self.clip_grad = clip_grad

        self.random_seed = random_seed
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.np_random = np.random.RandomState(seed=self.random_seed)
        torch.manual_seed(self.random_seed)
        
        self.layer_neurons=layer_neurons
        self.initial_std=initial_std
        self.Memory=Memory
        self.replay_memory = self.Memory(replay_memory_size, state_space.shape, (1+self.action_parameter_size,), next_actions=False)
        self.Q = QNN(self.state_space.shape[0], self.num_actions, self.action_parameter_size,self.layer_neurons,self.initial_std,activation=self.activation,update_rule=self.update_rule).to(device)
        self.Q_target = QNN(self.state_space.shape[0], self.num_actions, self.action_parameter_size,self.layer_neurons,self.initial_std,activation=self.activation,update_rule=self.update_rule).to(device)
        self.target_network_initialize(self.Q, self.Q_target)
        self.Q_target.eval()

        self.actor_param = actor_paramNN(self.state_space.shape[0], self.num_actions, self.action_parameter_size,self.layer_neurons,self.initial_std,activation=self.activation,update_rule=self.update_rule).to(device)
        self.actor_param_target = actor_paramNN(self.state_space.shape[0], self.num_actions, self.action_parameter_size,self.layer_neurons, self.initial_std,activation=self.activation,update_rule=self.update_rule).to(device)
        self.target_network_initialize(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        if self.update_rule=="adam":
            self.Q_optimiser = optim.Adam(self.Q.parameters(), lr=self.Q_learning_rate) 
            self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.actor_param_learning_rate) 
        elif self.update_rule=="SGD":
            self.Q_optimiser = optim.SGD(self.Q.parameters(), lr=self.Q_learning_rate) 
            self.actor_param_optimiser = optim.SGD(self.actor_param.parameters(), lr=self.actor_param_learning_rate) 
        else: # adam opt by default
            self.Q_optimiser = optim.Adam(self.Q.parameters(), lr=self.Q_learning_rate) 
            self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.actor_param_learning_rate) 

    def update_epsilon(self):
        # update epsilon (for exploration)
        self.episodes += 1

        ep = self.episodes
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def reshape_action(self, action, action_param):
        # reshapes the action by returning a tuple  (action, action_parameters)
        params = [np.zeros((1,)), np.zeros((1,)), np.zeros((1,))]
    
        params[action][:] = action_param
        return (action, params)
            
    def chooseAction(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)
#            all_action_parameters=all_action_parameters[0,:]
            
            rnd = self.np_random.uniform()
            if rnd < self.epsilon: # # action exploration
                action = self.np_random.choice(self.num_actions)
                all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                              self.action_parameter_max_numpy))
            else:
                # # action exploitation---choosing the best action using the Q network
                Q_a = self.Q.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)

            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            action_parameters = all_action_parameters[offset:offset+self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def inverting_gradients(self, gradient_val, action_params):
        # an approach for bounding the action space
        # Allows parameters to approach the bounds of the ranges without exceeding them
        # https://www.cs.utexas.edu/~pstone/Courses/394Rfall19/resources/week9-matthew.pdf
        
        pmax = self.action_parameter_max
        pmin = self.action_parameter_min
        range_val = self.action_parameter_range

        with torch.no_grad():
            index = gradient_val > 0
            gradient_val[index] *= (index.float() * (pmax - action_params) / range_val)[index]
            gradient_val[~index] *= ((~index).float() * (action_params - pmin) / range_val)[~index]

        return gradient_val

    def step(self, state, action, reward, next_state, next_action, terminal):
        act, all_action_parameters = action
        self._step += 1

        self._add_sample(state, np.concatenate(([act],all_action_parameters)).ravel(), reward, next_state, np.concatenate(([next_action[0]],next_action[1])).ravel(), terminal=terminal)
        if self._step >= self.batch_size and self._step >= self.memory_trigger:
            self.train()

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        # add one element to the replay buffer
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def train(self):
        # train the agent by updating the networks weights
        # the pseudocode for training is included in Algorithm 1 of the paper: https://arxiv.org/pdf/1810.06394v1.pdf
        # paper title: "Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space"
        if self._step < self.batch_size or self._step < self.memory_trigger: 
            return # wait until the replay buffer is sufficiently filled
        # Sample a random batch from the replay buffer
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
        # extract and reshape data to be used for training the networks
        states = torch.from_numpy(states).to(self.device) # states
        actions_combined = torch.from_numpy(actions).to(self.device)  # all actions combined
        actions = actions_combined[:, 0].long() # extract actions (integer)
        action_parameters = actions_combined[:, 1:]  # extract action parameters
        rewards = torch.from_numpy(rewards).to(self.device).squeeze() # create and reshape rewards
        next_states = torch.from_numpy(next_states).to(self.device) # create next states tensor
        terminals = torch.from_numpy(terminals).to(self.device).squeeze() # create terminals tensor

        # ********************* Update the Q-network **********************************
        
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states) # 
            pred_Q_a = self.Q_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # Compute the target
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # Compute current Q-values 
        Q_values = self.Q(states, action_parameters)
        y_predicted = Q_values.gather(1, actions.view(-1, 1)).squeeze()
        loss_Q = F.mse_loss(y_predicted, target) # calculate MSE

        self.Q_optimiser.zero_grad()
        loss_Q.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.clip_grad)
        self.Q_optimiser.step()

        # ************************* Update actor parameter weights******************
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        Q = self.Q(states, action_params)
        Q_loss = torch.mean(torch.sum(Q, 1))
        self.Q.zero_grad()
        Q_loss.backward()
        
        delta_inverted_grad = deepcopy(action_params.grad.data)
        action_params = self.actor_param(Variable(states))
        delta_inverted_grad[:] = self.inverting_gradients(delta_inverted_grad, action_params) 

        final = -torch.mul(delta_inverted_grad, action_params)
        self.actor_param.zero_grad()
        final.backward(torch.ones(final.shape).to(self.device))
        torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        self.update_weights(self.Q, self.Q_target, self.tau_Q) # perform soft update for Q network
        self.update_weights(self.actor_param, self.actor_param_target, self.tau_actor_param) # perform soft update for actor parameter network

    def update_weights(self,old_network, target_network, tau):
        # perform soft update using the parameter tau
        for target_param, param in zip(target_network.parameters(), old_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    def target_network_initialize(self, old_network, target_network):
        # copy network weights into target networks as a start
        for target_param, param in zip(target_network.parameters(), old_network.parameters()):
            target_param.data.copy_(param.data)
            
