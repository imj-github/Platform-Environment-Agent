import copy
import gym
import numpy as np
from gym.spaces import Tuple, Box


class Flatten_Action_Space(gym.ActionWrapper):
    """
    flatten the action space (becomes one tuple only)
    """
    def __init__(self, env):
        super(Flatten_Action_Space, self).__init__(env)
        old_action_space = env.action_space
        num_actions = old_action_space.spaces[0].n
        self.action_space = gym.spaces.Tuple((
            old_action_space.spaces[0], *(gym.spaces.Box(old_action_space.spaces[1].spaces[i].low, old_action_space.spaces[1].spaces[i].high, dtype=np.float32)  for i in range(0, num_actions))))

    def action(self, action):
        return action

    
class Normalise_actions(gym.ActionWrapper):
    """
    Normalises the action parameters to [-1,1].
    """

    def __init__(self, env):
        super(Normalise_actions, self).__init__(env)
        
        self.action_dimension = self.action_space.spaces[0].n
        self.upper_bound = [self.action_space.spaces[i].high for i in range(1, self.action_dimension + 1)]
        self.lower_bound = [self.action_space.spaces[i].low for i in range(1, self.action_dimension + 1)]
        self.range = [self.action_space.spaces[i].high - self.action_space.spaces[i].low for i in range(1, self.action_dimension + 1)]
        new_params = [Box(-np.ones(self.action_space.spaces[i].low.shape), np.ones(self.action_space.spaces[i].high.shape),dtype=np.float32) for i in range(1, self.action_dimension + 1)]
        
        self.old_action_space=self.action_space
        
        self.action_space = Tuple((self.old_action_space.spaces[0],*new_params,))

    def action(self, action):
        """
        Rescale from [-1,1] to original action-parameter range.

        :param action:
        :return:
        """
        action = copy.deepcopy(action)
        p = action[0]
        action[1][p] = self.range[p] * (action[1][p] + 1) / 2. + self.lower_bound[p]
        return action
    
class Normalize_state(gym.ObservationWrapper):
    """
    Normalises the observation space to the interval [-1,1]
    """

    def __init__(self, env):
        super(Normalize_state, self).__init__(env)
        obs = env.observation_space
        self.lower_bound = obs.spaces[0].low
        self.upper_bound = obs.spaces[0].high
        self.observation_space = Tuple((gym.spaces.Box(low=-np.ones(self.lower_bound.shape), high=np.ones(self.upper_bound.shape), dtype=np.float32), obs.spaces[1]))

    def scale_state(self, state):
        state = 2. * (state - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.
        return state

    def _unscale_state(self, scaled_state):
        state = (self.upper_bound - self.lower_bound) * (scaled_state + 1.) / 2. + self.lower_bound
        return state

    def observation(self, obs):
        state, steps = obs
        ret = (self.scale_state(state), steps)
        return ret

