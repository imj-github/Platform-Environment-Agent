# Platform-Environment-Agent

This repository develops, trains and evaluates an agent on the Platform gym environment: https://github.com/cycraig/gym-platform
The repository includes four file: 

1. Wrappers.py: The wrapper file that contains three wrappers for normalizing the observation space to [-1,1] as well as flattening and normalizing the action space to [-1,1]. Wrapper classes are commonly used in Gym environements to modify some parts of the environment systematically without the need for directly modifying the original environment (e.g., by inserting sub-classes)

2. RPDQN_agent.py: this is where the reinforcement learning agent is developed for the Platform environment. The training algorithm is Recurrent Parametrized Deep Q-Network (RP-DQN) with Experience Replay which is a modified version of the Parametrized Deep Q-Network (P-DQN) algorithm proposed in the following paper: https://arxiv.org/pdf/1810.06394.pdf

3. main.py: Run this file in the terminal to evaluate the performance of the trained agent. When you run the file `main.py`, the agent specified in `RPDQN_agent.py` interacts with the environment for 20,000 episodes. Then, it performs an evaluation of the agent performance (with no exploration) over 1000 episodes.

4. Example_running.ipynb: (optional) This is a colab notebook that shows an example on how to run the algorithm included in this repository.

# Dependencies
* python 3.9.7
* gym 0.10.5
* numpy 1.20.3
* torch 1.12.1+cpu

# Platform Domain description
The platform domain consists of three platforms and two gaps. The agent starts on the first platform and must pass through the platforms and gaps while avoiding enemies to reach the last platform. There is an enemy on the first and second platforms. An episode reaches an end if the agent reaches the goal (third platform) or when it touches an enemy or falls into a gap. A reward that is normalized to 1 is allocated to the agent based on the distance it travels. 

The figure below shows a snapshot from a pygame window that shows the agent in the platform domain

![platform_img](https://user-images.githubusercontent.com/126400048/222464475-5dd67d14-786f-48c4-8ab2-3bccfafd33f6.PNG)

# Running the algorithm
To run this algorithm, follow the following steps:

1. clone the gym platform environment
```
git clone https://github.com/cycraig/gym-platform
```
2. Install the gym platform environment
```
pip install -e 'gym-platform/[gym-platform]'
```
3. Clone this repository (or download it) using the following command
```
git clone https://github.com/imj-github/Platform-domain-agent.git
```
4. Finally run the `main.py` file for training and evaluating the agent. In this case, you can `cd` into the `Platform-domain-agent` directory and run `main.py` as follows:
```
python main.py
```
Otherwise, you can run the file by writing its full path (for example)
```
python ./Platform-domain-agent/main.py
```
Note that a colab notebook file that shows how these steps were run in order is included in the repository. 

# Performance evaluation
Once the agent is trained, the evaluation round starts and evaluates the performance of the agent over 1000 episodes and displays the obtained average reward.
In average, our reinforcement learning agent allows to reach an average reward of **0.9970407038391408**, which can be obtained by running the files as described in the steps above. The following figure is a plot that shows the rewards obtained for each of the 1000 evaluation episodes (after training the agent):

![evaluation_curve](https://user-images.githubusercontent.com/126400048/222470266-8fc57982-2c3e-4d93-b633-ae843a45b6cc.jpg)

The video below shows a successful sequence of actions taken by the trained agent to reach its goal. As can be seen, the agent hops over the enemies, leaps over the gaps, and reaches the last (third) platform.

https://user-images.githubusercontent.com/126400048/222469367-5ee52454-3ad2-42de-93ca-a8f54171a87d.mp4
