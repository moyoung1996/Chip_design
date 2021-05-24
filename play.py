import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path
from env import Env
import numpy as np
import torch
import datetime
sys.path.append(r'E:\Model\EasyRl\codes')
from PPO.agent import PPO
from common.plot import plot_rewards
from common.utils import save_results


SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/' # 生成保存的模型路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # 存储reward的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
    os.mkdir(RESULT_PATH)

class PPOConfig:
    def __init__(self) -> None:
        self.env = Env
        self.algo = 'PPO'
        self.batch_size = 5
        self.gamma = 0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.update_fre = 20  # frequency of agent update
        self.train_eps = 300  # max training episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check gpu
        
def play(cfg,env,agent):
    best_reward = env.reward_range[0]
    rewards= []
    ma_rewards = []  # moving average rewards
    avg_reward = 0
    running_steps = 0
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            print(state_, reward, action)
            running_steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)   # 调整到一个episode再Push一次，并全覆盖
            if running_steps % cfg.update_fre == 0:
                agent.update()
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        avg_reward = np.mean(rewards[-100:])
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(path=SAVED_MODEL_PATH)
        print('Episode:{}/{}, Reward:{:.1f}, avg reward:{:.1f}, Done:{}'.format(i_episode+1,cfg.train_eps,ep_reward,avg_reward,done))
    return rewards, ma_rewards

if __name__ == '__main__':
    cfg = PPOConfig()
    env = cfg.env

    # 需要预处理
    state_dim = env.observation_space.shape[0]
    print(state_dim)

    action_dim = len(env.get_action_space)
    print(action_dim)
    agent = PPO(state_dim, action_dim, cfg)
    rewards, ma_rewards = play(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards, tag="train", algo=cfg.algo, path=RESULT_PATH)