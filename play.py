import env
import agent

EPISODE = 500
MEMORY_CAPACITY = 200

for i_episode in range(EPISODE):

    # 初始化环境
    s = env.reset()

    # reward总和
    ep_r = 0

    while True:

        # 对state进行处理成observation
        o = s

        # agent选择动作
        a = agent.choose_action(o)

        # take action
        s_, r, done, info = env.step(a)

        # 对reward进行处理
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存储过程信息
        agent.store_transition(s, a, r, s_)

        # 计算总reward
        ep_r += r
        # print('reward: ', r)

        if agent.memory_counter > MEMORY_CAPACITY:
            agent.learn()
            if done:
                print('Ep: ', i_episode, '| Ep_r:', round(ep_r, 2))

        if done:
            break

        s = s_