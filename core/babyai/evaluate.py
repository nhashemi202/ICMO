# import numpy as np
# import gym
# from gym.wrappers import Monitor
# import torch

# # Returns the performance of the agent on the environment for a particular number of episodes.
# def evaluate(agent, env, episodes, model_agent=True, offsets=None):
#     # Initialize logs
#     if model_agent:
#         agent.model.eval()
#     logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

#     if offsets:
#         count = 0

#     for i in range(episodes):
#         if offsets:
#             # Ensuring test on seed offsets that generated successful demonstrations
#             while count != offsets[i]:
#                 obs = env.reset()
#                 count += 1

#         obs = env.reset()
#         agent.on_reset()
#         done = False

#         agent.model.reset_hiddens()

#         num_frames = 0
#         returnn = 0
#         obss = []
#         while not done:
#             action = agent.act(obs)['action']
#             obss.append(obs)
#             obs, reward, done, _ = env.step(action)
#             agent.analyze_feedback(reward, done)
#             num_frames += 1
#             returnn += reward


#         logs["observations_per_episode"].append(obss)
#         logs["num_frames_per_episode"].append(num_frames)
#         logs["return_per_episode"].append(returnn)
#     if model_agent:
#         agent.model.train()
#     return logs


# def evaluate_demo_agent(agent, episodes):
#     logs = {"num_frames_per_episode": [], "return_per_episode": []}

#     number_of_demos = len(agent.demos)

#     for demo_id in range(min(number_of_demos, episodes)):
#         logs["num_frames_per_episode"].append(len(agent.demos[demo_id]))

#     return logs


# class ManyEnvs(gym.Env):

#     def __init__(self, envs):
#         self.envs = envs
#         self.done = [False] * len(self.envs)

#     def seed(self, seeds):
#         [env.seed(int(seed)) for seed, env in zip(seeds, self.envs)]

#     def reset(self):
#         many_obs = [env.reset() for env in self.envs]
#         self.done = [False] * len(self.envs)
#         return many_obs

#     def step(self, actions):
#         self.results = [env.step(action) if not done else self.last_results[i]
#                         for i, (env, action, done)
#                         in enumerate(zip(self.envs, actions, self.done))]
#         self.done = [result[2] for result in self.results]
#         self.last_results = self.results
#         return zip(*self.results)

#     def render(self):
#         raise NotImplementedError


# # Returns the performance of the agent on the environment for a particular number of episodes.
# def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=True, 
#     use_compositional_split=False, penv_leftout_seeds=None, compositional_test_splits=None):
#     num_envs = min(256, episodes)

#     #print(use_compositional_split, compositional_test_splits)

#     envs = []
#     for i in range(num_envs):
#         env = gym.make(env_name)
#         envs.append(env)
#     env = ManyEnvs(envs)

#     logs = {
#         "num_frames_per_episode": [],
#         "return_per_episode": [],
#         "observations_per_episode": [],
#         "actions_per_episode": [],
#         "seed_per_episode": [],
#         "rule_ids_per_episode": [],
#         "obs_per_episode": [],
#         "instruction": [],
#         "intrinsic_reward": []
#     }

#     for i in range((episodes + num_envs - 1) // num_envs):
#         if use_compositional_split:
#             n = int(np.ceil(num_envs / len(compositional_test_splits)))
#             seeds = []
#             for k, v in penv_leftout_seeds.items():
#                 seeds.extend(v[:n])
#             seeds = list(np.random.permutation(seeds))[:num_envs]
#         else:
#             seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)

#         env.seed(seeds)

#         many_obs = env.reset()
#         logs['instruction'].extend([many_obs[i]['mission'] for i in range(num_envs)])

#         agent.model.reset_hiddens()

#         cur_num_frames = 0
#         num_frames = np.zeros((num_envs,), dtype='int64')
#         returns = np.zeros((num_envs,))
#         already_done = np.zeros((num_envs,), dtype='bool')
#         if return_obss_actions:
#             obss = [[] for _ in range(num_envs)]
#             actions = [[] for _ in range(num_envs)]
#             intrinsic_rew = [[] for _ in range(num_envs)]
#         rule_ids = []
#         obs_ts = []
#         while (num_frames == 0).any():
#             t = agent.act_batch(many_obs)
#             action = t['action']
#             obs_model = t['obs']
#             print(max(t['intrinsic_rew']))
#             if return_obss_actions:
#                 for _ in range(num_envs):
#                     if not already_done[_]:
#                         obss[_].append(many_obs[_])
#                         actions[_].append(action[_].item())
#                         intrinsic_rew[_].append(t['intrinsic_rew'][_].item())
                        
#             if not already_done[_] and t['rule_ids'] is not None:
#                 rule_ids.append(t['rule_ids'][0])
#                 obs_ts.append(obs_model)
#                 intrinsic_rew.append(t['intrinsic_rew'])
#             many_obs, reward, done, _ = env.step(action)
#             agent.analyze_feedback(reward, done)
#             done = np.array(done)
#             just_done = done & (~already_done)
#             returns += reward * just_done
#             cur_num_frames += 1
#             num_frames[just_done] = cur_num_frames
#             already_done[done] = True

#         logs["num_frames_per_episode"].extend(list(num_frames))
#         logs["return_per_episode"].extend(list(returns))
#         logs["seed_per_episode"].extend(list(seeds))
#         logs["rule_ids_per_episode"].extend(rule_ids)
#         logs["obs_per_episode"].extend(obs_ts)
#         logs['intrinsic_reward'].extend(intrinsic_rew)
#         if return_obss_actions:
#             logs["observations_per_episode"].extend(obss)
#             logs["actions_per_episode"].extend(actions)
#     return logs


# # Visulizations
# def visualize(agent, env, device, n):

#     agent.model.eval()
#     logs = {"num_frames_per_episode": [], "return_per_episode": [], 
#     "rule_ids_per_episode": [], "observations_per_episode": [], "actions_per_episode": [], "m_ts_per_episode": []}
#     env = Monitor(env, './video', force=True)
#     for i in range(1):

#         obs = env.reset()
#         agent.on_reset()
#         done = False

#         agent.model.reset_hiddens()

#         num_frames = 0
#         reward = 0
#         returnn = 0
#         obss = []
#         rule_ids_all = []
#         actions_all = []
#         m_ts = []
#         while not done and reward == 0:
#             t = agent.act(obs) # later add more returned data to `act` method
#             action = t['action']
#             actions_all.append(action)

#             m_ts.append(t['m_ts'])
#             rule_ids = t['rule_ids']
#             obss.append(obs)
#             rule_ids_all.append([[i.item() for i in rule_ids_lst] for rule_ids_lst in rule_ids])
#             obs, reward, done, _ = env.step(action)
#             agent.analyze_feedback(reward, done)
#             num_frames += 1
#             returnn += reward

#             hist = torch.histc(torch.tensor(rule_ids).view(-1).to(device), bins=n, min=0, max=n-1)
#             print(hist, done, reward)
#         env.close()

#         print(obs['mission'])
#         logs['mission'] = obs['mission']
#         logs["observations_per_episode"].append(obss)
#         logs["rule_ids_per_episode"].append(rule_ids_all)
#         logs["num_frames_per_episode"].append(num_frames)
#         logs["return_per_episode"].append(returnn)
#         logs["actions_per_episode"].append(actions_all)
#         logs["m_ts_per_episode"].append(m_ts)
#     return logs


import numpy as np
import gym
from gym.wrappers import Monitor
import torch

# Returns the performance of the agent on the environment for a particular number of episodes.
def evaluate(agent, env, episodes, model_agent=True, offsets=None):
    # Initialize logs
    if model_agent:
        agent.model.eval()
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

    if offsets:
        count = 0

    for i in range(episodes):
        if offsets:
            # Ensuring test on seed offsets that generated successful demonstrations
            while count != offsets[i]:
                obs = env.reset()
                count += 1

        obs = env.reset()
        agent.on_reset()
        done = False

        agent.model.reset_hiddens()

        num_frames = 0
        returnn = 0
        obss = []
        while not done:
            action = agent.act(obs)['action']
            obss.append(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward


        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
    if model_agent:
        agent.model.train()
    return logs


def evaluate_demo_agent(agent, episodes):
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    number_of_demos = len(agent.demos)

    for demo_id in range(min(number_of_demos, episodes)):
        logs["num_frames_per_episode"].append(len(agent.demos[demo_id]))

    return logs


class ManyEnvs(gym.Env):

    def __init__(self, envs):
        self.envs = envs
        self.done = [False] * len(self.envs)

    def seed(self, seeds):
        [env.seed(int(seed)) for seed, env in zip(seeds, self.envs)]

    def reset(self):
        many_obs = [env.reset() for env in self.envs]
        self.done = [False] * len(self.envs)
        return many_obs

    def step(self, actions):
        self.results = [env.step(action) if not done else self.last_results[i]
                        for i, (env, action, done)
                        in enumerate(zip(self.envs, actions, self.done))]
        self.done = [result[2] for result in self.results]
        self.last_results = self.results
        return zip(*self.results)

    def render(self):
        raise NotImplementedError


# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=True, 
    use_compositional_split=False, penv_leftout_seeds=None, compositional_test_splits=None):
    num_envs = min(256, episodes)

    # print(use_compositional_split, compositional_test_splits)

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        envs.append(env)
    env = ManyEnvs(envs)

    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": [],
        "rule_ids_per_episode": [],
        "rewards": []
    }

    for i in range((episodes + num_envs - 1) // num_envs):
        if use_compositional_split:
            n = int(np.ceil(num_envs / len(compositional_test_splits)))
            seeds = []
            for k, v in penv_leftout_seeds.items():
                seeds.extend(v[:n])
            seeds = list(np.random.permutation(seeds))[:num_envs]
        else:
            seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)

        env.seed(seeds)

        many_obs = env.reset()


        agent.model.reset_hiddens()

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')
        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
            rewards = [[] for _ in range(num_envs)]
        rule_ids = []
        while (num_frames == 0).any():
            t = agent.act_batch(many_obs)
            action = t['action']
            
            many_obs, reward, done, _ = env.step(action)

            if return_obss_actions:
                for _ in range(num_envs):
                    if not already_done[_]:
                        obss[_].append(many_obs[_])
                        actions[_].append(action[_].item())
                        rewards[_].append(reward[_])
            if not already_done[_] and t['rule_ids'] is not None:
                rule_ids.append(t['rule_ids'][0])
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        logs["rule_ids_per_episode"].extend(rule_ids)
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)
            logs['rewards'].extend(rewards)
    return logs


# Visulizations
def visualize(agent, env, device, n):

    agent.model.eval()
    logs = {"num_frames_per_episode": [], "return_per_episode": [], 
    "rule_ids_per_episode": [], "observations_per_episode": [], "actions_per_episode": [], "m_ts_per_episode": []}
    env = Monitor(env, './video', force=True)
    for i in range(1):

        obs = env.reset()
        agent.on_reset()
        done = False

        agent.model.reset_hiddens()

        num_frames = 0
        reward = 0
        returnn = 0
        obss = []
        rule_ids_all = []
        actions_all = []
        m_ts = []
        while not done and reward == 0:
            t = agent.act(obs) # later add more returned data to `act` method
            action = t['action']
            actions_all.append(action)

            m_ts.append(t['m_ts'])
            rule_ids = t['rule_ids']
            obss.append(obs)
            rule_ids_all.append([[i.item() for i in rule_ids_lst] for rule_ids_lst in rule_ids])
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward

            hist = torch.histc(torch.tensor(rule_ids).view(-1).to(device), bins=n, min=0, max=n-1)
            print(hist, done, reward)
        env.close()

        print(obs['mission'])
        logs['mission'] = obs['mission']
        logs["observations_per_episode"].append(obss)
        logs["rule_ids_per_episode"].append(rule_ids_all)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
        logs["actions_per_episode"].append(actions_all)
        logs["m_ts_per_episode"].append(m_ts)
    return logs


