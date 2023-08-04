#!/usr/bin/env python3

"""
Script to train the agent through reinforcement learning.
"""

import gym
import torch
import numpy as np

import sys
root_project_dir = os.path.join(Path(__file__).parent.parent, 'models')

sys.path.insert(0, os.path.join(root_project_dir, 'NPSUtils'))
sys.path.insert(2, os.path.join(root_project_dir, 'VQ'))
sys.path.insert(3, os.path.join(root_project_dir, 'slot_attention'))
sys.path.insert(3, os.path.join(root_project_dir, 'Recurrent-Independent-Mechanisms'))

import babyai.utils as utils
from core.scripts.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import visualize
from babyai.utils.agent import ModelAgent
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")

# NPS params
                                
parser.add_argument("--in_dim", type=int, default=3)                                
parser.add_argument("--hidden_dim", type=int, default=3)                                
parser.add_argument("--num_rules", type=int, default=8)                                
parser.add_argument("--rule_dim", type=int, default=64)                                
parser.add_argument("--query_dim", type=int, default=32)                                
parser.add_argument("--value_dim", type=int, default=32)                                
parser.add_argument("--key_dim", type=int, default=32)                                  
parser.add_argument("--act_dim", type=int, default=1)                                
parser.add_argument("--num_heads", type=int, default=4)                                
parser.add_argument("--dropout", type=float, default=0.1)                                
parser.add_argument("--num_contexts", type=int, default=5)                                
parser.add_argument("--num_variables", type=int, default=49)                                
parser.add_argument("--apply_mission", action='store_true')                            
parser.add_argument("--use_all_slots", action='store_true')                            
parser.add_argument("--flag", type=str, default='More info on the current run') 
parser.add_argument("--bottleneck_size", type=int, default=0)
parser.add_argument("--use_slot_rnn", action='store_true')
parser.add_argument("--append_coord", action='store_true')
parser.add_argument("--test_env", type=str, default=None)
parser.add_argument("--NS_loss", action='store_true')   
parser.add_argument("--instr_to_rule_mode", type=str, default='conv1d')                             
parser.add_argument("--rule_codebook_size", type=int, default=15)  
parser.add_argument("--continue_pretrained", type=str, default=None)                           
parser.add_argument("--film_d", type=int, default=128) 
parser.add_argument("--use_large_model", action='store_true') 
parser.add_argument("--use_slot_attention", action='store_true') 
parser.add_argument("--use_compositional_split", action='store_true')   
parser.add_argument("--compositional_step", type=int, default=1)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--use_rim", action='store_true')
parser.add_argument("--rim_num_units", type=int, default=5)
parser.add_argument("--rim_k", type=int, default=3)
parser.add_argument("--rim_slowness_factor", type=int, default=4)
parser.add_argument("--concat_instr_to_mem", action='store_true')
parser.add_argument("--free_rule_param", action='store_true') 

current_args = parser.parse_args()


compositional_test_splits = {
    'BabyAI-GoToLocal-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],
    'BabyAI-OpenTwoDoors-v0': [
        'open the blue door, then open the yellow door',
        'open the green door, then open the grey door',
        'open the grey door, then open the red door',
        'open the yellow door, then open the purple door',
        'open the red door, then open the green door',
        'open the purple door, then open the blue door'
    ],
    'BabyAI-BossLevel-v0': [],
}

args = torch.load(f'./models/{current_args.model_path}/args.pkl')

for a in vars(current_args):
    if not a in vars(args):
        args.__dict__.update({a: vars(current_args)[a]})

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(100 * args.seed + i)
    envs.append(env)

if 'emb' in args.arch and False:
    obss_preprocessor = utils.IntObssPreprocessor(current_args.model_path, 
        envs[0].observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(current_args.model_path, 
        envs[0].observation_space, args.pretrained_model)

test_env_name = args.env if args.test_env is None else args.test_env
m = torch.load(f'./models/{current_args.model_path}/model.pt').to(device)
acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                  image_dim=args.image_dim, memory_dim=args.memory_dim, instr_dim=args.instr_dim,
                  use_instr=not args.no_instr, lang_model=args.instr_arch, use_memory=not args.no_mem, arch=args.arch,
                  NPS_info=dict(
                        in_dim=args.in_dim, hidden_dim=args.hidden_dim, 
                        num_rules=args.num_rules, rule_dim=args.rule_dim if args.no_instr else args.instr_dim,
                        query_dim=args.query_dim, value_dim=args.value_dim,
                        key_dim=args.key_dim, act_dim=args.act_dim,
                        num_heads=args.num_heads, dropout=args.dropout, 
                        num_contexts = args.num_contexts,
                        num_variables=args.num_variables, 
                        apply_mission=args.apply_mission,
                        device=device,
                        use_all_slots=args.use_all_slots,
                        compositional_step=args.compositional_step,
                        use_slot_rnn=args.use_slot_rnn,
                        bottleneck_size=args.bottleneck_size if args.bottleneck_size > 0 else None,
                        append_coord=args.append_coord,
                        instr_to_rule_mode=args.instr_to_rule_mode
                    ), rule_codebook_size=args.rule_codebook_size).to(device)
acmodel.__dict__ = m.__dict__

agent = ModelAgent(current_args.model_path, obss_preprocessor, argmax=True)
agent.model = acmodel
agent.model.eval()
logs = visualize(agent, gym.make(test_env_name), device, args.num_rules)
mean_return = np.mean(logs["return_per_episode"])
success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
rule_ids = logs['rule_ids_per_episode'][0] # num_steps_ep*compositional_steps*49

T = len(logs['observations_per_episode'][0])
all_obs = np.concatenate([logs['observations_per_episode'][0][i]['image'].reshape(-1, 3) for i in range(T)], axis=0)
all_obs = all_obs[:, 0] * 100 + all_obs[:, 1] * 10 + all_obs[:, 2]
all_rules = np.concatenate([[[rule_ids[i][j][k] for j in range(args.compositional_step)] for k in range(49)] for i in range(T)], axis=1).reshape(-1, args.compositional_step)
all_actions = torch.tensor([logs['actions_per_episode'][0][i].item() for i in range(T)]).unsqueeze(-1).repeat(1, 49).reshape(-1).numpy()

################################# plot animated m_ts ##################################
code2name = {
    '0-0': 'unseen',
    '5-2': 'obstacle',
    '0-1': 'empty',
} 
for p in [(4, 'door'), (5, 'key'), (6, 'ball'), (7, 'box')]:
    for q in [(0, 'red'), (1, 'green'), (2, 'blue'), (3, 'purple'), (4, 'yellow'), (5, 'grey')]:
        code2name[f'{q[0]}-{p[0]}'] = f'{q[1]} {p[1]}'

m_ts = logs['m_ts_per_episode'][0]
obs_episode = logs['observations_per_episode'][0]
step = 0 # which compositional step to plot: len(m_ts[0][0])

# create the figure and axes objects
fig, ax = plt.subplots()
# function that draws each frame of the animation
def animate(i):
    m_t = m_ts[i][step][0].cpu() # m_t is of shape 49 * act_dim (tensor: cuda)
    m_t = torch.nn.functional.softmax(m_t, dim=0)
    obs = obs_episode[i]['image']
    entities = [[code2name[f'{e[1]}-{e[0]}'] for e in row] for row in obs]
    ax.clear()
    sns.heatmap(m_t[:, 0].reshape(7, 7), annot=entities, fmt="", cmap='RdYlGn', linewidths=0.30, ax=ax, cbar=False)
    ax.set_title(f'{logs["mission"]}, step = {i+1}')

# run the animation
anim = FuncAnimation(fig, animate, frames=T, interval=2000, repeat=False)

anim.save('animated_plots.gif', writer='pillow', fps=0.5)

#######################################################################################
'''
ignored_obs = []
ignored_rule = []
for i in range(len(all_obs)):
    if all_obs[i] == 0 or all_obs[i] == 100:
        continue
    else:
        ignored_rule.append(all_rules[i])
        ignored_obs.append(all_obs[i])
ignored_rule = np.stack(ignored_rule, axis=0)

for i in range(args.compositional_step):
    fig = plt.subplots(figsize =(10, 10))
    plt.hist2d(all_actions, all_rules[:, i], bins=[len(np.unique(all_actions)), len(np.unique(all_rules[:, i]))])
    plt.title(f"Action vs. Rule ID (step={i+1})")
    plt.xlabel('action') 
    plt.ylabel('rule') 
    plt.colorbar()
    plt.show()
    plt.savefig(f'./video/current_action_rule_{i+1}.png')

for i in range(args.compositional_step):
    fig = plt.subplots(figsize =(10, 10))
    plt.hist2d(all_obs, all_rules[:, i], bins=[len(np.unique(all_obs)), len(np.unique(all_rules[:, i]))])
    plt.title(f"Entity vs. Rule ID (step={i+1})")
    plt.xlabel('entity') 
    plt.ylabel('rule') 
    plt.colorbar()
    plt.show()
    plt.savefig(f'./video/current_entity_rule_{i+1}.png')

for i in range(args.compositional_step):
    fig = plt.subplots(figsize =(10, 10))
    plt.hist2d(ignored_obs, ignored_rule[:, i], bins=[len(np.unique(all_obs)), len(np.unique(all_rules[i, :]))])
    plt.title(f"Entity vs. Rule ID (step={i+1}")
    plt.xlabel('entity') 
    plt.ylabel('rule') 
    plt.colorbar()
    plt.show()
    plt.savefig(f'./video/current_entity_rule_ig_{i+1}.png')
'''
