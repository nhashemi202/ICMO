#!/usr/bin/env python3

"""
Script to train the agent through reinforcement learning.
"""

import os
import logging
import csv
import json
import csv
import gym
import matplotlib.pylab as plt
# import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
# from gymnasium.wrappers import PixelObservationWrapper
import datetime
import torch
import numpy as np
import subprocess
import sys
from pathlib import Path
import pickle
import math

root = Path(__file__).absolute().parent.parent

sys.path.insert(0, os.path.join(root, ''))

root_project_dir = os.path.join(root, 'models')
print(root_project_dir)

sys.path.insert(1, os.path.join(root_project_dir, ''))

sys.path.insert(2, os.path.join(root_project_dir, 'NPSUtils'))
sys.path.insert(3, os.path.join(root_project_dir, 'VQ'))
sys.path.insert(4, os.path.join(root_project_dir, 'slot_attention'))
sys.path.insert(5, os.path.join(root_project_dir, 'Recurrent-Independent-Mechanisms'))

import babyai
import babyai.utils as utils
import babyai.rl
from arguments import ArgumentParser
from models.acmodel import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent
from algos.fast_and_slow_ppo import FastAndSlowPPOAlgo
from torch.distributions.categorical import Categorical

import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def do_experiments(args, log_dir, model_dir, ignore_rules):

    # assert args.use_all_slots, f'Sequential NPS not implemented'
    assert not args.use_slot_attention, f'Slot attention not implemented'
    assert args.bottleneck_size == 0, f'Bottleneck not implemented'

    args.num_variables = args.num_variables if args.use_all_slots else 1 # 49 or 1

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
        'BabyAI-OpenDoorsOrderN4-v0': [
            'open the blue door, then open the yellow door',
            'open the green door, then open the grey door',
            'open the grey door, then open the red door',
            'open the yellow door, then open the purple door',
            'open the red door, then open the green door',
            'open the purple door, then open the blue door',

            'open the blue door after you open the yellow door',
            'open the green door after you open the grey door',
            'open the grey door after you open the red door',
            'open the yellow door after you open the purple door',
            'open the red door after you open the green door',
            'open the purple door after you open the blue door'
        ],

        'BabyAI-PutNextLocal-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],
        'BabyAI-PutNextLocalS6N4-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],

        'BabyAI-ActionObjDoor-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key', 'red door'],

        'BabyAI-PickupDist-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],
        'BabyAI-PickupLoc-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],

        'BabyAI-GoToSeq-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],
        'BabyAI-GoToObjMazeS5-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],
        'BabyAI-GoToSeqS5R2-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],

    }

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    continue_pretrained = args.continue_pretrained
    use_pretrained = not continue_pretrained is None
    if use_pretrained:
        prev_args = torch.load(model_dir + f'/{continue_pretrained}/args.pkl')
        for a in vars(args):
            if not a in vars(prev_args):
                prev_args.__dict__.update({a: vars(args)[a]})
        args = prev_args

    args.pretrained_model = continue_pretrained

    utils.seed(args.seed)

    # Generate environments
    envs = []
    for i in range(args.procs):
        env = gym.make(args.env)
        if args.pixel_images:
            env = RGBImgObsWrapper(env)
        else:
            env.seed(seed = 100 * args.seed + i)
        envs.append(env)

    #print(args.model, ignore_rules, args.pretrained_model, continue_pretrained)

    # Define obss preprocessor
    if 'emb' in args.arch:
        obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
    else:
        obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)

    # Define actor-critic model
    acmodel = None
    verb_ids = [obss_preprocessor.instr_preproc.vocab[v] for v in ['open', 'pick', 'go', 'put']]
    acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                  memory_dim=args.memory_dim, instr_dim=args.instr_dim,
                  use_instr=not args.no_instr, lang_model=args.instr_arch,
                  use_memory=not args.no_mem, arch=args.arch, NPS_info=dict(
                    in_dim=args.in_dim, hidden_dim=args.hidden_dim,
                    num_rules=args.num_rules, rule_dim=args.rule_dim if args.no_instr else args.instr_dim,
                    query_dim=args.query_dim, value_dim=args.value_dim,
                    key_dim=args.key_dim, act_dim=args.act_dim,
                    num_heads=args.num_heads, dropout=args.dropout,
                    num_contexts=args.num_contexts,
                    num_variables=args.num_variables,
                    apply_mission=args.apply_mission,
                    device=device,
                    use_all_slots=args.use_all_slots,
                    use_null_slot=args.use_null_slot,
                    fuse_instr_obs=args.fuse_instr_obs,
                    compositional_step=args.compositional_step,
                    compositional_step_by_verbs=args.compositional_step_by_verbs,
                    verb_ids=verb_ids,
                    use_slot_rnn=args.use_slot_rnn,
                    bottleneck_size=args.bottleneck_size if args.bottleneck_size > 0 else None,
                    append_coord=args.append_coord,
                    instr_to_rule_mode=args.instr_to_rule_mode,
                    use_slot_attention=args.use_slot_attention,
                    free_rule_param=args.free_rule_param,
                    sparse_features=args.sparse_features,
                    use_hidden_feedback_contextual=args.use_hidden_feedback_contextual,
                    use_hidden_feedback_rule=args.use_hidden_feedback_rule,
                    ignore_rules=ignore_rules,
            ), rule_codebook_size=args.rule_codebook_size,
                  film_d=args.film_d,
                  use_large_model=args.use_large_model,
                  use_rim=args.use_rim,
                  rim_num_units=args.rim_num_units, rim_k=args.rim_k,
                  concat_instr_to_mem=args.concat_instr_to_mem,
                  use_hidden_feedback=args.use_hidden_feedback_contextual or args.use_hidden_feedback_rule,
                  use_intrinsic_rew=args.use_intrinsic_rew,
                  intrinsic_rew_coef=args.intrinsic_rew_coef,
                  new_rim_impl=args.new_rim_impl,
                  concat_obs_to_AC=args.concat_obs_to_AC,
                  pixel_images=args.pixel_images,
                  intrinsic_rew_mode=args.intrinsic_rew_mode,
                  input_mem_concat_instr=args.input_mem_concat_instr,
                  )
    if args.pretrained_model:
        model = utils.load_model(args.pretrained_model, raise_not_found=True)
        acmodel.load_state_dict(model.state_dict())
        # nps_model = acmodel.image_conv
        # nps_linear = acmodel.NPS_linear
        # nps_model.__dict__.update(model.image_conv.__dict__)
        # nps_model.rule_net.__dict__.update(model.image_conv.rule_net.__dict__)
        # nps_linear.__dict__.update(model.NPS_linear.__dict__)
        # acmodel.__dict__.update(model.__dict__)
        # acmodel.image_conv = nps_model
        # acmodel.NPS_linear = nps_linear
        # acmodel.image_conv.rule_net.use_hidden_feedback_rule = args.use_hidden_feedback_rule
    else:
        raise 'Model not specified.'

    if device == 'cuda':
        acmodel.cuda()

    reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
    if args.algo == "ppo":
        if args.use_rim and args.fast_and_slow_rim:
            algo = FastAndSlowPPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1,
                                      args.beta2,
                                      args.gae_lambda,
                                      args.entropy_coef, args.value_loss_coef, args.max_grad_norm,
                                      args.recurrence,
                                      args.optim_eps, args.clip_eps, args.batch_size, obss_preprocessor,
                                      reshape_reward, NS_loss=args.NS_loss,
                                      use_compositional_split=args.use_compositional_split,
                                      compositional_test_splits=compositional_test_splits[args.env],
                                      rim_slowness_factor=args.rim_slowness_factor,
                                      intrinsic_rew_coef=args.intrinsic_rew_coef,
                                      device=device, new_rim_impl=args.new_rim_impl
                                      )
        else:
            algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1,
                                     args.beta2,
                                     args.gae_lambda,
                                     args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                     args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                                     reshape_reward, NS_loss=args.NS_loss,
                                     use_compositional_split=args.use_compositional_split,
                                     compositional_test_splits=compositional_test_splits[args.env],
                                     device=device)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    utils.seed(args.seed)

    vocab = json.load(open(model_dir + f'/{continue_pretrained}/vocab.json'))
    id2vocab = {k: v for v, k in vocab.items()}

    print(vocab, id2vocab, id2vocab[1])

    test_env_name = args.env if args.test_env is None else args.test_env

    agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
    agent.model = acmodel
    agent.model.eval()

    print('evaluating ....')

    rules_info = []
    missions_info = []
    mean_return_first_experiment = [0 for _ in range(len(ignore_rules))]
    overall_logs = []
    intrinsic_rewards = []
    total_missions = []
    total_entity = []

    print('These Rules will be ignored: ', ignore_rules)
    for i in range(len(ignore_rules) + 1):
        if i == len(ignore_rules):
            agent.model.ignore_rules = []
        else:
            agent.model.ignore_rules = [ignore_rules[i]]

        logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes,
                              use_compositional_split=args.use_compositional_split,
                              penv_leftout_seeds=algo.env.leftout_seeds,
                              return_obss_actions=True,
                              compositional_test_splits=compositional_test_splits[args.env])
        overall_logs.append(logs)
        rule_ids = torch.cat([i for j in logs['rule_ids_per_episode'] for i in j], dim=0)
        obs = torch.cat([j.image for j in logs['obs_per_episode']], dim=0)
        missions = [j.instr for j in logs['obs_per_episode']]
        intrinsic_rewards.append(logs['intrinsic_reward'])

        per_rule_mean_return_first_experiment = [0 for i in range(len(compositional_test_splits[test_env_name]))]
        count_return_first_experiment = [0 for i in range(len(compositional_test_splits[test_env_name]))]
        #print(missions)
        #print(logs['instruction'])
        #total_missions.extend(logs['instruction'])

        if i < len(ignore_rules):
            test_split = compositional_test_splits[test_env_name]
            for k in range(len(test_split)):
                for j in range(len(logs['instruction'])):
                    if test_split[k] in logs['instruction'][j]:
                        per_rule_mean_return_first_experiment[k] += logs['return_per_episode'][j]
                        count_return_first_experiment[k] += 1
            per_rule_mean_return_first_experiment = np.array(per_rule_mean_return_first_experiment) / np.array(count_return_first_experiment)
            mean_return_first_experiment[i] = per_rule_mean_return_first_experiment

        mean_return = np.mean(logs["return_per_episode"])
        success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        print(np.array(logs['return_per_episode']).shape, obs.shape, ' RPE =========================')
        print(np.array(logs['num_frames_per_episode']).shape, obs.shape, ' NF =========================')
        print(missions[0].shape, len(missions))
        print(sum(logs['num_frames_per_episode']))

        collapse_avg = 0
        collapse_w = 0
        if args.arch == 'NPS_vanilla':
            rule_ids = torch.cat([i for j in logs['rule_ids_per_episode'] for i in j], dim=0)
            R = args.num_rules
            p = torch.zeros(R)
            for r in rule_ids:
                p[r] += 1
            p /= len(rule_ids)
            print(p)
            q = [max(0, 1 / R - x) for x in p]
            collapse_avg = R / (R - 1) * sum(q)
            collapse_w = 1 - R * p.min()

        # We run with/without rule elimination here in the last iteration, so I distinguish between ablation and all-rule considered run with this if
        if i < len(ignore_rules):
            print(f'IGNORED RULE {i + 1}')
            print("Return {: .2f}".format(mean_return))
            print("SR {: .2f}".format(success_rate))
            print("C_A {: .2f}".format(collapse_avg))
            print("C_W {: .2f}".format(collapse_w))
            print("----------------------------------------------")

        # Visualize entity vs. test combination per rule per level
        code2name = {
            '0-0': 'unseen',
            '5-2': 'obstacle',
            '0-1': 'empty',
        } 
        for p in [(4, 'door'), (5, 'key'), (6, 'ball'), (7, 'box')]:
            for q in [(0, 'red'), (1, 'green'), (2, 'blue'), (3, 'purple'), (4, 'yellow'), (5, 'grey')]:
                code2name[f'{q[0]}-{p[0]}'] = f'{q[1]} {p[1]}'
        get_intr = lambda id_lst: ' '.join([id2vocab[id.item()] for id in id_lst if id.item() > 0])
        rule_ids = rule_ids.reshape(-1, 49)
        bs = missions[0].shape[0]
        obs = obs.reshape(-1, 49, 3)
        info = []
        missions_unique = []
        for j in range(obs.shape[0]):
            current_obs = obs[j]
            split_obs = None
            m = missions[int(j/bs)][j % bs]
            instr = get_intr(m)
            if not instr in missions_unique:
                missions_unique.append(instr)
            for split in compositional_test_splits[args.env]:
                if split in instr:
                    split_obs = split
                    break
            for s, slot in enumerate(current_obs):
                entity = code2name[f'{int(slot[1])}-{int(slot[0])}']
                info.append((entity, rule_ids[j][s].item(), instr, split_obs))
                total_entity.append(entity)
                total_missions.append(instr)
                # print(entity, int(rule_ids[j][s]), instr, split_obs, '=================')

        rules_info.append(info)
        missions_info.append((missions_unique, logs['return_per_episode'], np.array([1 if r > 0 else 0 for r in logs['return_per_episode']])))
        print(len(missions_unique))

    # Save data
    print('Saving results ....')

    path = os.path.join(log_dir, 'rules_info.pkl')
    print(path)

    with open(path, 'wb') as handle:
        pickle.dump(rules_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path = os.path.join(log_dir, 'intrinsic_rewards.pkl')

    with open(path, 'wb') as handle:
        pickle.dump(intrinsic_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #path = os.path.join(log_dir, 'overall_logs.pkl')

    # with open(path, 'wb') as handle:
    #     pickle.dump(overall_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #

    # Plot Heat Maps

    # Experiment 1
    print('Plot heat maps ....')
    plot_heat_map(mean_return_first_experiment, os.path.join(log_dir, f"ablation.jpg"), test_split, ignore_rules, 'Split', 'Ignored Rule Index', 'Mean Return in the absence of each rule')

    total_entity = list(set(total_entity))
    total_entity.sort()

    total_missions = list(set(total_missions))
    total_missions.sort()
    # Experiment 2
    i = len(ignore_rules)
    result_1 = second_expr_ready(rules_info, i, k=3, collection=test_split, total_entity=total_entity, args=args)
    plot_second_expr(result_1, 'split', x_label=test_split, y_label=total_entity, log_dir=log_dir)

    result_2 = second_expr_ready(rules_info, i, k=2, collection=total_missions, total_entity=total_entity, args=args)
    plot_second_expr(result_2, 'mission', x_label=total_missions, y_label=total_entity, log_dir=log_dir)

    with open(os.path.join(log_dir, 'result_1.pkl'), 'wb') as handle:
        pickle.dump(result_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(log_dir, 'result_2.pkl'), 'wb') as handle:
        pickle.dump(result_2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(log_dir, 'mean_return_first_experiment.pkl'), 'wb') as handle:
        pickle.dump(mean_return_first_experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_second_expr(result, label, x_label, y_label, log_dir):
    tmp = result.sum(dim=-1)
    result_freq = result / tmp.unsqueeze(-1)
    min_ = torch.min(result_freq, dim=-1)[0].unsqueeze(-1)
    max_ = torch.max(result_freq, dim=-1)[0].unsqueeze(-1)
    result_freq = (result_freq - min_) / (max_ - min_)
    fre_res = torch.zeros((result_freq.shape[0], result_freq.shape[1]))
    for i in range(result_freq.shape[0]):
        for j in range(result_freq.shape[1]):
            fre_res[i, j] = Categorical(result_freq[i][j]).entropy()

    max_res = torch.zeros((result_freq.shape[0], result_freq.shape[1]))
    for i in range(result_freq.shape[0]):
        for j in range(result_freq.shape[1]):
            idx = result_freq[i][j].argmax(dim=-1)
            if result_freq[i, j, idx] > 0:
                max_res[i, j] = idx + 1
            else:
                max_res[i, j] = float('nan')

    # print('max index: ', max_res)
    # print('max value', result[:, :, max_res])

    if label == 'mission':
        plot_heat_map(fre_res, os.path.join(log_dir, f"modularity_fre_{label}.jpg"), x_label, y_label, 'Split', 'Entity', 'Modularity based on Entropy of modules activation frequency', figsize=(16, 20), rounding=-1)
    else:
        plot_heat_map(fre_res, os.path.join(log_dir, f"modularity_fre_{label}.jpg"), x_label, y_label, 'Split', 'Entity', 'Modularity based on Entropy of modules activation frequency', figsize=(16, 20))

    plot_heat_map(max_res, os.path.join(log_dir, f"modularity_max_{label}.jpg"), x_label, y_label, 'Split', 'Entity', 'Modularity Based on most activated module', figsize=(16, 20), rounding=0)


def second_expr_ready(rules_info, i, k, collection, total_entity, args):
    # rule_data = dict()
    # for j in range(len(rules_info[i])):
    #     if rules_info[i][j][0] not in rule_data.keys():
    #         rule_data[rules_info[i][j][0]] = {rules_info[i][j][k]: {rules_info[i][j][1]: 1}}
    #     elif rules_info[i][j][k] not in rule_data[rules_info[i][j][0]].keys():
    #         rule_data[rules_info[i][j][0]][rules_info[i][j][k]] = {rules_info[i][j][1]: 1}
    #     elif rules_info[i][j][1] not in rule_data[rules_info[i][j][0]][rules_info[i][j][k]].keys():
    #         rule_data[rules_info[i][j][0]][rules_info[i][j][k]][rules_info[i][j][1]] = 1
    #     else:
    #         rule_data[rules_info[i][j][0]][rules_info[i][j][k]][rules_info[i][j][1]] += 1

    result = torch.zeros(size=(len(total_entity), len(collection), args.num_rules))
    for j in range(len(rules_info[i])):
        a1 = total_entity.index(rules_info[i][j][0])
        a2 = collection.index(rules_info[i][j][k])
        a3 = rules_info[i][j][1]
        result[a1, a2, a3] += 1

    # result = torch.zeros(size=(len(total_entity), len(collection), args.num_rules))
    # for id1, key1 in enumerate(rule_data.keys()):
    #     for _, key2 in enumerate(rule_data[key1].keys()):
    #         id2 = collection.index(key2)
    #         for id3 in range(args.num_rules):
    #             if id3 in rule_data[key1][key2].keys():
    #                 result[id1, id2, id3] = rule_data[key1][key2][id3]
    #             else:
    #                 result[id1, id2, id3] = 0
    return result


def plot_heat_map(data, dir_path, x_ticket, y_ticket, x_label, y_label, title, figsize=(10, 10), rounding=2):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, cmap='viridis')

    plt.colorbar(im)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_ticket)))
    ax.set_yticks(np.arange(len(y_ticket)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(x_ticket)
    ax.set_yticklabels(y_ticket)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    if rounding > 0:
        for i in range(len(y_ticket)):
            for j in range(len(x_ticket)):
                x = data[i][j]
                if torch.is_tensor(x):
                    x = data[i][j].item()

                text = ax.text(j, i, round(x, rounding), ha="center", va="center", color="w")

    elif rounding == 0:
        for i in range(len(y_ticket)):
            for j in range(len(x_ticket)):
                x = data[i][j]
                if torch.is_tensor(x):
                    x = data[i][j].item()
                if math.isnan(x):
                    text = ax.text(j, i, x, ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, int(x), ha="center", va="center", color="w")
    else:
        pass

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(dir_path)
    plt.clf()


def main():
    # plot_heat_map([[0, 0, 0], [0, 0]], 'men.jpg', ['a', 'a', 'a'], ['a', 'a'], '', '', '')
    # return
    file = open('/HDD/nhashemi/NPS/npsrl_final/npsrl/core/run/logs/run_paths.csv')
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    del rows[0]
    servers = list(set([rows[i][0] for i in range(len(rows))]))
    servers = {key: [] for key in servers}

    for data in rows:
        if (data[2] == 'NPS-D-MFC-ConcatAC' or data[2] == 'NPS-D-MFC-ConcatMem' or data[2] == 'NPS-IC-GRU' or data[2] == 'NPS-D-instr-concat') and \
                (data[4] == 'BabyAI-PickupLoc-v0' or data[4] or data[4] == 'BabyAI-ActionObjDoor-v0'):
            servers[data[0]].append([data[1], data[2], data[3], data[4]])

    # Parse arguments
    parser = ArgumentParser()
    args = parser.parse_args()
    ignore_rules = [0, 1, 2, 3, 4, 5, 6, 7]

    log_dir = os.path.join(root, 'run/logs/logs/test_heatmap/test_heatmap')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for key in servers.keys():
        if key == '18':
            model_dir = os.path.join(root, 'run/logs')
        elif key == '224':
            model_dir = os.path.join(root, 'run/logs_224/logs')
        elif key == '229':
            model_dir = os.path.join(root, 'run/logs_229/logs')
        elif key == '230':
            model_dir = os.path.join(root, 'run/logs_230/logs')
        elif key == '248':
            model_dir = os.path.join(root, 'run/logs_248/logs')
        elif key == '224-v1':
            model_dir = os.path.join(root, 'run/logs_224_v1/logs')
        elif key == '224':
            model_dir = os.path.join(root, 'run/logs_224_v1/logs')
        elif key == '176':
            model_dir = os.path.join(root, 'run/logs_176/logs')
        elif key == '176-v1':
            model_dir = os.path.join(root, 'run/logs_176_v1/logs')
        else:
            raise 'There is a problem!'

        os.environ['BABYAI_STORAGE'] = model_dir
        model_dir = os.path.join(model_dir, 'models')

        for model in servers[key]:
            args.continue_pretrained = model[0]
            args.test_env = model[3]
            args.env = model[3]
            args.seed = int(model[2])
            task_log_dir = os.path.join(log_dir, f'{model[1]}_{model[3]}_{model[2]}')
            if not os.path.exists(task_log_dir):
                os.makedirs(task_log_dir)
                print(f'******* Do Experiment on {model[1]} on task: {model[3]} server: {key}')
                print(f'******* log_dir: {task_log_dir} , model_dir: {model_dir}')
                do_experiments(args, log_dir=task_log_dir, model_dir=model_dir, ignore_rules=ignore_rules)
            else:
                pass


if __name__ == '__main__':
    main()
