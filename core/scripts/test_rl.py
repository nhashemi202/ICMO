#!/usr/bin/env python3

"""
Script to train the agent through reinforcement learning.
"""

import os
import logging
import csv
import json
import gym
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

root = Path(__file__).absolute().parent.parent
log_dir = os.path.join(root.parent, 'logs')
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

import time

os.environ['BABYAI_STORAGE'] = log_dir
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main():

    # Parse arguments
    parser = ArgumentParser()

    args = parser.parse_args()

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

    server = args.server

    if server == '224':
        log_dir = os.path.join(root, 'run/logs_224/logs')
        model_dir = os.path.join(root, 'run/logs_224/logs/models')
    if server == '229':
        log_dir = os.path.join(root, 'run/logs_229/logs')
        model_dir = os.path.join(root, 'run/logs_229/logs/models')
    if server == '230':
        log_dir = os.path.join(root, 'run/logs_230/logs')
        model_dir = os.path.join(root, 'run/logs_230/logs/models')
    if server == '248':
        log_dir = os.path.join(root, 'run/logs_248/logs')
        model_dir = os.path.join(root, 'run/logs_248/logs/models')
    if server == '224-v1':
        log_dir = os.path.join(root, 'run/logs_224_v1/logs')
        model_dir = os.path.join(root, 'run/logs_224_v1/logs/models')
    if server == '176' or server == '176-v1':
        log_dir = os.path.join(root, 'run/logs_176/logs')
        model_dir = os.path.join(root, 'run/logs_176/logs/models')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    continue_pretrained = args.continue_pretrained
    use_pretrained = not continue_pretrained is None
    if use_pretrained:
        prev_args = torch.load(log_dir + f'/models/{continue_pretrained}/args.pkl')
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

    print(args.model, args.ignore_rules)

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
                    ignore_rules=args.ignore_rules,
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

    vocab = json.load(open(log_dir + f'/models/{continue_pretrained}/vocab.json'))
    id2vocab = {k: v for v, k in vocab.items()}

    print(vocab, id2vocab, id2vocab[1])

    test_env_name = args.env if args.test_env is None else args.test_env

    agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
    agent.model = acmodel
    agent.model.eval()

    print('evaluating ....')

    rules_info = []
    missions_info = []

    for i in range(args.num_rules):
        agent.model.ignore_rules = [i]
        logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes,
                              use_compositional_split=args.use_compositional_split,
                              penv_leftout_seeds=algo.env.leftout_seeds,
                              return_obss_actions=True,
                              compositional_test_splits=compositional_test_splits[args.env])
        rule_ids = torch.cat([i for j in logs['rule_ids_per_episode'] for i in j], dim=0)
        obs = torch.cat([j.image for j in logs['obs_per_episode']], dim=0)
        missions = [j.instr for j in logs['obs_per_episode']]

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
            if split_obs is None:
                print("oops")
            for s, slot in enumerate(current_obs):
                entity = code2name[f'{int(slot[1])}-{int(slot[0])}']
                info.append((entity, rule_ids[j][s].item(), instr, split_obs))
                # print(entity, int(rule_ids[j][s]), instr, split_obs, '=================')
        print("OKKKKKKKKKKKKKKKKKKKKKKKKKKKK")

        rules_info.append(info)
        missions_info.append((missions_unique, logs['return_per_episode'], np.array([1 if r > 0 else 0 for r in logs['return_per_episode']])))
        print(len(missions_unique))


        print(info[:10])

    with open(log_dir + f'rules_info_{args.continue_pretrained}.pkl', 'wb') as handle: 
        pickle.dump(rules_info, handle, protocol=pickle.HIGHEST_PROTOCOL) 



if __name__ == '__main__':
    main()
