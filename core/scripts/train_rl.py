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


def save_custom(obj, path):
    pass


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    # Parse arguments
    parser = ArgumentParser()

    args = parser.parse_args()

    # # run with "python3 ../scripts/train_rl.py" to generate default arguments in a csv file
    # with open('arguments_default.csv', 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['argument', 'default value'])
    #     for k, v in vars(args).items():
    #         writer.writerow([k, v])

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

        # 'BabyAI-MiniBossLevel-v0': [],
        'BabyAI-BossLevel-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],

        'BabyAI-SynthS5R2-v0': [

            'put the red ball next to the green key',
            'put the purple box next to the yellow ball',
            'put the blue key next to the grey box',

            'go to the red box',
            'go to the green ball',

            'pick up the purple key',
            'pick up a purple key',
            'pick up the yellow box',
            'pick up a yellow box',

            'open the blue door',
            'open the grey door',

        ],
        # 'BabyAI-SynthLoc-v0': [],

        'BabyAI-PutNextLocal-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],
        'BabyAI-PutNextLocalS6N4-v0': ['red box', 'green ball', 'purple key', 'yellow box', 'blue ball', 'grey key'],

        # 'BabyAI-MoveTwoAcrossS5N2-v0': [],

        # 'BabyAI-KeyCorridorS5R3-v0': [],

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
        prev_args = torch.load(f'models/{continue_pretrained}/args.pkl')
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

    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = args.instr_arch if args.instr_arch else "noinstr"
    mem = "mem" if not args.no_mem else "nomem"
    
    model_name_parts = {
        'model': args.model,
        'env': args.env,
        'algo': args.algo,
        'arch': args.arch,
        'instr': instr,
        'mem': mem,
        'seed': args.seed,
        'start_time': suffix}
    default_model_name = "{model}_{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}_{start_time}".format(**model_name_parts)
    if args.pretrained_model:
        default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
    args.model = default_model_name

    s = utils.get_log_dir(args.model)
    utils.configure_logging(utils.get_log_dir(args.model))
    logger = logging.getLogger(__name__)

    # Define obss preprocessor
    if 'emb' in args.arch:
        obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
    else:
        obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)

    # Define actor-critic model
    acmodel = None
    if args.load_model:
        acmodel = utils.load_model(args.model, raise_not_found=False)
    c = 0
    verb_ids = [obss_preprocessor.instr_preproc.vocab[v] for v in ['open', 'pick', 'go', 'put']]
    if acmodel is None:
        if args.pretrained_model:
            acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
        else:
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
            # print(acmodel)
            # c = sum([count_parameters(mod) for mod in [acmodel.mha_head, acmodel.embed_K, acmodel.embed_V, acmodel.layer_norm_1, acmodel.layer_norm_2, acmodel.FFN]])
            # print(count_parameters(acmodel.image_conv)+count_parameters(acmodel.controllers[0]), count_parameters(acmodel))
            # raise 'done'

    obss_preprocessor.vocab.save()
    utils.save_model(acmodel, args.model)

    if device == 'cuda':
        acmodel.cuda()
        
    # Define actor-critic algo

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

    # When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
    # Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
    # the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

    utils.seed(args.seed)

    # Restore training status

    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                  'num_episodes': 0,
                  'num_frames': 0}

    # Define logger and Tensorboard writer and CSV writer

    header = (["update", "episodes", "frames", "FPS", "duration"]
              + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["success_rate"]
              + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
    if args.tb:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(utils.get_log_dir(args.model))
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Log code state, command, availability of CUDA and model

    babyai_code = list(babyai.__path__)[0]
    try:
        last_commit = subprocess.check_output(
            'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
        logger.info('LAST COMMIT INFO:')
        logger.info(last_commit)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    try:
        diff = subprocess.check_output(
            'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
        if diff:
            logger.info('GIT DIFF:')
            logger.info(diff)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    logger.info('COMMAND LINE ARGS:')
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(acmodel)

    # Train model

    total_start_time = time.time()
    best_success_rate = 0
    test_env_name = args.env if args.test_env is None else args.test_env

    utils.save_obj(args, args.model, 'args.pkl')

    while status['num_frames'] < args.frames:
        # Update parameters

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs
        if status['i'] % args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"]]

            format_str = (
                    f"{args.env} {args.use_compositional_split} {args.frames} {args.instr_to_rule_mode}, {args.instr_arch}, {args.num_rules}, {args.rule_codebook_size} " + "U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                                                                                                                                                                 "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                                                                                                                                                                 "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")
            logger.info(format_str.format(*data))
            if args.tb:
                assert len(header) == len(data)
                for key, value in zip(header, data):
                    writer.add_scalar(key, float(value), status['num_frames'])

            csv_writer.writerow(data)

        # Save obss preprocessor vocabulary and model
        x = utils.get_model_dir(args.model)
        x = utils.get_log_dir(args.model)
        if args.save_interval > 0 and status['i'] % args.save_interval == 0:
            obss_preprocessor.vocab.save()
            with open(status_path, 'w') as dst:
                json.dump(status, dst)
                utils.save_model(acmodel, args.model)

            # Testing the model before saving
            agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
            agent.model = acmodel
            agent.model.eval()
            logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes,
                                  use_compositional_split=args.use_compositional_split,
                                  penv_leftout_seeds=algo.env.leftout_seeds,
                                  return_obss_actions=True,
                                  compositional_test_splits=compositional_test_splits[args.env])

            agent.model.train()

            mean_return = np.mean(logs["return_per_episode"])
            success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])

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

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                utils.save_model(acmodel, args.model + '_best')
                obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
                logger.info("Return {: .2f}; best model is saved".format(mean_return))
                logger.info("SR {: .2f}; best model is saved".format(success_rate))
                logger.info("C_A {: .2f}; best model is saved".format(collapse_avg))
                logger.info("C_W {: .2f}; best model is saved".format(collapse_w))
            else:
                logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
                logger.info("SR {: .2f}; not the best model; not saved".format(success_rate))
                logger.info("C_A {: .2f}; not the best model; not saved".format(collapse_avg))
                logger.info("C_W {: .2f}; not the best model; not saved".format(collapse_w))


if __name__ == '__main__':
    main()
