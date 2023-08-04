import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess
import pickle

import sys

root_project_dir = os.path.join(Path(__file__).parent.parent, 'models')

sys.path.insert(0, os.path.join(root_project_dir, 'NPSUtils'))
import babyai
import babyai.utils as utils
import babyai.rl

import time

env = gym.make('BabyAI-GoToObj-v0')
print(env.reset()['image'].sum(axis=-1))

for i in range(10):
	a = env.action_space.sample()
	obs, _, _, _ = env.step(a)
	print(obs['image'].sum(axis=-1), a, '\n=========================\b')
	time.sleep(5)
