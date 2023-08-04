import os
import csv
import sys
from pathlib import Path
import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
import json
from collections import defaultdict


def get_test_stats(test_log_dir):
	test_SRs = []
	test_MRs = []
	test_CAs = []
	test_CWs = []
	test_best = 0
	with open(test_log_dir, 'r') as f:
		f = f.readlines()
		cnt = 0
		for line in f:
			val = line.split(';')[0][-4:]
			try:
				val = float(val)
			except ValueError:
				continue
			if 'SR' in line:
				test_SRs.append(val)
				if '; best model' in line:
					test_best = cnt
				cnt += 1
			if 'Return' in line:
				test_MRs.append(val)
			if 'C_A' in line:
				test_CAs.append(val)
			if 'C_W' in line:
				test_CWs.append(val)
	return test_SRs, test_MRs, test_CAs, test_CWs, test_best


modes = {
'NPS-IC-GRU-free-rule-param': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 
				use_hidden_feedback_contextual=False, 
				free_rule_param=True, 
				concat_instr_to_mem=False, 
				no_instr=False),
'NPS-IC-MHA': dict(arch='NPS_vanilla',
		instr_arch='MHA', 
		instr_to_rule_mode='MHA', 
		concat_obs_to_AC=False, 
		fuse_instr_obs=False, 
		use_intrinsic_rew=False, 
		use_hidden_feedback_contextual=False, 
		free_rule_param=False, 
		concat_instr_to_mem=False, 
		no_instr=False),
'NPS-IC-GRU-memory-feedback': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 
				use_hidden_feedback_contextual=True, 
				free_rule_param=False, 
				concat_instr_to_mem=False, 
				no_instr=False),
'CNN-instr-concat': dict(arch='cnn1',
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=True, 
			no_instr=True),
'NPS-D-instr-concat': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=True, 
			no_instr=True),
'FiLM': dict(arch='expert_filmcnn',
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=False, 
			no_instr=False),
'NPS-IC-GRU': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=False, 
			no_instr=False),
'NPS-IC-GRU-merge-instr-with-obs': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=True, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=False, 
			no_instr=False),
'NPS-IC-GRU-intrinsic-reward': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=True, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=False, 
			no_instr=False),
'NPS-IC-GRU-concat-obs-to-AC': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=True, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=False, 
			no_instr=False),
'NPS-D-instr-concat-normalized': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=True, 
			no_instr=True), 
'NPS-IC-GRU-instr-concat-normalized': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=True,
			no_instr=False), 
'NPS-IC-GRU-instr-concat': dict(arch='NPS_vanilla',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=True,
			no_instr=False),
'Fusion': dict(arch='fusion',
			instr_arch='gru', 
			instr_to_rule_mode='linear', 
			concat_obs_to_AC=False, 
			fuse_instr_obs=False, 
			use_intrinsic_rew=False, 
			use_hidden_feedback_contextual=False, 
			free_rule_param=False, 
			concat_instr_to_mem=False,
			no_instr=False),
'NPS-IC-MHA-concat-instr': dict(arch='NPS_vanilla',
		instr_arch='MHA', 
		instr_to_rule_mode='MHA', 
		concat_obs_to_AC=False, 
		fuse_instr_obs=False, 
		use_intrinsic_rew=False, 
		use_hidden_feedback_contextual=False, 
		free_rule_param=False, 
		concat_instr_to_mem=True, 
		no_instr=False),
'Raw': dict(arch='raw',
		instr_arch='gru', 
		instr_to_rule_mode='linear', 
		concat_obs_to_AC=False, 
		fuse_instr_obs=False, 
		use_intrinsic_rew=False, 
		use_hidden_feedback_contextual=False, 
		free_rule_param=False, 
		concat_instr_to_mem=True, 
		no_instr=False),

'NPS-D-MFC-ConcatAC': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=True, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=True, 
				input_mem_concat_instr=False,

				no_instr=True),
'NPS-D-MFC-ConcatMem': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=True, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=False, 
				input_mem_concat_instr=True,

				no_instr=True),
'NPS-D-MFR-ConcatAC': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=False, 
				use_hidden_feedback_rule=True,

				free_rule_param=False, 

				concat_instr_to_mem=True, 
				input_mem_concat_instr=False,

				no_instr=True),
'NPS-D-MFR-ConcatMem': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=False, 
				use_hidden_feedback_rule=True,

				free_rule_param=False, 

				concat_instr_to_mem=False, 
				input_mem_concat_instr=True,

				no_instr=True),


'NPS-D-ConcatAC': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=False, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=True, 
				input_mem_concat_instr=False,

				no_instr=True),
'NPS-D-ConcatMem': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=False, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=False, 
				input_mem_concat_instr=True,

				no_instr=True),

'NPS-D-IR': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=True, 
				intrinsic_rew_mode='entropy-count-obs',


				use_hidden_feedback_contextual=False, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=True, 
				input_mem_concat_instr=False,

				no_instr=True),
'NPS-D-MFC-ConcatMem-IR': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=True, 
				intrinsic_rew_mode='entropy-count-obs',


				use_hidden_feedback_contextual=True, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=False, 
				input_mem_concat_instr=True,

				no_instr=True),

'NPS-IC-GRU-IR': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=True, 
				intrinsic_rew_mode='entropy-count-obs',


				use_hidden_feedback_contextual=False, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=False, 
				input_mem_concat_instr=False,

				no_instr=False),

'NPS-D-MFC-ConcatMemAC': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=True, 
				use_hidden_feedback_rule=False,

				free_rule_param=False, 

				concat_instr_to_mem=True, 
				input_mem_concat_instr=True,

				no_instr=True), 

'NPS-D-MFCR-ConcatMem': dict(arch='NPS_vanilla',
				instr_arch='gru', 
				instr_to_rule_mode='linear', 
				concat_obs_to_AC=False, 
				fuse_instr_obs=False, 
				use_intrinsic_rew=False, 

				use_hidden_feedback_contextual=True, 
				use_hidden_feedback_rule=True,

				free_rule_param=False, 

				concat_instr_to_mem=False, 
				input_mem_concat_instr=True,

				no_instr=True), 
}

filters = dict()
filters['18'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['NPS-IC-GRU-free-rule-param', 'NPS-IC-MHA', 'Fusion', 'Raw', 'NPS-IC-GRU-IR'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-IC-GRU-free-rule-param', 'NPS-IC-GRU-memory-feedback', 'NPS-IC-MHA', 'Fusion', 'NPS-IC-MHA-concat-instr', 'Raw', 'NPS-IC-GRU-IR'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=['CNN-instr-concat', 'NPS-D-instr-concat', 'NPS-IC-GRU-free-rule-param', 
		'NPS-IC-GRU-memory-feedback', 'NPS-IC-MHA', 'NPS-IC-MHA-concat-instr', 'Raw', 'NPS-D-MFC-ConcatMem-IR'],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['NPS-IC-MHA', 'NPS-IC-MHA-concat-instr', 'Raw', 'NPS-IC-GRU-IR']
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=['NPS-IC-GRU-free-rule-param', 'NPS-IC-MHA', 'Fusion', 'Raw', 'NPS-D-MFC-ConcatMem']
	),
}
filters['176'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['FiLM', 'NPS-D-instr-concat', 'NPS-IC-GRU-intrinsic-reward'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['CNN-instr-concat', 'FiLM', 'NPS-D-instr-concat', 'NPS-IC-GRU-intrinsic-reward'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=['FiLM', 'NPS-IC-GRU-intrinsic-reward', 'Fusion'],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['CNN-instr-concat', 'FiLM', 'NPS-D-instr-concat', 'NPS-IC-GRU-memory-feedback', 'NPS-IC-GRU-intrinsic-reward', 'Fusion']
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=['FiLM', 'NPS-IC-GRU-intrinsic-reward']
	),
}
filters['224'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['CNN-instr-concat', 'NPS-IC-GRU', 'NPS-IC-GRU-memory-feedback', 'NPS-IC-GRU-merge-instr-with-obs', 'NPS-IC-GRU-concat-obs-to-AC'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-IC-GRU', 'NPS-IC-GRU-merge-instr-with-obs', 'NPS-IC-GRU-concat-obs-to-AC'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=['NPS-IC-GRU', 'NPS-IC-GRU-merge-instr-with-obs', 'NPS-IC-GRU-concat-obs-to-AC'],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['NPS-IC-GRU', 'NPS-IC-GRU-free-rule-param', 'NPS-IC-GRU-merge-instr-with-obs', 'NPS-IC-GRU-concat-obs-to-AC']
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=['CNN-instr-concat', 'NPS-D-instr-concat', 'NPS-IC-GRU', 'NPS-IC-GRU-memory-feedback', 'NPS-IC-GRU-merge-instr-with-obs', 'NPS-IC-GRU-concat-obs-to-AC']
	),
}
filters['224-v1'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['NPS-D-instr-concat-normalized', 'NPS-IC-GRU-instr-concat-normalized', 'NPS-IC-GRU-instr-concat'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-IC-GRU-instr-concat-normalized', 'NPS-IC-GRU-instr-concat'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=['NPS-IC-GRU-instr-concat-normalized', 'NPS-IC-GRU-instr-concat'],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['NPS-IC-GRU-instr-concat-normalized', 'NPS-IC-GRU-instr-concat'],
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=['NPS-IC-GRU-instr-concat-normalized', 'NPS-IC-GRU-instr-concat'],
	),
}
filters['176-v1'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=[],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-D-instr-concat-normalized'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=['NPS-D-instr-concat-normalized'],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['NPS-D-instr-concat-normalized']
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=['NPS-D-instr-concat-normalized']
	),
}
filters['229'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['NPS-D-MFR-ConcatAC', 'NPS-D-MFC-ConcatAC'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-D-ConcatMem'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=['NPS-D-MFC-ConcatMem'],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['NPS-D-MFC-ConcatAC', 'NPS-D-IR', 'NPS-D-ConcatMem']
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=[]
	),
}
filters['230'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['NPS-D-IR'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-D-IR'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=[],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['NPS-D-MFR-ConcatAC', 'NPS-D-MFR-ConcatMem', 'NPS-D-MFC-ConcatMem', 'NPS-D-MFC-ConcatMem-IR']
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=[]
	),
}
filters['248'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['NPS-D-MFR-ConcatMem', 'NPS-D-MFC-ConcatMem', 'NPS-D-ConcatMem', 'NPS-D-MFC-ConcatMem-IR'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-D-MFR-ConcatAC', 'NPS-D-MFC-ConcatAC', 'NPS-D-MFR-ConcatMem', 'NPS-D-MFC-ConcatMem', 'NPS-D-MFC-ConcatMem-IR'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=[],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=[]
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=[]
	),
}
filters['112'] = {
	'BabyAI-ActionObjDoor-v0': dict(
		thresh=7.6e6,
		runs=['NPS-D-MFCR-ConcatMem', 'NPS-D-MFC-ConcatMemAC'],
	),
	'BabyAI-GoToSeqS5R2-v0': dict(
		thresh=20e6,
		runs=['NPS-D-MFCR-ConcatMem', 'NPS-D-MFC-ConcatMemAC'],
	),
	'BabyAI-PutNextLocalS6N4-v0': dict(
		thresh=25e6,
		runs=[],
	),
	'BabyAI-PickupLoc-v0': dict(
		thresh=39e6,
		runs=['NPS-D-MFC-ConcatMemAC', 'NPS-D-MFCR-ConcatMem']
	),
	'BabyAI-OpenDoorsOrderN4-v0': dict( 
		thresh=7.6e6,
		runs=[]
	),
}


root = Path(__file__).absolute().parent.parent

seeds = {e: defaultdict(list) for e in filters['18'].keys()}

run_paths = []

for server in ['18', '224', '176', '224-v1', '176-v1', '230', '229', '248', '112']: #
	log_dir = os.path.join(root, 'run/logs/logs/')
	model_dir = os.path.join(root, 'run/logs/models/')

	if server == '224':
		log_dir = os.path.join(root, 'run/logs_224/logs/logs/')
		model_dir = os.path.join(root, 'run/logs_224/logs/models/')
	if server == '229':
		log_dir = os.path.join(root, 'run/logs_229/logs/logs/')
		model_dir = os.path.join(root, 'run/logs_229/logs/models/')
	if server == '112':
		log_dir = os.path.join(root, 'run/logs_112/logs/logs/')
		model_dir = os.path.join(root, 'run/logs_112/logs/models/')
	if server == '230':
		log_dir = os.path.join(root, 'run/logs_230/logs/logs/')
		model_dir = os.path.join(root, 'run/logs_230/logs/models/')
	if server == '248':
		log_dir = os.path.join(root, 'run/logs_248/logs/logs/')
		model_dir = os.path.join(root, 'run/logs_248/logs/models/')
	if server == '224-v1':
		log_dir = os.path.join(root, 'run/logs_224_v1/logs/logs/')
		model_dir = os.path.join(root, 'run/logs_224_v1/logs/models/')
	if server == '176' or server == '176-v1':
		log_dir = os.path.join(root, 'run/logs_176/logs/logs/')
		model_dir = os.path.join(root, 'run/logs_176/logs/models/')

	# print(log_dir, model_dir)
	all_logs = glob.glob(os.path.join(log_dir, 'ABLATION*/'))

	for p in all_logs:
		logcsv = os.path.join(p, 'log.csv')
		if os.path.exists(logcsv):
			with open(logcsv, 'r') as f:
				reader = csv.reader(f)
				reader = csv.reader(x.replace('\0', '') for x in f)
				row_count = sum(1 for row in reader) * 12800
				if row_count >= 7.6e6:
					ids = p.split('/')[-2]
					try:
						args = torch.load(os.path.join(model_dir, ids, 'args.pkl'))
						# model = torch.load(os.path.join(model_dir, ids, 'model.pt'))
					except Exception as e:
						# print(e)
						continue
					args.model = 'placeholder'
					env = args.env
					target_filters = filters[server][env]
					thresh = target_filters['thresh']
					runs = target_filters['runs']
					for run in runs:
						# print(run)
						# print([(vars(args)[k], modes[run][k]) for k in modes[run] if k in vars(args)])
						if row_count >= thresh and args.use_compositional_split == True and args.use_all_slots == True and \
							all([vars(args)[k] == modes[run][k] for k in modes[run] if k in vars(args)]):
							test_log_dir = os.path.join(log_dir, ids, 'log.log')
							try:
								test_SRs, test_MRs, test_CAs, test_CWs, test_best = get_test_stats(test_log_dir)
							except ValueError:
								continue
							df = pd.read_csv(logcsv)
							# separate normalized instruction concatenations (no args specified!)
							if run == 'NPS-D-instr-concat' or run == 'NPS-IC-GRU-instr-concat':
								if any([cond in 'ids' for cond in ['23_04_06', '23_04_07', '23_04_08', '23_04_09']]) :
									run += '-normalized'
							if 'NPS-D-ConcatMem' in run:
								print(run, env, args.seed, args.no_instr, server)

							run_paths.append([server, ids, run, args.seed, env])

							seeds[env][run].append(dict(ids=ids, seed=args.seed,
								train_reports=dict(SR=list(df['success_rate']), frames=list(df['frames']), 
									episode_length=list(df['num_frames_mean']), episodes=list(df['episodes']), 
									return_mean=list(df['return_mean']), return_std=list(df['return_std'])), 
								test_SRs=test_SRs, test_MRs=test_MRs, test_CAs=test_CAs, test_CWs=test_CWs, test_best=test_best))


log_dir = os.path.join(root, 'run/logs/')
with open(os.path.join(log_dir, f'server_all_results.json'), 'w') as f:
    json.dump(seeds, f)

# print(run_paths)

log_dir = os.path.join(root, 'run/logs/')
with open(os.path.join(log_dir, f'run_paths.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['server', 'path_id', 'run', 'seed', 'env'])
    for row in run_paths:
    	writer.writerow(row)


