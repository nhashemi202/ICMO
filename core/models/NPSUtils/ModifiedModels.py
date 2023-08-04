import torch
import torch.nn as nn
import math
import numpy as np
from utilities.GroupLinearLayer import GroupLinearLayer
from utilities.attention_rim import MultiHeadAttention
import itertools
from utilities.attention import SelectAttention

from babyai.model import ExpertControllerFiLM

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
    	m.weight.data.normal_(0, 1)
    	m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
    	if m.bias is not None:
    		m.bias.data.fill_(0)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Identity(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input):
		return input * 1.0
	def backward(ctx, grad_output):
		#print(torch.sqrt(torch.sum(torch.pow(grad_output,2))))
		print(grad_output)
		return grad_output * 1.0

class ArgMax(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		idx = torch.argmax(input, 1)
		ctx._input_shape = input.shape
		ctx._input_dtype = input.dtype
		ctx._input_device = input.device
		#ctx.save_for_backward(idx)
		op = torch.zeros(input.size()).to(input.device)
		op.scatter_(1, idx[:, None], 1)
		ctx.save_for_backward(op)
		return op

	@staticmethod
	def backward(ctx, grad_output):
		op, = ctx.saved_tensors
		grad_input = grad_output * op
		return grad_input

class GroupMLP(nn.Module):
	def __init__(self, in_dim, out_dim, num):
		super().__init__()
		self.group_mlp1 = GroupLinearLayer(in_dim, 128, num)
		self.group_mlp2 = GroupLinearLayer(128, out_dim, num)
		#self.group_mlp3 = GroupLinearLayer(128, 128, num)
		#self.group_mlp4 = GroupLinearLayer(128, out_dim, num)
		self.dropout = nn.Dropout(p = 0.5)


	def forward(self, x):
		x = torch.relu(self.group_mlp1(x))
		x = self.group_mlp2(x)
		#x = torch.relu(self.dropout(self.group_mlp3(x)))
		#x = torch.relu(self.dropout(self.group_mlp4(x)))
		return x

class MLP(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.mlp1 = nn.Linear(in_dim, 128)
		self.mlp2 = nn.Linear(128, out_dim)
		self.mlp3 = nn.Linear(128, 128)
		self.mlp4 = nn.Linear(128, out_dim)
		#self.dropout = nn.Dropout(p = 0.5)

	def forward(self, x):
		x = torch.relu(self.mlp1(x))
		x = self.mlp2(x)
		#x = torch.relu(self.mlp3(x))
		#x = self.mlp4(x)
		#x = torch.relu(self.mlp3(x))
		#x = self.mlp4(x)
		return x

class Hook():
    def __init__(self, inp):
        self.hook = inp.register_hook(self.hook_fn)
        self.mask = None
    def hook_fn(self, grad):
        grad = grad * self.mask
        return grad
    def close(self):
        self.hook.remove()

class RuleNetwork(nn.Module):
	def __init__(self, hidden_dim, num_variables, num_rules = 4, num_contexts = 1,
		rule_dim = 64, act_dim=1, query_dim = 32, value_dim = 64, key_dim = 32, num_heads = 4, 
		dropout = 0.1, design_config = None, device='cpu', 
		rule_w=None, use_all_slots=False, instr_to_rule_mode='conv1d', 
		free_rule_param=False, sparse_features=False, use_hidden_feedback_contextual=False,
		use_null_slot = False, fuse_instr_obs=False, use_hidden_feedback_rule=False):
		super().__init__()
		self.rule_dim = rule_dim
		self.num_heads = num_heads
		self.key_dim = key_dim
		self.device = device
		self.value_dim = value_dim
		self.act_dim = act_dim
		self.query_dim = query_dim
		self.hidden_dim = hidden_dim
		self.design_config = design_config
		self.num_contexts = num_contexts
		self.num_variables = num_variables
		self.num_rules = num_rules
		self.instr_to_rule_mode = instr_to_rule_mode
		self.free_rule_param = free_rule_param
		self.sparse_features = sparse_features
		self.use_hidden_feedback_contextual = use_hidden_feedback_contextual
		self.use_all_slots = use_all_slots
		self.use_null_slot = use_null_slot
		self.fuse_instr_obs = fuse_instr_obs
		self.use_hidden_feedback_rule = use_hidden_feedback_rule

		self.rule_activation = []
		self.variable_activation = []
		self.softmax = []
		self.masks = []
		
		self.reset_activations()

		self.rule_embeddings = nn.Parameter(rule_w) # torch.randn(1, num_rules, rule_dim).to(self.device)

		if instr_to_rule_mode == 'conv1d':
			self.instr_to_rule = nn.Conv1d(1, self.num_rules, kernel_size=1)
		elif instr_to_rule_mode == 'MHA':
			self.instr_to_rule = None
		elif instr_to_rule_mode == 'linear':
			self.instr_to_rule = nn.Linear(self.rule_dim, self.rule_dim * self.num_rules)
			self.instr_to_rule.apply(initialize_parameters)

		if self.fuse_instr_obs:
			self.instr_obs_fuser = ExpertControllerFiLM(
                        in_features=self.rule_embeddings.shape[-1], out_features=self.hidden_dim,
                        in_channels=self.hidden_dim, imm_channels=64)

		if self.use_hidden_feedback_rule:
			self.variable_rule_select = SelectAttention(self.rule_dim * (int(self.free_rule_param) + 1), 
				self.hidden_dim * 2, d_k=32, num_read = self.num_rules, 
				num_write = 49, share_query = True)
		else:
			self.variable_rule_select = SelectAttention(self.rule_dim * (int(self.free_rule_param) + 1), 
				self.hidden_dim , d_k=32, num_read = self.num_rules, 
				num_write = 49, share_query = True)


		if self.sparse_features:
			self.group_sparse_feature_mask = nn.Sequential(
				GroupLinearLayer(self.hidden_dim * (1 + self.num_contexts), self.hidden_dim * (1 + self.num_contexts), self.num_rules),
				nn.ReLU()
			)

		self.action_mlp = GroupMLP((self.num_contexts + 1) * self.hidden_dim, self.act_dim, self.num_rules)

		if self.use_hidden_feedback_contextual:
			self.contextual_selector = SelectAttention(2 * self.hidden_dim, self.hidden_dim,
														d_k=32, num_read = num_variables,
														num_write = 49)

		else:
			self.contextual_selector = SelectAttention(self.hidden_dim, self.hidden_dim,
														d_k=32, num_read = num_variables,
														num_write = 49)


	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
	    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
	    x = x.view(*new_x_shape)
	    return x.permute(0, 2, 1, 3)

	def parallel(self, scores, hidden, hidden_feedback=None):

		batch_size, num_variables, variable_dim = hidden.size()

		if self.use_null_slot:
			scores = torch.cat([scores, torch.zeros(scores.shape[0], 1, scores.shape[2], device=self.device)], dim=1)

		if self.training:
			mask = torch.nn.functional.gumbel_softmax(scores, dim = 1, tau = 1.0, hard = True)
			mask = mask.permute(0, 2, 1)
			scores = scores.permute(0, 2, 1).float()
		else:
			mask = ArgMax().apply(scores)
			mask = mask.permute(0, 2, 1)
			scores = scores.permute(0, 2, 1).float()

		rule_ids = torch.argmax(mask.detach(), dim=2) # as actions for the high-level policy
		primary_variable = hidden 

		############# contextual ############

		contextual_query = primary_variable
		if self.use_hidden_feedback_contextual:
			# TODO additive?
			hidden_feedback = hidden_feedback.unsqueeze(1).repeat(1, 49, 1)
			contextual_query = torch.cat((hidden_feedback, contextual_query), dim=-1)

		variable_score = self.contextual_selector(contextual_query, hidden)
		
		variable_score = torch.nn.functional.gumbel_softmax(variable_score, 
			dim=-1, hard=False, tau=0.5) 
		# detach?
		contextual_variable_ids = variable_score.topk(self.num_contexts, 
			dim=-1, largest=True, sorted=True).indices
		contextual_variable = torch.gather(hidden, 1, contextual_variable_ids.repeat(1, 1, self.hidden_dim))

		######################################
		rule_mlp_input = torch.cat([primary_variable, contextual_variable], dim=-1) 
		# considerig the null rule
		rule_mlp_input = rule_mlp_input.reshape(batch_size * self.num_variables, -1).unsqueeze(1).repeat(1, self.num_rules, 1)

		action_mlp_output = None
		if self.sparse_features:
			raise 'recheck sparse_features for parallel'
			rule_mlp_input = torch.sign(input=self.group_sparse_feature_mask(rule_mlp_input)) * rule_mlp_input
		

		action_mlp_output = self.action_mlp(rule_mlp_input)
		action_mlp_output = torch.sum(action_mlp_output * mask[:, :, :self.num_rules].reshape(batch_size * num_variables, self.num_rules, 1), 
									dim=1).reshape(batch_size, self.num_variables, self.act_dim)
		# filter null slots
		if self.use_null_slot:
			action_mlp_output = torch.where(rule_ids.reshape(batch_size, self.num_variables, 1).repeat(1, 1, self.act_dim) < self.num_rules, 
										action_mlp_output, 
										torch.zeros_like(action_mlp_output, device=self.device))
		return action_mlp_output, rule_ids, contextual_variable_ids, mask


	def sequential(self, scores, hidden, hidden_feedback=None):

		batch_size, num_variables, variable_dim = hidden.size()

		if self.training:
			mask = torch.nn.functional.gumbel_softmax(scores.reshape(batch_size, -1), dim = 1, tau = 1.0, hard = True)
			mask = mask.reshape(batch_size, self.num_rules, num_variables)
			mask = mask.permute(0, 2, 1)
			scores = scores.permute(0, 2, 1).float()
		else:
			mask = ArgMax().apply(scores.reshape(batch_size, -1)).reshape(batch_size, self.num_rules, num_variables)
			mask = mask.permute(0, 2, 1)
			scores = scores.permute(0, 2, 1).float()

		rule_ids = torch.argmax(mask.detach().sum(dim=1), dim = 1).reshape(batch_size, self.num_variables) 
		primary_variable_ids = torch.argmax(mask.detach().sum(dim=2), dim = 1) 
		primary_variable = hidden.gather(1, primary_variable_ids.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hidden.shape[2]))
		
		############# contextual ############

		contextual_query = primary_variable
		if self.use_hidden_feedback_contextual:
			contextual_query = torch.cat((hidden_feedback.unsqueeze(1), contextual_query), dim=-1) # recheck hidden feedback's shape

		variable_score = self.contextual_selector(contextual_query, hidden)
		
		variable_score = torch.nn.functional.gumbel_softmax(variable_score, 
			dim=-1, hard=False, tau=0.5) 
		# detach?
		contextual_variable_ids = variable_score.topk(self.num_contexts, 
			dim=-1, largest=True, sorted=True).indices
		contextual_variable = torch.gather(hidden, 1, contextual_variable_ids.repeat(1, 1, self.hidden_dim))

		######################################
		rule_mlp_input = torch.cat([primary_variable, contextual_variable], dim=-1) 
		rule_mlp_input = rule_mlp_input.reshape(batch_size * self.num_variables, -1).unsqueeze(1).repeat(1, self.num_rules, 1)

		action_mlp_output = None

		if self.sparse_features:
			raise 'recheck sparse_features for sequential'
			rule_mlp_input = torch.sign(input=self.group_sparse_feature_mask(rule_mlp_input)) * rule_mlp_input

		action_mlp_output = self.action_mlp(rule_mlp_input)

		action_mlp_output = torch.sum(torch.mul(action_mlp_output, mask.sum(dim=1).unsqueeze(-1).repeat(1, 1, self.act_dim)), 
									dim=1).reshape(batch_size, self.num_variables, self.act_dim)

		return action_mlp_output, rule_ids, contextual_variable_ids, mask


	def forward(self, hidden, obs_mission, consider_all_primaries=False, 
		instr_embedding=None, hidden_feedback=None, ignore_rules=None):


		batch_size, num_variables, variable_dim = hidden.size()

		num_rules = self.rule_embeddings.size(1)
		free_rule_param_tensor = self.rule_embeddings.repeat(batch_size, 1, 1)
		rule_emb_orig = free_rule_param_tensor
		if not instr_embedding is None:
			if self.instr_to_rule_mode == 'conv1d':
				rule_emb_orig = self.instr_to_rule(instr_embedding.unsqueeze(1))
			elif self.instr_to_rule_mode in ['MHA', 'embed_only', 'VQ']:
				rule_emb_orig = instr_embedding
			elif self.instr_to_rule_mode == 'linear':
				emb_size = instr_embedding.shape[-1]
				rule_emb_orig = self.instr_to_rule(instr_embedding)
				rule_emb_orig = rule_emb_orig.reshape(batch_size, num_rules, emb_size)
			else:
				raise 'Not specified yet.'
			
			if self.free_rule_param:
				rule_emb_orig = torch.cat([free_rule_param_tensor, rule_emb_orig], dim=-1)

		if self.fuse_instr_obs:
			# Change static values to variables later (7, 49, 3)
			instr_obs = self.instr_obs_fuser(hidden.reshape(-1, 7, 7, 3).permute(0, 3, 1, 2), instr_embedding)
			hidden = instr_obs.permute(0, 2, 3, 1).reshape(-1, 49, 3)
			rule_emb = free_rule_param_tensor
		else:
			rule_emb = rule_emb_orig

		if self.use_hidden_feedback_rule:
			rule_select_query_like = torch.cat((hidden_feedback.unsqueeze(1).repeat(1, 49, 1), hidden), dim=-1)
			scores = self.variable_rule_select(rule_emb, rule_select_query_like)
		else:
			scores = self.variable_rule_select(rule_emb, hidden)

		if not ignore_rules is None:
			for i in ignore_rules:
				scores[:, i, :] = scores[:, i, :] + float('-inf')

		if self.use_all_slots:
			action_mlp_output, rule_ids, contextual_variable_ids, mask = self.parallel(scores, hidden, hidden_feedback=hidden_feedback)
		else:
			action_mlp_output, rule_ids, contextual_variable_ids = self.sequential(scores, hidden, hidden_feedback=hidden_feedback)

		return action_mlp_output, rule_ids, contextual_variable_ids, mask


	# def forward(self, hidden, obs_mission, consider_all_primaries=False, 
	# 	output_variable=True, use_this_primary_ids=None, 
    #     instr_embedding=None, hidden_feedback=None):
	# 	"""
	# 	:attr:consider_all_primaries: if True, we do not need to select the primary slot any more, 
	# 								  just consider each one in a loop. It has to be True for next
	# 								  state prediction as needed in DADS.
	# 	"""

	# 	batch_size, num_variables, variable_dim = hidden.size()

	# 	num_rules = self.rule_embeddings.size(1)
	# 	free_rule_param_tensor = self.rule_embeddings.repeat(batch_size, 1, 1)
	# 	rule_emb_orig = free_rule_param_tensor
	# 	if not instr_embedding is None:
	# 		if self.instr_to_rule_mode == 'conv1d':
	# 			rule_emb_orig = self.instr_to_rule(instr_embedding.unsqueeze(1))
	# 		elif self.instr_to_rule_mode in ['MHA', 'embed_only', 'VQ']:
	# 			rule_emb_orig = instr_embedding
	# 		elif self.instr_to_rule_mode == 'linear':
	# 			emb_size = instr_embedding.shape[-1]
	# 			rule_emb_orig = self.instr_to_rule(instr_embedding)
	# 			rule_emb_orig = rule_emb_orig.reshape(batch_size, self.num_rules, emb_size)
	# 		else:
	# 			raise 'Not specified yet.'
			
	# 		if self.free_rule_param:
	# 			rule_emb_orig = torch.cat([free_rule_param_tensor, rule_emb_orig], dim=-1)

	# 	rule_emb = rule_emb_orig

	# 	scores = self.variable_rule_select(rule_emb, hidden)

	# 	if not use_this_primary_ids is None:
	# 		use_this_primary_ids = torch.from_numpy(use_this_primary_ids).to(self.device)
	# 		t = use_this_primary_ids.unsqueeze(1).unsqueeze(1).repeat(1, self.num_rules, 1).type(torch. int64)
	# 		scores = scores.gather(2, t)
	# 		num_variables = 1

	# 	if self.training:
	# 		mask = torch.nn.functional.gumbel_softmax(scores.reshape(batch_size, -1), dim = 1, tau = 1.0, hard = True)
	# 		mask = mask.reshape(batch_size, num_rules, num_variables)
	# 		mask = mask.permute(0, 2, 1)
	# 		scores = scores.permute(0, 2, 1).float()
	# 	else:
	# 		mask = ArgMax().apply(scores.reshape(batch_size, -1)).reshape(batch_size, num_rules, num_variables)
	# 		mask = mask.permute(0, 2, 1)
	# 		scores = scores.permute(0, 2, 1).float()
	# 		mask_print = mask

	# 	variable_mask = torch.sum(mask, dim = 2) if use_this_primary_ids is None else torch.nn.functional.one_hot(use_this_primary_ids.type(torch.int64), num_classes=self.num_variables) # stores [1] for the primary variable per batch, one-hot
	# 	variable_mask = variable_mask.unsqueeze(-1)
	# 	rule_mask = torch.sum(mask, dim = 1).unsqueeze(-1) # stores one-hot representation of the selected rule per batch
	# 	rule_mask_print = torch.sum(mask, dim = 1).detach()
	# 	rule_ids = torch.argmax(rule_mask_print, dim = 1) # as actions for the high-level policy
	# 	primary_variable_ids = torch.argmax(torch.sum(mask, dim = 2).detach(), dim = 1) if use_this_primary_ids is None else use_this_primary_ids

	# 	primary_variable = (hidden * variable_mask).sum(dim = 1) # difficult way of selecting the primary slot, why not index selection?

	# 	############# contextual ############

	# 	contextual_query = primary_variable.unsqueeze(1)
	# 	if self.use_hidden_feedback_contextual:
	# 		# TODO additive?
	# 		contextual_query = hidden_feedback.unsqueeze(1) + contextual_query

	# 	variable_score = self.contextual_selector(contextual_query, hidden) # selects the contextual slot
		
	# 	variable_score = torch.nn.functional.gumbel_softmax(variable_score, 
	# 		dim=-1, hard=False, tau=0.5) # dim is -1, because the last dimension corresponds to contextual candidates per variable (primary)
	# 	contextual_variable_ids = variable_score.topk(self.num_contexts, 
	# 		dim=-1, largest=True, sorted=True).indices
		
	# 	contextual_variable = torch.gather(hidden, 1, contextual_variable_ids.permute(0, 2, 1).repeat(1, 1, self.hidden_dim)).reshape(batch_size, -1)
	# 	######################################

	# 	rule_mlp_input = torch.cat([primary_variable, contextual_variable], dim=-1) # appending primary and contextuals
	# 	# Following lines calculate the output of the selected rule mlp
	# 	rule_mlp_input = rule_mlp_input.unsqueeze(1).repeat(1, rule_mask.size(1), 1)
	# 	"""
	# 	In order to separate flow of the gradient for high and low level policies, we
	# 	have to **clone** and detach one copy of the input and mask here, or a few lines before this point.
	# 	Also we have to return the attached copy to apply high level gradients on it later.
	# 	"""
	# 	rule_mlp_output, action_mlp_output = None, None
	# 	if self.sparse_features:
	# 		rule_mlp_input = torch.sign(input=self.group_sparse_feature_mask(rule_mlp_input)) * rule_mlp_input
	# 	action_mlp_output = self.action_mlp(rule_mlp_input)
	# 	action_mlp_output = torch.sum(action_mlp_output * rule_mask, dim = 1).unsqueeze(1)

	# 	return rule_mlp_output, action_mlp_output, rule_ids, primary_variable_ids

	def reset_activations(self):
		self.rule_activation = []
		self.variable_activation = []
		self.rule_probabilities = []
		self.variable_probabilities = []
