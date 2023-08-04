import sys
import os
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
import babyai.rl
import time
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from babyai.rl.utils.supervised_losses import required_heads
from .NPSUtils.BabyAIModels import VanillaModel
from VQ.vector_quantization import VectorQuantize
import RIM
from .RIM_new.block_wrapper import BlocksWrapper

from babyai.model import ExpertControllerFiLM

from collections import defaultdict


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 and classname.find('Group') == -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class VQRuleEmbedder(nn.Module):
    def __init__(self, emb_dim, rule_codebook_size, num_rules, total_tokens, use_large_model=False):
        super(VQRuleEmbedder, self).__init__()
        self.vq = VectorQuantize(
            dim=emb_dim,
            codebook_size=rule_codebook_size,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.,  # the weight on the commitment loss
            k=num_rules,
            use_cosine_sim=True,
        )
        self.instr_embedder = nn.Embedding(total_tokens, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        if use_large_model:
            self.linear = nn.Sequential(
                nn.Linear(emb_dim, 4 * emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim * 4, emb_dim)
            )
        else:
            self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, instr):
        """
        instr: (bs, num_tokens)
        """
        instr_embedding = self.gru(self.instr_embedder(instr))[1][-1]
        out = self.linear(instr_embedding)
        quantized, indices, vq_loss = self.vq(out.unsqueeze(1))
        return quantized.squeeze(1), indices.squeeze(1), vq_loss, instr_embedding


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, memory_dim=64, instr_dim=64,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None, NPS_info=None, rule_codebook_size=20,
                 film_d=128, use_large_model=False, use_rim=False, rim_num_units=4, rim_k=3,
                 concat_instr_to_mem=False, use_hidden_feedback=False,
                 att_fusion_info=None, use_intrinsic_rew=False, 
                 intrinsic_rew_coef=0.001, new_rim_impl=False, 
                 concat_obs_to_AC=False, pixel_images=False,
                 intrinsic_rew_mode='minmax',
                 input_mem_concat_instr=False):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.new_rim_impl = new_rim_impl
        self.use_intrinsic_rew = use_intrinsic_rew
        self.intrinsic_rew_coef = intrinsic_rew_coef
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.act_dim = NPS_info['act_dim']
        self.att_fusion_info = att_fusion_info
        self.rule_codebook_size = rule_codebook_size
        self.use_large_model = use_large_model
        self.use_rim = use_rim
        self.rim_num_units = rim_num_units
        self.rim_k = rim_k
        self.device = NPS_info['device']
        self.concat_instr_to_mem = concat_instr_to_mem
        self.use_hidden_feedback = use_hidden_feedback
        self.obs_space = obs_space
        self.concat_obs_to_AC = concat_obs_to_AC
        self.intrinsic_rew_mode = intrinsic_rew_mode
        self.intrinsic_rew_counts = defaultdict(int)
        self.input_mem_concat_instr = input_mem_concat_instr
        self.ignore_rules = NPS_info['ignore_rules']
        
        del NPS_info['ignore_rules']

        if self.arch == 'raw':
            self.memory_dim = 147

        if arch == 'fusion':
            # assert att_fusion_info is not None
            self.random_shuffled = False #att_fusion_info['random_shuffled']
            self.instr_sents = 1 # att_fusion_info['instr_sents']
            self.enable_instr = False # att_fusion_info['enable_instr']
            self.instr_only = False # att_fusion_info['instr_only']

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=memory_dim, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=film_d, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(film_d),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=film_d, out_channels=film_d, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(film_d),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        elif arch.startswith("NPS_vanilla"):

            self.image_conv = VanillaModel(**NPS_info)
            v = 1
            if NPS_info['use_all_slots']:
                if NPS_info['bottleneck_size'] is None:
                    v = NPS_info['num_variables']
                else:
                    v = NPS_info['bottleneck_size']
            self.NPS_linear = nn.Linear((self.act_dim + int(NPS_info['append_coord'])) * v, self.memory_dim)

        elif arch == "fusion":
            if not self.use_instr:
                raise ValueError("fusion architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.w_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=self.instr_sents+1, kernel_size=(3, 3), padding=1)
            )
            self.combined_conv = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2)),
                nn.ReLU()
            )
            self.combined_linear = nn.Linear(64, 1024)
            self.anohter_instr_rnn = None
        elif arch == 'raw':
            self.combine_raw = nn.Linear(147, self.memory_dim)
            assert self.concat_instr_to_mem
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        self.final_instr_dim = instr_dim
        if self.use_instr or self.concat_instr_to_mem or self.input_mem_concat_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(self.obs_space["instr"], 
                    self.instr_dim, padding_idx=0)
                if self.arch == 'fusion':
                    self.another_word_embedding = nn.Embedding(self.obs_space["instr"], 
                        self.instr_dim, padding_idx=0)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    if self.arch == 'fusion':
                        self.anohter_instr_rnn = nn.GRU(
                                self.instr_dim, gru_dim, batch_first=True,
                                bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            if self.lang_model == 'MHA':
                assert self.arch.startswith(
                    'NPS_vanilla'), 'MHA instruction embedding cannot be used without NPS model.'
                self.embed_K = nn.Embedding(self.obs_space["instr"], NPS_info['rule_dim'], padding_idx=0)
                self.embed_V = nn.Embedding(self.obs_space["instr"], self.instr_dim, padding_idx=0)
                # get query embeddings from image_conv.rule_network.rule_embeddings
                # self.MHA_heads = nn.ModuleList(
                #     [nn.MultiheadAttention(embed_dim=NPS_info['rule_dim'], num_heads=1) \
                #     for _ in range(NPS_info['num_rules'])])
                self.mha_head = nn.MultiheadAttention(embed_dim=NPS_info['rule_dim'], num_heads=NPS_info['num_rules'])
                if 'use_large_model' in self.__dict__ and self.use_large_model:
                    self.layer_norm_1 = nn.LayerNorm(NPS_info['rule_dim'])
                    self.layer_norm_2 = nn.LayerNorm(NPS_info['rule_dim'])
                    self.FFN = nn.Sequential(
                        nn.Linear(NPS_info['rule_dim'], 4 * NPS_info['rule_dim']),
                        nn.Sigmoid(),
                        nn.Linear(4 * NPS_info['rule_dim'], NPS_info['rule_dim'])
                    )
                self.final_instr_dim = self.instr_dim

            if self.lang_model == 'embed_only':
                self.rule_embedder = nn.Embedding(self.obs_space["instr"], NPS_info['rule_dim'], padding_idx=0)
                self.final_instr_dim = self.instr_dim  # should be equal to the rule dim

            if self.lang_model == 'VQ':
                self.rule_embedder = VQRuleEmbedder(NPS_info['rule_dim'], self.rule_codebook_size,
                                                    NPS_info['num_rules'], self.obs_space["instr"],
                                                    use_large_model=self.use_large_model)
                self.final_instr_dim = self.instr_dim  # should be rule dim

        # Define memory
        if self.use_memory:
            if self.use_rim:
                if self.new_rim_impl:
                    self.memory_rnn = BlocksWrapper(1, self.memory_dim, self.memory_dim).to(self.device)
                else:
                    self.memory_rnn = RIM.RIMCell(
                        self.device,
                        self.memory_dim,
                        self.semi_memory_size // self.rim_num_units,
                        self.rim_num_units,
                        self.rim_k,
                        'LSTM',
                        input_value_size=64,
                        comm_value_size=self.semi_memory_size // self.rim_num_units
                    )
            else:
                self.memory_rnn = nn.LSTMCell(self.memory_dim + self.input_mem_concat_instr * self.instr_dim, self.memory_dim)

        # # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.concat_instr_to_mem:
            self.embedding_size += self.final_instr_dim

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
                print('num_controllers: ', num_module)
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module - 1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=film_d, in_channels=film_d, imm_channels=film_d)
                else:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim, out_features=self.memory_dim,
                        in_channels=film_d, imm_channels=film_d)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size + (7*7*3 if self.concat_obs_to_AC else 0), 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size + (7*7*3 if self.concat_obs_to_AC else 0), 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        ############### NEW ################
        self.hierarchical_policy = False

        if self.hierarchical_policy:
            self.sparse_activation_gating = nn.Linear(49, self.sparse_activation_gating)

        self.memory_to_feedback = nn.Sequential(nn.Linear(self.memory_dim, self.memory_dim//2),
                                                nn.ReLU(),
                                                nn.Linear(self.memory_dim//2, 3))
        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def reset_hiddens(self):
        if self.arch.startswith('NPS'):
            self.image_conv.reset_t()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def add_obs_count(self, obs, mask):
        counts = []
        for i in range(obs.shape[0]):
            o = obs[i]
            m = mask[i]
            k = ''.join(list(o.detach().cpu().numpy().flatten().astype(int).astype(str)))
            if not m: # mask is 1 - done.
                self.intrinsic_rew_counts[k] = 0
            self.intrinsic_rew_counts[k] += 1
            counts.append(self.intrinsic_rew_counts[k])
        return torch.tensor(counts).to(self.device)




    def forward(self, obs, memory, instr_embedding=None, mask=None):
        vq_loss = None
        residual_instr = None
        if (self.use_instr or self.concat_instr_to_mem or self.input_mem_concat_instr) and instr_embedding is None:
            if self.arch == "fusion":
                instr_embedding, instr_embedding2 = self._get_instr_embedding(obs.instr)
            else:
                instr_embedding = self._get_instr_embedding(obs.instr)
            if self.lang_model == 'VQ':
                vq_loss = instr_embedding['vq_loss']
                residual_instr = instr_embedding['emb']
                instr_embedding = instr_embedding['q']
            elif self.lang_model == 'MHA':
                residual_instr = instr_embedding['emb']
                instr_embedding = instr_embedding['out']
            elif self.lang_model == 'gru':
                residual_instr = instr_embedding
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            instr_embedding = instr_embedding[:, :mask.shape[1]]
            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        # print('before conv: ', x.shape)
        rule_ids = None
        m_ts = None
        masks = None
        intrinsic_rew = torch.zeros((obs.image.shape[0],)).to(self.device)
        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            middle_x = x
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        elif self.arch.startswith("NPS_vanilla"):
            hidden_feedback = None
            if self.use_hidden_feedback:
                hidden_feedback = self.memory_to_feedback(memory[:, :self.semi_memory_size].clone().detach())
            x, rule_ids, m_ts, masks = self.image_conv(dict(image=obs.image, mission=obs.instr),
                                                instr_embedding=instr_embedding if self.use_instr else None, 
                                                hidden_feedback=hidden_feedback,
                                                ignore_rules=self.ignore_rules)

            if self.use_intrinsic_rew:
                x_reward = x.detach()
                if self.intrinsic_rew_mode == 'minmax':
                    intrinsic_rew = torch.max(x_reward, dim=-1)[0] - torch.min(x_reward, dim=-1)[0] 
                    intrinsic_rew -= intrinsic_rew.min(0, keepdim=True)[0]
                    intrinsic_rew /= intrinsic_rew.max(0, keepdim=True)[0]
                    intrinsic_rew *= self.intrinsic_rew_coef
                elif self.intrinsic_rew_mode == 'entropy-count-obs':
                    epsilon = 1e-7
                    n, m = x_reward.shape
                    x_max = torch.max(x_reward, dim=-1)[0].unsqueeze(-1).repeat(1, m)
                    x_min = torch.min(x_reward, dim=-1)[0].unsqueeze(-1).repeat(1, m)
                    intrinsic_rew = (x_reward - x_min) / (x_max - x_min + epsilon)
                    intrinsic_rew += epsilon
                    intrinsic_rew = self.intrinsic_rew_coef / Categorical(probs=intrinsic_rew).entropy()
                    if not mask is None:
                        counts = self.add_obs_count(obs.image, mask)
                        intrinsic_rew /= counts
                elif self.intrinsic_rew_mode == 'entropy-exp':
                    epsilon = 1e-7
                    n, m = x_reward.shape
                    x_max = torch.max(x_reward, dim=-1)[0].unsqueeze(-1).repeat(1, m)
                    x_min = torch.min(x_reward, dim=-1)[0].unsqueeze(-1).repeat(1, m)
                    intrinsic_rew = (x_reward - x_min) / (x_max - x_min + epsilon)
                    intrinsic_rew += epsilon
                    entropy = Categorical(probs=intrinsic_rew).entropy()
                    intrinsic_rew = 0.01 * (torch.exp(- entropy * entropy / 2) - 1)

            if self.hierarchical_policy:
                x_mean, x_std = torch.mean(x, dim=-1), torch.std(x, dim=-1)
                mean, std = torch.mean(x, dim=-1), torch.std(x, dim=-1)
                x_normal = (x-mean)/std
                x_ = self.sparse_activation_gating(x_normal - x_mean)
                x_min, x_max = torch.min(x_, dim=-1), torch.max(x_, dim=-1)

                x_ = (x_ - x_min) / (x_max - x_min)

            middle_x = x
            x = self.NPS_linear(x)
        elif self.arch == "fusion":
            x_feat = self.image_conv(x)
            w = self.w_conv(x)
            N, _, W, H = w.shape
            w = w.view([N, self.instr_sents + 1, -1])
            w = F.softmax(w, dim=1)
            y = torch.matmul(instr_embedding.unsqueeze(-1), w[:, :-1]).view([N, self.instr_dim, W, H])

            x = torch.cat([x_feat, y], axis=1)
            x = self.combined_conv(x)

            x = x.view(x.shape[0], x.shape[1], 1, 1)
            if self.enable_instr:
                raise 'Buggy'
                # for controler in self.controllers:
                #     x = controler(x, instr_embedding2)
                # x = F.relu(x)
            x = self.combined_linear(x.view([N, -1]))
            middle_x = x
        elif self.arch == "raw":
            # x = self.combine_raw(x.reshape(x.shape[0], -1))
            middle_x = x
        else:
            x = self.image_conv(x)
            middle_x = x

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            bs, mh = memory.shape
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            if self.use_rim:
                assert mh % self.rim_num_units == 0
                hidden = list(hidden)
                if self.new_rim_impl:
                    hidden = self.memory_rnn(x, hidden)
                else:
                    hidden[0] = hidden[0].view(hidden[0].size(0), self.rim_num_units, -1)
                    hidden[1] = hidden[0].view(hidden[1].size(0), self.rim_num_units, -1)
                    x = x.unsqueeze(1)

                    hidden = self.memory_rnn(x, hidden[0], hidden[1])
                    hidden = list(hidden)
                    hidden[0] = hidden[0].view(hidden[0].size(0), -1)
                    hidden[1] = hidden[1].view(hidden[1].size(0), -1)
            else:
                if self.input_mem_concat_instr:
                    hidden = self.memory_rnn(torch.cat((x, residual_instr), dim=-1), hidden)
                else:
                    hidden = self.memory_rnn(x, hidden)

            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.concat_instr_to_mem and not "filmcnn" in self.arch and not "fusion" in self.arch:
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            residual_instr = torch.nn.functional.normalize(residual_instr, dim=1)
            embedding = torch.cat((embedding, residual_instr), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        raw_obs = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3).reshape(x.shape[0], -1)

        if self.concat_obs_to_AC:
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            raw_obs = torch.nn.functional.normalize(raw_obs, dim=-1)
            embedding = torch.cat((embedding, raw_obs), dim=-1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions,
                'middle_reps': dict(middle_x=middle_x), 'vq_loss': vq_loss, 'rule_ids': rule_ids, 'm_ts': m_ts, 
                'intrinsic_rew':intrinsic_rew, 'masks': masks}

    def _get_instr_embedding(self, instr):

        if self.lang_model == 'VQ':
            q, _, vq_loss, emb = self.rule_embedder(instr)
            return dict(q=q, vq_loss=vq_loss, emb=emb)

        if self.lang_model in ['embed_only']:
            return self.rule_embedder(instr)

        if self.lang_model == 'MHA':
            Qs = self.image_conv.rule_net.rule_embeddings
            V = self.embed_V(instr)
            K = self.embed_K(instr)
            Qs = Qs.repeat(V.shape[0], 1, 1)
            out = self.mha_head(Qs, K, V)[0]
            if 'use_large_model' in self.__dict__ and self.use_large_model:
                out = self.layer_norm_1(out + Qs)
                out = self.layer_norm_2(out + self.FFN(out))
            return dict(out=out, emb=K.mean(dim=1))
            # temp = torch.empty_like(Qs)
            # for i in range(Qs.shape[1]):
            #     temp[:, i, :] = self.MHA_heads[i](Qs[:, i, :].unsqueeze(1), K, V)[0].squeeze(1)
            # return temp

        if self.lang_model == 'gru':
            _, hidden = self.instr_rnn(self.word_embedding(instr).to(self.device))
            if self.arch == 'fusion':
                _, hidden2 = self.instr_rnn(self.another_word_embedding(instr).to(self.device))
                return hidden[-1], hidden2[-1]
            return hidden[-1]

        elif self.lang_model in ['bigru', 'attgru']:
            raise 'Not implemented properly!'
            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            if outputs.shape[1] < masks.shape[1]:
                masks = masks[:, :(outputs.shape[1] - masks.shape[1])]
                # the packing truncated the original length
                # so we need to change mask to fit it

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
