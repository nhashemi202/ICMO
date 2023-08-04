from ModifiedModels import RuleNetwork
from torch import nn
from einops import rearrange

import torch
import numpy as np

from utilities.attention import SelectAttention
from slot_attention import SlotAttention


class VanillaModel(nn.Module):

    def __init__(self, in_dim=3, hidden_dim=3,
                 num_rules=8, rule_dim=64, act_dim=1,
                 query_dim=32, value_dim=32, key_dim=32,
                 num_heads=4, dropout=0.1, num_contexts=5,
                 num_variables=49, apply_mission=False, use_all_slots=False, 
                 use_null_slot=False, fuse_instr_obs=False,
                 device='cpu', bottleneck_size=None, use_slot_rnn=False,
                 append_coord=False, instr_to_rule_mode='conv1d',
                 use_slot_attention=False, compositional_step=1,
                 free_rule_param=False, sparse_features=False, 
                 use_hidden_feedback_contextual=False,
                 use_hidden_feedback_rule=False,
                 compositional_step_by_verbs=False,
                 verb_ids=[]):

        super(VanillaModel, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.num_variables = num_variables
        self.device = device
        self.use_all_slots = use_all_slots
        self.use_slot_rnn = use_slot_rnn
        self.bottleneck_size = num_variables if bottleneck_size is None else bottleneck_size
        self.append_coord = append_coord
        self.use_slot_attention = use_slot_attention
        self.compositional_step = compositional_step
        self.compositional_step_by_verbs = compositional_step_by_verbs
        self.verb_ids = verb_ids
        self.use_hidden_feedback_contextual = use_hidden_feedback_contextual

        self.rule_net = RuleNetwork(hidden_dim, num_variables,
                                    num_rules=num_rules,
                                    num_contexts=num_contexts,
                                    rule_dim=rule_dim,
                                    query_dim=query_dim,
                                    value_dim=value_dim,
                                    key_dim=key_dim,
                                    act_dim=act_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    design_config=None,
                                    device=device,
                                    use_all_slots=use_all_slots,
                                    use_null_slot=use_null_slot,
                                    fuse_instr_obs=fuse_instr_obs,
                                    instr_to_rule_mode=instr_to_rule_mode,
                                    rule_w=torch.randn(1, num_rules, rule_dim).to(self.device),
                                    free_rule_param=free_rule_param,
                                    sparse_features=sparse_features,
                                    use_hidden_feedback_contextual=use_hidden_feedback_contextual,
                                    use_hidden_feedback_rule=use_hidden_feedback_rule
                                    )  # modify to accept missions, too

        # recurrent part to maintain a hidden state of slots
        if self.use_slot_rnn:
            self.rnn = nn.ModuleList(
                [nn.GRU(in_dim, hidden_dim, batch_first=True).to(device) for _ in range(self.num_variables)])
        self.prev_h = None

        if self.bottleneck_size < self.num_variables:
            self.sparse_output_select = SelectAttention(act_dim, in_dim, d_k=32, num_read=num_variables,
                                                        num_write=num_variables)

        if self.use_slot_attention:
            self.slot_attention = SlotAttention(num_slots=num_variables,
                                                dim=3,
                                                hidden_dim=64,
                                                iters=3)

    def reset_t(self):  # MUST CALL AT THE BEGINNING OF EACH EPISODE!!!!
        self.prev_h = None

    def num_verb(self, missions):
        counts = (missions <= 4).sum(dim=1)
        return counts.max().item()

    def forward(self, obs_t, instr_embedding=None, hidden_feedback=None, ignore_rules=None):
        """
        obs_t: dict(image=(batch size, w, h, d), mission=[an array with `batch size` missions])
        (We expect to see a batch of observations)
        """
        assert not self.use_hidden_feedback_contextual or hidden_feedback is not None

        obs_image = obs_t['image']
        obs_mission = obs_t['mission']

        bs, w, h, d = obs_image.shape
        assert d == self.in_dim, f'Conflict in slot dimension, {d} != {self.in_dim}'
        # assert w * h == self.num_variables, f'Conflict in number of variables, {w}*{h} != {self.num_variables}'

        if self.prev_h is None and self.use_slot_rnn:
            self.prev_h = [torch.zeros(1, bs, self.hidden_dim, device=self.device) for _ in range(len(self.rnn))]

        outputs = rearrange(obs_image, 'bs w h d -> bs (w h) d')

        m_ts = []
        rule_ids_all = []
        masks = []
        compositional_step = self.num_verb(obs_mission) if self.compositional_step_by_verbs else self.compositional_step
        for _ in range(compositional_step):
            action_mlp_out, rule_ids, contextual_variable_ids, mask = self.rule_net(outputs, obs_mission,
                                                                                instr_embedding=instr_embedding,
                                                                                hidden_feedback=hidden_feedback,
                                                                                ignore_rules=ignore_rules)
            m_ts.append(action_mlp_out)
            rule_ids_all.append(rule_ids)
            masks.append(mask)
            outputs = action_mlp_out

        return action_mlp_out.reshape(bs, self.num_variables * self.act_dim), rule_ids_all, m_ts, masks

    # def forward(self, obs_t, instr_embedding=None, hidden_feedback=None):
    #     """
    #     obs_t: dict(image=(batch size, w, h, d), mission=[an array with `batch size` missions])
    #     (We expect to see a batch of observations)
    #     """
    #     assert not self.use_hidden_feedback_contextual or hidden_feedback is not None

    #     obs_image = obs_t['image']
    #     obs_mission = obs_t['mission']

    #     bs, w, h, d = obs_image.shape
    #     assert d == self.in_dim, f'Conflict in slot dimension, {d} != {self.in_dim}'
    #     # assert w * h == self.num_variables, f'Conflict in number of variables, {w}*{h} != {self.num_variables}'

    #     if self.prev_h is None and self.use_slot_rnn:
    #         self.prev_h = [torch.zeros(1, bs, self.hidden_dim, device=self.device) for _ in range(len(self.rnn))]

    #     obs_image = rearrange(obs_image, 'bs w h d -> bs (w h) d')

    #     try:
    #         if self.use_slot_attention and not self.bottleneck_size < self.num_variables:
    #             obs_image = self.slot_attention(obs_image)
    #     except:
    #         pass
    #     # if self.use_slot_rnn:
    #     #     output_pair = [rnn(obs_image[:, i, :].unsqueeze(1), self.prev_h[i]) for i, rnn in enumerate(self.rnn)]
    #     #     outputs, self.prev_h = [a for a, _ in output_pair], [b for _, b in output_pair]
    #     #     outputs = torch.cat(outputs, dim=1)
    #     # else:
    #     #     outputs = obs_image

    #     outputs = obs_image

    #     m_ts = []
    #     rule_ids_all = []
    #     if self.use_all_slots:
    #         for _ in range(self.compositional_step):
    #             actions_out = []
    #             rule_ids_all_step = []
    #             # We could add another dimension to BLOCKMLP and do it without a for loop, but it would multiple the parameter count.
    #             for i in range(self.num_variables):
    #                 rule_mlp_out, action_mlp_out, rule_ids, primary_variable_ids = self.rule_net(outputs, obs_mission,
    #                                                                                              instr_embedding=instr_embedding,
    #                                                                                              use_this_primary_ids=np.array(
    #                                                                                                  [i] * bs),
    #                                                                                              hidden_feedback=hidden_feedback)
    #                 actions_out.append(action_mlp_out)
    #                 rule_ids_all_step.append(rule_ids)
    #             outputs = torch.cat(actions_out, dim=1)
    #             m_ts.append(outputs)
    #             rule_ids_all.append(rule_ids_all_step)
    #         action_mlp_output = outputs
    #         if self.bottleneck_size < self.num_variables:  # if there is a bottleneck
    #             # Bottleneck on the action_mlp_slots via self attention
    #             out = self.sparse_output_select(action_mlp_output, obs_image)
    #             # Choose the slots with the most cumulative score across all input slots
    #             variable_score = torch.nn.functional.gumbel_softmax(out, dim=1, hard=False, tau=0.5).sum(dim=2,
    #                                                                                                      keepdim=True).repeat(
    #                 1, 1, self.act_dim)
    #             topk = variable_score.topk(self.bottleneck_size, dim=1, largest=True, sorted=True).indices
    #             action_mlp_output = torch.gather(action_mlp_output, 1, topk)
    #             if self.append_coord:
    #                 # Add the selected coordinates among 49 possible slots to the representation of each slot via topk[:, :, 0].
    #                 action_mlp_output = torch.cat((action_mlp_output,
    #                                                topk[:, :, 0].unsqueeze(-1)), dim=-1)

    #     else:
    #         rule_mlp_out, action_mlp_output, rule_ids, primary_variable_ids = self.rule_net(outputs, obs_mission,
    #                                                                                         instr_embedding=instr_embedding)

    #     return action_mlp_output.reshape(bs, (self.bottleneck_size if self.use_all_slots else 1) * (
    #                 self.act_dim + int(self.append_coord))), rule_ids_all, m_ts


class HierarchicalModel(nn.Module):
    def __init__(self):
        super(HierarchicalModel, self).__init__()
        pass
