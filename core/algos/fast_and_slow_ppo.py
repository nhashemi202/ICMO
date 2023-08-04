import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyai.rl.algos.base import BaseAlgo

import re


def NS_single_step(pred, target):
    metric = nn.MSELoss()
    bs = pred.shape[0]
    return metric(pred, target.reshape(bs, -1))


def slow_param(p, new_rim_impl):
    if new_rim_impl:
        x = bool(re.match(r"memory_rnn\.myrnn\.bc_lst\.0\.mha\.*|memory_rnn\.myrnn\.bc_lst\.0\.inp_att\.*|critic.*",
                 str(p)))
        print(p, x)
        return x
    return bool(
        re.match(r"memory_rnn\.key.*|memory_rnn\.value.*|memory_rnn\.query.*|memory_rnn\.comm_attention.*|critic.*",
                 str(p)))


def fast_param(p, new_rim_impl):
    return not slow_param(p, new_rim_impl)


class FastAndSlowPPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))
    combined with a rough idea of Fast and Slow Learning  of 
    Recurrent Independent Modules
    ([Madan et al., 2021](https://arxiv.org/abs/2105.08710))
    """

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None, NS_loss=False, use_compositional_split=False,
                 compositional_test_splits=None, rim_slowness_factor=4, intrinsic_rew_coef=0.01, device=None, new_rim_impl=False):

        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info, use_compositional_split=use_compositional_split, 
                         compositional_test_splits=compositional_test_splits, device=device)

        self.clip_eps = clip_eps

        self.rim_slowness_factor = rim_slowness_factor
        self.original_recurrence = recurrence

        self.n_procs = len(envs)
        self.batch_size = batch_size
        self.fast_counter = 0

        # TODO
        assert self.batch_size % (self.rim_slowness_factor * self.original_recurrence) == 0
        assert num_frames_per_proc >= (self.rim_slowness_factor * self.original_recurrence)
        assert num_frames_per_proc * self.n_procs ==  self.batch_size

        self.optimizer_fast = torch.optim.Adam([p for name, p in self.acmodel.named_parameters() if fast_param(name, new_rim_impl)],
                                               lr, (beta1, beta2), eps=adam_eps)
        self.optimizer_slow = torch.optim.Adam([p for name, p in self.acmodel.named_parameters() if slow_param(name, new_rim_impl)],
                                               lr, (beta1, beta2), eps=adam_eps)
        self.optimizer = self.optimizer_fast
        self.batch_num = 0
        self.NS_loss = NS_loss

    def update_parameters(self):
        # Collect experiences

        # TODO
        exps, logs = self.collect_experiences()
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''

        for _ in range(self.n_procs + 1):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_losses = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one). Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            is_slow = _ == self.n_procs
            if is_slow:
                self.fast_counter = 0
                self.recurrence = self.original_recurrence * self.rim_slowness_factor
                self.optimizer = self.optimizer_slow
            else:
                self.recurrence = self.original_recurrence
                self.optimizer = self.optimizer_fast
            
            for inds in self._get_batches_starting_indexes(slow=is_slow):
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                self.acmodel.reset_hiddens()

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    model_results = self.acmodel(sb.obs, memory * sb.mask)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Add auxiliary losses
                    if self.NS_loss:
                        middle_x = model_results['middle_reps']['middle_x']
                        assert middle_x.shape[-1] == torch.numel(
                            sb.obs.image[0]), 'Next state dimension does not match.'
                        next_sb = None
                        if all([i + 1 < len(exps[k]) for k in inds]):
                            next_sb = exps[inds + i + 1]
                            loss += NS_single_step(middle_x, next_sb.obs.image)

                    if 'vq_loss' in model_results and not model_results['vq_loss'] is None:
                        loss += model_results['vq_loss']

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                # because we have a loop on variables in NPS and shared layers occur, set retain_graph to True
                batch_loss.backward(retain_graph=self.acmodel.arch.startswith("NPS"))
                grad_norm = sum(
                    p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self, slow=True):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """
        if slow:
            bs = self.batch_size
            nf = self.num_frames
            i = 0
            j = nf
        else:
            bs = self.batch_size // self.n_procs
            nf = self.num_frames // self.n_procs
            self.fast_counter += 1
            i = self.fast_counter * nf
            j = i + min(nf, self.batch_size - i)

        indexes = numpy.arange(i, j, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = bs // self.recurrence

        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
