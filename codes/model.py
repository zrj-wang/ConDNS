#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
import torch.nn.init as init

from torch.utils.data import DataLoader
from model_cond import Diffusion_Cond
from dataloader import TestDataset
import time
# replace the whole KGEmodel, add model_cond,model_diffusion

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.double_entity_embedding = double_entity_embedding
        self.double_relation_embedding = double_relation_embedding

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        diff_dim = 64 if double_entity_embedding else 32

        self.linear_layer = nn.Linear(self.entity_dim, diff_dim).cuda()
        self.linear_layer_diff = nn.Linear(diff_dim, self.entity_dim).cuda()
        self.linear_layer_relation = nn.Linear(self.relation_dim, diff_dim).cuda()
        self.linear_layer_dim = nn.Linear(64, 32).cuda() # if de dr ,用来将关系转换成 32
        #
        self._initialize_weights()

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # undirected graph : nrelation*2
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def _initialize_weights(self):
        init.xavier_uniform_(self.linear_layer.weight)
        init.xavier_uniform_(self.linear_layer_diff.weight)
        init.xavier_uniform_(self.linear_layer_relation.weight)
        init.xavier_uniform_(self.linear_layer_dim.weight)



    def forward(self, sample, mode='single', method=None, h_emb_diff=None, t_emb_diff= None, h_emb=None, r_emb=None, t_emb=None):
        if method == 'id':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:,0]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=sample[:,1]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:,2]
                ).unsqueeze(1)

            elif mode == 'head-batch':
                tail_part, head_part = sample
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

                head = torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=head_part.view(-1)
                        ).view(batch_size, negative_sample_size, -1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)


                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            else:
                raise ValueError('mode %s not supported' % mode)
        else:
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:,0]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=sample[:,1]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:,2]
                ).unsqueeze(1)

            elif mode == 'head-batch':
                head = h_emb_diff

                relation = r_emb

                tail = t_emb

            elif mode == 'tail-batch':
                head = h_emb

                relation = r_emb

                tail = t_emb_diff




            else:
                raise ValueError('mode %s not supported' % mode)


        # if head is not None and head.shape[2] == 1000:
        #     head = self.linear_layer(head)
        # if relation is not None and relation.shape[2] == 1000:
        #     relation = self.linear_layer(relation)
        # if tail is not None and tail.shape[2] == 1000:
        #     tail = self.linear_layer(tail)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:

            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, diffusion_head, d_optimizer_head, diffusion_tail, d_optimizer_tail, neighbors, relations):

        model.train()
        optimizer.zero_grad()


        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        n_nodes = model.nentity
        num = 20

        output = model.entity_embedding
        output_relation_embedding = model.relation_embedding

        #######################
        output = model.linear_layer(output).detach()
        output_relation_embedding = model.linear_layer_relation(output_relation_embedding).detach()

        head_entities_batch = positive_sample[:, 0].tolist()
        relation_batch = positive_sample[:, 1].tolist()
        tail_entities_batch = positive_sample[:, 2].tolist()

        h_emb = output[head_entities_batch]  #
        r_emb = output_relation_embedding[relation_batch]  # relation embedding
        t_emb = output[tail_entities_batch]


        def train_diffusion():
            t_outputs = []
            hr_outputs = []
            rt_outputs = []
            h_outputs = []

            if mode == 'head-batch':   # we need r+t
                for i in head_entities_batch:
                    neighbors_i = neighbors[i]
                    relations_i = relations[i]

                    t_output_i = output[neighbors_i]
                    r_output_i = output_relation_embedding[relations_i]
                    h_output_i = output[i].detach().repeat(num, 1)

                    rt_output_i = r_output_i+t_output_i

                    rt_outputs.append(rt_output_i)
                    h_outputs.append(h_output_i)

                rt_output = torch.cat(rt_outputs, dim=0)
                h_output = torch.cat(h_outputs, dim=0)

                for epoch in range(args.d_epoch):

                    d_optimizer_head.zero_grad()
                    dif_loss_head = diffusion_head(h_output, rt_output).cuda()
                    dif_loss_head.backward(retain_graph=True)
                    d_optimizer_head.step()

            elif mode == 'tail-batch':  # we need h+r
                for i in head_entities_batch:
                    neighbors_i = neighbors[i]
                    relations_i = relations[i]

                    t_output_i = output[neighbors_i]
                    r_output_i = output_relation_embedding[relations_i]
                    h_output_i = output[i].detach().repeat(num, 1)

                    hr_output_i = h_output_i+r_output_i

                    t_outputs.append(t_output_i)
                    hr_outputs.append(hr_output_i)

                t_output = torch.cat(t_outputs, dim=0)    # tail
                hr_output = torch.cat(hr_outputs, dim=0)     # h+r

                for epoch in range(args.d_epoch):
                    d_optimizer_tail.zero_grad()
                    dif_loss_tail = diffusion_tail(t_output, hr_output).cuda()
                    dif_loss_tail.backward(retain_graph=True)
                    d_optimizer_tail.step()

        if mode == 'tail-batch' or mode == 'head-batch':

            train_diffusion()


        h_syn_tail = diffusion_tail.sample((h_emb+r_emb).shape, (h_emb+r_emb))  # use h+r to get t
        h_syn_head = diffusion_head.sample((r_emb + t_emb).shape, (r_emb + t_emb))  # use r+t to get h


        h_syn_tail_tensor = torch.stack(h_syn_tail, dim=1)
        h_syn_head_tensor = torch.stack(h_syn_head, dim=1)


        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode, method='id')

        if model.double_entity_embedding and not model.double_relation_embedding:
            r_emb = model.linear_layer_dim(r_emb)

        negative_score_diff = model((positive_sample, negative_sample), mode=mode, method='embedding',
                                    h_emb_diff=h_syn_head_tensor, t_emb_diff= h_syn_tail_tensor, h_emb=h_emb.unsqueeze(1), r_emb=r_emb.unsqueeze(1), t_emb=t_emb.unsqueeze(1))

        if args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)



        positive_score = model(positive_sample, method='id')


        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)


        if args.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        w = [1, 0.9, 0.8, 0.7]
        repeat_counts = [6, 6, 4, 4]

        w = [weight for weight, count in zip(w, repeat_counts) for _ in range(count)]

        weighted_logsigmoid = F.logsigmoid(-negative_score_diff) * torch.tensor(w).cuda()

        negative_diff_loss = (-weighted_logsigmoid.sum(dim=1) / len(w)).mean()

        negative_sample_loss = (negative_sample_loss + negative_diff_loss)

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'negative_diff_loss': negative_diff_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        model.eval()
        # if args.cuda:
        #     model = model.cuda()
            # if diffusion:
            #     diffusion = diffusion.cuda()

        if args.countries:
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)
            auc_pr = average_precision_score(y_true, y_score)
            metrics = {'auc_pr': auc_pr}

        else:
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            logs = []
            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode, method='id')
                        score += filter_bias

                        argsort = torch.argsort(score, dim=1, descending=True)
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0 :
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics