#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @File : main.py
# @Software: PyCharm

from __future__ import division
from model import WholeModel
from tf_model import train_input_fn, eval_input_fn
import argparse
import datetime
import numpy as np

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='thg', help='dataset name: diginetica/thg')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
opt = parser.parse_args()

print('Loading train data.')
assert(opt.dataset)
# train_data = pickle.load(open('datasets/' + opt.dataset + '/processed/train.txt', 'rb'))
# print('Loading test data.')
# test_data = pickle.load(open('datasets/' + opt.dataset + '/processed/test.txt', 'rb'))

# print('Loading number of nodes.')
# n_node=0
# with open('datasets/' + opt.dataset + '/processed/all_train_seq.txt', 'rb') as file:
#     n_node = len(np.unique(np.hstack(pickle.load(file))))+1
# print(f'Dataset has {n_node} unique nodes.')

print('Processing train data.')
# train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
train_data, n_node, dataset_size, max_seq  = train_input_fn(opt.batchSize)

# train_data = TFData(f'../datasets/{opt.dataset}/processed/train.txt')

print('Processing test data.')
# test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)
test_data =eval_input_fn(opt.batchSize)


print('Loading model.')

model = WholeModel(max_seq = max_seq,
                n_node=n_node,
                l2=opt.l2,
                step=opt.step,
                lr=opt.lr,
                decay=opt.lr_dc_step * dataset_size / opt.batchSize,
                lr_dc=opt.lr_dc,
                hidden_size=opt.hiddenSize,
                out_size=opt.hiddenSize,
                batch_size=opt.batchSize,
                nonhybrid=opt.nonhybrid)

print(opt)
best_result = [0, 0]
best_epoch = [0, 0]


for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    print('start training: ', datetime.datetime.now())
    loss_ = []

    for batch in train_data.take(10):
        A_in, A_out, alias_inputs, items, mask, labels = batch
        loss = model.train_step(item=items, adj_in=A_in, adj_out=A_out, mask = mask, alias = alias_inputs, labels=labels)
        loss_.append(loss)
    print(np.mean(loss_))


    # for i, j in zip(slices, np.arange(len(slices))):
    #     adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
    #     _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias,  mask)
    #     loss_.append(loss)
    # loss = np.mean(loss_)
    # slices = test_data.generate_batch(model.batch_size)
    # print('start predicting: ', datetime.datetime.now())
    # hit, mrr, test_loss_ = [], [],[]
    # for i, j in zip(slices, np.arange(len(slices))):
    #     adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
    #     scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,  mask)
    #     test_loss_.append(test_loss)
    #     index = np.argsort(scores, 1)[:, -20:]
    #     for score, target in zip(index, targets):
    #         hit.append(np.isin(target - 1, score))
    #         if len(np.where(score == target - 1)[0]) == 0:
    #             mrr.append(0)
    #         else:
    #             mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
    # hit = np.mean(hit)*100
    # mrr = np.mean(mrr)*100
    # test_loss = np.mean(test_loss_)
    # if hit >= best_result[0]:
    #     best_result[0] = hit
    #     best_epoch[0] = epoch
    # if mrr >= best_result[1]:
    #     best_result[1] = mrr
    #     best_epoch[1]=epoch
    # print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'%
    #       (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))


