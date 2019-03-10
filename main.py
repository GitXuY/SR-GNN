from __future__ import division
from model import WholeModel
from tf_model import train_input_fn, eval_input_fn
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import csv

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

n_node = 0
max_seq = 0
max_n_node = 0


with open(f"datasets/{opt.dataset}/processed/train.csv", "r") as data_file:
    data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]
    n_node = max(n_node,np.amax([np.amax(z) for z in data])+1)
    max_seq = max(max_seq,len(max(data, key=len)))
    max_n_node = max(max_n_node,len(max([np.unique(i) for i in data], key=len)))
    train_dataset_size = len(data)

with open(f"datasets/{opt.dataset}/processed/test.csv", "r") as data_file:
    data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]
    n_node = max(n_node,np.amax([np.amax(z) for z in data])+1)
    max_seq = max(max_seq,len(max(data, key=len)))
    max_n_node = max(max_n_node,len(max([np.unique(i) for i in data], key=len)))
    test_dataset_size = len(data)


print('Processing train data.')
train_data = train_input_fn(opt.batchSize, max_seq, max_n_node)

print('Processing test data.')
test_data = eval_input_fn(opt.batchSize, max_seq, max_n_node)

print('Loading model.')

model = WholeModel(n_node=n_node,
                l2=opt.l2,
                step=opt.step,
                lr=opt.lr,
                decay=opt.lr_dc_step * train_dataset_size / opt.batchSize,
                lr_dc=opt.lr_dc,
                hidden_size=opt.hiddenSize,
                out_size=opt.hiddenSize)


best_result = [0, 0]
best_epoch = [0, 0]

for epoch in range(opt.epoch):
    print('start training: ', datetime.datetime.now())

    loss_ = []
    with tqdm(total = np.floor(train_dataset_size/opt.batchSize)+1) as pbar:
        for A_in, A_out, alias_inputs, items, mask, labels in train_data:
            pbar.update(1)
            train_loss, logits = model.train_step(item=items, adj_in=A_in, adj_out=A_out, mask = mask, alias = alias_inputs, labels=labels)
            train_loss = train_loss.numpy()
            pbar.set_description(f"Training model. Epoch: {epoch}")
            pbar.set_postfix(loss=train_loss)

            loss_.append(train_loss)

    hit, mrr, test_loss_ = [], [], []
    with tqdm(total = np.floor(test_dataset_size/opt.batchSize)+1) as pbar:
        for A_in, A_out, alias_inputs, items, mask, labels in test_data:
            pbar.update(1)
            test_loss, logits = model.train_step(item=items, adj_in=A_in, adj_out=A_out, mask=mask, alias=alias_inputs, labels=labels, train=False)
            test_loss = test_loss.numpy()
            pbar.set_description(f"Testing model. Epoch: {epoch}")
            pbar.set_postfix(loss=test_loss)
            test_loss_.append(test_loss)

            index = np.argsort(logits, 1)[:, -20:]

            for score, target in zip(index, labels):
                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (20 - np.where(score == target - 1)[0][0]))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = epoch

    print(f"train_loss: {train_loss}, test_loss: {test_loss}, Recall@20: {best_result[0]}, MMR@20: {best_result[1]}, Epoch: {best_epoch[0]}:{best_epoch[1]}")