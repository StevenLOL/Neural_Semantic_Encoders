#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator
from collections import Counter
import re
import sys
import datetime

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random
from random import randint

def preprocess4(sent):
	return ' '.join([x.strip() for x in re.split('(\W+)?', sent) if x.strip()])

def filterl(sents, l):
	return np.array([len(s) <= l for s in sents])

def filterl1(sents, lmin, lmax):
	return np.array([len(s) >= lmin and len(s) <= lmax for s in sents])

def filterl2(ds, lmax):
	r = []
	for sents in ds:
		max_sent = max(len(x) for x in sents)
		r.append(lmax >= max_sent)
	return np.array(r)

def split_dataset(dataset, lbls, test_size):
	dataset_index = StratifiedShuffleSplit(lbls, 1, test_size=test_size, random_state=123)

	for train_index, test_index in dataset_index:
		tr_dataset = [dataset[i] for i in train_index]
		test_dataset = [dataset[i] for i in test_index]
		tr_lbls = [lbls[i] for i in train_index]
		test_lbls = [lbls[i] for i in test_index]
	
	return tr_dataset, tr_lbls, test_dataset, test_lbls

def split_dataset2(dataset, lbls, steps, test_size):
	dataset_index = StratifiedShuffleSplit(lbls, 1, test_size=test_size, random_state=123)

	for train_index, test_index in dataset_index:
		tr_dataset = [dataset[i] for i in train_index]
		test_dataset = [dataset[i] for i in test_index]
		tr_lbls = [lbls[i] for i in train_index]
		test_lbls = [lbls[i] for i in test_index]
		tr_steps = [steps[i] for i in train_index]
		test_steps = [steps[i] for i in test_index]
	
	return tr_dataset, tr_lbls, tr_steps, test_dataset, test_lbls, test_steps

def fill_batch(batch, pad):
	max_len = max(len(x) for x in batch)
	return [[pad] * (max_len - len(x) + 1) + x for x in batch]

def fill_sent(sent, sent_start, sent_end):
	return [sent_start] + sent + [sent_end]

def fill_batch2(batch, pad):
	max_len = max(len(x) for x in batch)
	return [[pad] * (max_len - len(x)) + x for x in batch]

def fill_batch4(batch, garbage_c):
	max_len = max(len(x) for x in batch)
	return [[pad] * (max_len - len(x)) + x + [garbage_c] for x in batch]

def fill_batch3(batch, pad):
	max_len = max(len(x) for x in batch)
	batch = [[pad] * (max_len - len(x)) + x for x in batch]
	batch_bool = [[False] * (max_len - len(x)) + [True] * len(x) for x in batch]
	return batch, batch_bool

def fill_con_batch2(batch, pad):
	max_len = max(len(s) for x in batch for s in x)
	b = []
	for x in batch:
		ex = []
		for s in x:
			ex += [pad] * (max_len - len(s)) + s
		b.append(ex)
	return b, len(b[0])/len(batch[0])

def fill_con_batch3(batch, pad):
	max_len = max(len(s) for x in batch for s in x)
	max_sent = max(len(x) for x in batch)
	b = []
	for x in batch:
		ex = []
		for s in x:
			ex += [pad] * (max_len - len(s)) + s
		for i in range(max_sent - len(x)):
			ex +=  [pad] * (max_len)
		b.append(ex)
	return b, len(b[0])/max_sent

def batch(dataset, indexes):
	return [dataset[i] for i in indexes]

def all_vocab(ds):
	vocab = set()
	for l in ds:
		vocab.update(l.split())
	return vocab

def bucket_len(dataset):
	len_id = {}
	for x in xrange(len(dataset)):
		l = len(dataset[x])
		len_id[l] = len_id.get(l, []) + [x]
	return len_id

def prepare_batch_bucket(dataset_len2id, batch_size):
	rev_ids_batches = []
	for rev_len in dataset_len2id.keys():
		rev_ids = dataset_len2id[rev_len]
		np.random.shuffle(rev_ids)
		rev_ids_batches+=[rev_ids[x:x + batch_size] for x in xrange(0, len(rev_ids), batch_size)]
	return rev_ids_batches

def prepare_batch_bucket2(dataset_len2id, batch_size):
	rev_ids_batches = []
	id2len = {}
	for rev_len in dataset_len2id.keys():
		rev_ids = dataset_len2id[rev_len]
		for i in rev_ids:
			id2len[i]=rev_len
	ids = id2len.keys()
	while len(ids) > batch_size:
		np.random.shuffle(ids)
		b = {i:id2len[i] for i in ids[:batch_size*10]}
		batch = sorted(b.items(), key=operator.itemgetter(1))[:batch_size]
		batch = [i for i,l in batch]
		np.random.shuffle(batch)
		rev_ids_batches += [batch]
		ids = list(set(ids) - set(batch))
	if len(ids) > 0:
		np.random.shuffle(ids)
		rev_ids_batches += [ids]
	return rev_ids_batches

def random_batch_bucket(dataset_len2id, batch_size, lenght):
	if lenght < 10:
		batch_size = 64
	elif lenght > 10 and lenght < 20:
		batch_size = 16
	elif lenght > 20 and lenght < 30:
		batch_size = 16
	elif lenght > 30 and lenght < 40:
		batch_size = 16
	else:
		batch_size = 4
	rev_ids = dataset_len2id[lenght]
	np.random.shuffle(rev_ids)
	return rev_ids[:batch_size]

def flatten(container):
	for i in container:
		if isinstance(i, list) or isinstance(i, tuple):
			for j in flatten(i):
				yield j
		else:
			yield i

def rand_bin(n): 
	return [str(random.randint(0, 1)) for x in xrange(n)]

def generate_binary_reverse(n_batch, l_min, l_max):
	#1048575
	n = randint(l_min, l_max)
	s = [rand_bin(n) for i in range(n_batch)]
	return s, s[::-1]