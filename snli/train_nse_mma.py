import math
import sys
sys.path.append('../')
import time
import copy
import pandas as pd
import numpy as np
import six
from collections import Counter
import gc
import argparse

import chainer
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

from utils.Preprocessing import preprocess4, batch

from NSE_MMA import NSE_MMA
from NSE_MMA_attention import NSE_MMA_attention

print "chainer version:", chainer.__version__

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0,
					help='GPU ID (negative value indicates CPU)')
parser.add_argument('--snli', '-i', default='path/to/snli_1.0',
					help='Directory to read SNLI data')
parser.add_argument('--glove', '-w', default='path/to/glove.840B.300d.txt',
					help='File to read glove vectors')
parser.add_argument('--out', '-o', default='result',
					help='Directory to output the trained models')
parser.add_argument('--test', '-t', type=bool, default=False,
					help='Use tiny datasets for quick tests')

args = parser.parse_args()

gpu = args.gpu
input_dir = args.snli
glove_path = args.glove
out_dir = args.out
test_run = args.test

n_epoch   = 40   # number of epochs
n_units   = 300  # number of units per layer
batch_size = 128  # minibatch size
eval_batch = 128

EMPTY = np.random.uniform(-0.1, 0.1, (1, 300)).astype(np.float32)
joblib.dump(EMPTY, out_dir + '/NSE_MMA.empty')

def get_vec(word):
	try:
		vec = vectors[word]
	except:
		vec = np.zeros((1, 300), dtype=np.float32)
	return vec

def word2vec(sent):
	words = sent.split()[:32]
	words = ['<EMPTY>'] * (32 - len(words)) + words
	return [get_vec(word) for word in words]

print ("Loading data...")
snli_train = pd.read_csv(input_dir + '/snli_1.0_train.txt', sep='\t', usecols=['gold_label', 'sentence1', 'sentence2'])
snli_dev = pd.read_csv(input_dir + '/snli_1.0_dev.txt', sep='\t', usecols=['gold_label', 'sentence1', 'sentence2'])
snli_test = pd.read_csv(input_dir + '/snli_1.0_test.txt', sep='\t', usecols=['gold_label', 'sentence1', 'sentence2'])
def all_vocab(snli):
	vocab = set()
	for i in xrange(len(snli.gold_label)):
		try:
			sent1 = snli.sentence1[i]
			sent2 = snli.sentence2[i]
			vocab.update(preprocess4(sent1).split())
			vocab.update(preprocess4(sent2).split())
			if i > 1000 and test_run:
				break
		except Exception, e:
			print "got exception in line ", i
	return vocab

vocab = all_vocab(snli_train) | all_vocab(snli_dev) | all_vocab(snli_test)
with open(glove_path, 'r') as f:
		vectors = {}
		for line in f:
			vals = line.rstrip().split(' ')
			if vals[0] in vocab:
				vectors[vals[0]] =  np.array(vals[1:], ndmin=2, dtype=np.float32)
		vectors['<EMPTY>'] = EMPTY
vocab = None
gc.collect()

print ("Preprocessing...")
def preprocess_set(snli):
	lbls = []
	ds = []
	s = 0
	for i in xrange(len(snli.gold_label)):
		try:
			lbl = snli.gold_label[i]
			sent1 = snli.sentence1[i]
			sent2 = snli.sentence2[i]
			if lbl != '-':
				sents = [word2vec(preprocess4(sent)) for sent in [sent1, sent2]]
				ds.append(sents)
				lbls.append(lbl)
			if i > 1000 and test_run:
				break
		except Exception, e:
			print "got exception in line ", i
			print e
			s+=1
	print "docs can't preprocess:", s
	return ds, lbls

def stack_pairs(sent_batch):
	sents1 = []
	sents2 = []
	for sent1, sent2 in sent_batch:
		sents1.append(sent1)
		sents2.append(sent2)
	return sents1, sents2

snli_train, lbls_tr = preprocess_set(snli_train)
snli_dev, lbls_dev = preprocess_set(snli_dev)
snli_test, lbls_test = preprocess_set(snli_test)
vectors = None
gc.collect()
le = LabelEncoder()
le.fit(lbls_tr)
n_outputs = le.classes_.shape[0]
lbls_tr = le.transform(lbls_tr)
lbls_dev = le.transform(lbls_dev)
lbls_test = le.transform(lbls_test)

print "Train size:", len(snli_train)
print "Output size:", n_outputs

n_train = len(snli_train)
n_dev = len(snli_dev)
n_test = len(snli_test)

print "batch_size", batch_size
print "GPU", gpu

maxes = []
print "Building model..."
model = NSE_MMA(n_units, gpu)
# model = NSE_MMA_attention(n_units, gpu)
print "model:",model
model.init_optimizer()
max_epch = 0
max_tr = 0
max_dev = 0
max_test = 0
print "Train looping..."
for i in xrange(0, n_epoch):
	print "epoch={}".format(i)
	#Shuffle the data
	shuffle = np.random.permutation(n_train)
	preds=[]
	preds_true=[]
	aLoss = 0
	ss = 0
	begin_time = time.time()
	for j in six.moves.range(0, n_train, batch_size):
		c_b = shuffle[j:min(j+batch_size, n_train)]
		ys = batch(lbls_tr, c_b)
		preds_true.extend(ys)
		y_data = np.array(ys, dtype=np.int32)
		sent_batch = batch(snli_train, c_b)
		a_batch, q_batch = stack_pairs(sent_batch)
		y_s, loss = model.train(a_batch, q_batch, y_data)
		aLoss += loss.data
		preds.extend(y_s)
		ss += 1
	print "loss:", aLoss/ss
	print 'secs per train epoch={}'.format(time.time() - begin_time)
	f1_tr = accuracy_score(preds_true, preds)
	print 'train accuracy_score={}'.format(f1_tr)
	preds = []
	preds_true=[]
	for j in six.moves.range(0, n_dev, eval_batch):
		ys = lbls_dev[j:j+eval_batch]
		preds_true.extend(ys)
		y_data = np.array(ys, dtype=np.int32)
		sent_batch =  snli_dev[j:j+eval_batch]
		a_batch, q_batch = stack_pairs(sent_batch)
		y_s = model.predict(a_batch, q_batch)
		preds.extend(y_s)
	f1_dev = accuracy_score(preds_true, preds)
	print 'dev accuracy_score={}'.format(f1_dev)
	if f1_dev > max_dev:
		preds = []
		for j in six.moves.range(0, n_test, eval_batch):
			sent_batch = snli_test[j:j+eval_batch]
			a_batch, q_batch = stack_pairs(sent_batch)
			y_s = model.predict(a_batch, q_batch)
			preds.extend(y_s)
		f1_test = accuracy_score(lbls_test, preds)
		print 'test accuracy_score={}'.format(f1_test)
		max_dev = f1_dev
		max_test = f1_test
		max_tr = f1_tr
		max_epch = i
		# print 'saving model...'
		# model.save(out_dir + '/NSE_MMA.' + str(i))
		# print 'loading back model...'
		# loadModel = NSE_MMA.load(out_dir + '/NSE_MMA.' + str(i), n_units, gpu)
	print "best results so far (dev):"
	print "epoch=",max_epch
	print "dev f1-score=",max_dev
	print "test f1-score=",max_test
	# if i - max_epch > 5:
	# 	print "No recent improvement on dev, early stopping..."
	# 	break