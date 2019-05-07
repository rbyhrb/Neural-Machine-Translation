#####################################################
#	Author: Boya Ren @ Umass
#	Functions: Training and evaluation of the model
#	Last modified: May 01, 2019
#####################################################

from io import open
import random
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as Data
from torch import optim
import torch.nn.functional as F
from utils_shared import *
from helper import *
import copy
import warnings
from adabound import AdaBound
import nltk
import os, sys, subprocess
from queue import PriorityQueue
from transformer_model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('punkt')
SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

# ----------------------------Some helper Functions----------------------------

def create_masks(batch_x, batch_y):
	source_msk = (batch_x != PAD_token).unsqueeze(1)
	target_msk = (batch_y != PAD_token).unsqueeze(1)
	size = batch_y.size(1)
	nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
	nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(DEVICE)
	target_msk = target_msk & nopeak_mask
	return source_msk, target_msk

def cal_loss(pred, gold, smoothing=True):
	''' Calculate cross entropy loss, apply label smoothing if needed. '''
	gold = gold.contiguous().view(-1)
	if smoothing:
		eps = 0.1
		n_class = pred.size(1)
		one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
		one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
		log_prb = F.log_softmax(pred, dim=1)
		non_pad_mask = gold.ne(PAD_token)
		loss = -(one_hot * log_prb).sum(dim=1)
		loss = loss.masked_select(non_pad_mask).mean()
	else:
		loss = F.cross_entropy(pred, gold, ignore_index=PAD_token)
	return loss

class BeamNode(object):
	def __init__(self, seq, logProb, length):
		self.seq = seq
		self.logProb = logProb
		self.length = length
	def eval(self, alpha=0.7):
		norm = ((self.length+5)/6) ** alpha
		return self.logProb / norm

def print_model(args):
	print()
	print(args.model+" transformer model")
	print("maximum sequence lenghth: %d"%(args.max_length))
	print("model dimension: %d"%(args.d_model))
	print("number of heads: %d"%(args.n_heads))
	print("number of hidden units in feadforward layers: %d"%(args.n_ff_hidden))
	print("number of layers: %d"%(args.n_layers))
	print("dropout rate: %.1f"%(args.dropout_p))
	print("vocabulary size: %dk"%(args.vocab_size))
	print("optimizer: "+args.optimizer)
	print("batch size: %d"%(args.batch_size))
	print("warmup steps: %d"%(args.warmup_steps))
	print("step increment: %.3f"%(args.step_add))
	if args.use_dataset_A:
		print("using the given dataset")
	if args.use_dataset_B:
		print("using the extra dataset")
	if args.use_dataset_A and args.use_dataset_B:
		if args.ratio >= 2:
			print("the ratio between two datasets %d"%(int(args.ratio)))
		else:
			print("the ratio between two datasets %.2f"%(args.ratio))
	print()

def print_beam(args):
	print("beam width: %d"%(args.beam_width))
	print("relative threshold pruning: %.2f"%(args.rp))
	print("relative local threshold pruning: %.2f"%(args.rpl))
	print("maximum candidate: %.1f"%(args.mc))
	
def get_name(args):
	save_name = "%s_transformer_model_%d_%d_%d_%d_%d_%dk.pkl"%(args.model, args.max_length,
					args.d_model, args.n_heads, args.n_ff_hidden, args.n_layers, args.vocab_size)
	save_name = os.path.join(args.save_path,save_name)
	return save_name

# --------------------------------Training--------------------------------

def train(model, test_loader, lang, args, pairs, extra_loader):
	start = time.time()
	
	if args.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
	elif args.optimizer == "adabound":
		optimizer = AdaBound(model.parameters(), lr=0.0001, final_lr=0.1)
	else:
		print("unknow optimizer.")
		exit(0)

	print_model(args)

	save_name = get_name(args)
	if not os.path.exists(args.save_path):
		os.mkdir(args.save_path)
	print("the model is saved in: "+save_name)

	n_epochs = args.n_epochs
	step = 0.0
	begin_epoch = 0
	best_val_bleu = 0
	if not args.from_scratch:
		if os.path.exists(save_name):
			checkpoint = torch.load(save_name)
			model.load_state_dict(checkpoint['model_state_dict'])
			lr = checkpoint['lr']
			step = checkpoint['step']
			begin_epoch = checkpoint['epoch'] + 1
			best_val_bleu = checkpoint['bleu']
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
			print("load successful!")
			checkpoint = []
		else:
			print("load unsuccessful!")

	if args.use_dataset_B:
		extra_iter = iter(extra_loader)
		num_iter = 1
		if args.ratio >= 2:
			num_iter = int(args.ratio)
	else:
		extra_iter = None
		num_iter = 0

	for epoch in range(begin_epoch ,n_epochs):

		model.train()

		train_loader = Data.DataLoader(pairs, batch_size=args.batch_size, shuffle=True)

		print_loss_total, step, extra_iter, lr = train_epoch(model, lang, args, train_loader, 
															extra_loader, extra_iter, num_iter, optimizer, step)
			
		print('total loss: %f'%(print_loss_total))
		model.eval()
		curr_bleu = evaluate(model, test_loader, lang, args.max_length)
		print('%s (epoch: %d %d%%)' % (timeSince(start, (epoch+1-begin_epoch)/(n_epochs-begin_epoch)), epoch, (epoch+1-begin_epoch)/(n_epochs-begin_epoch)*100))

		if curr_bleu > best_val_bleu:
			best_val_bleu = curr_bleu
			torch.save({
									'model_state_dict': model.state_dict(),
									'optimizer_state_dict': optimizer.state_dict(),
									'lr': lr,
									'step': step,
									'epoch': epoch,
									'bleu': curr_bleu,
									}, save_name)
			print("checkpoint saved!")
		print()

def step_forward(model, optimizer, batch_x, batch_y):
	targets_y = batch_y[:, 1:].contiguous().view(-1)
	inputs_y = batch_y[:, :-1]
	src_mask, trg_mask = create_masks(batch_x, inputs_y)
	preds = model(batch_x, inputs_y, src_mask, trg_mask)
	optimizer.zero_grad()
	loss = cal_loss(preds.view(-1, preds.size(-1)), targets_y)
	loss.backward()
	optimizer.step()
	return float(loss)

def step_reverse(model, optimizer, batch_x, batch_y):
	targets_x = batch_x[:, 1:].contiguous().view(-1)
	inputs_x = batch_x[:, :-1]
	src_mask, trg_mask = create_masks(batch_y, inputs_x)
	preds = model(batch_y, inputs_x, src_mask, trg_mask, reverse=True)
	optimizer.zero_grad()
	loss = cal_loss(preds.view(-1, preds.size(-1)), targets_x)
	loss.backward()
	optimizer.step()
	return float(loss)

def train_epoch(model, lang, args, train_loader, extra_loader, extra_iter, num_iter, optimizer, step):
	print_loss_total = 0.0
	for pair in train_loader:
		if args.use_dataset_A:
			if args.unigram_sampling:
				batch_x, batch_y = tempFromPairs(lang, pair,
														args.max_length, start=True, sample=True)
			else:
				batch_x, batch_y = tempFromPairs(lang, pair,
														args.max_length, start=True)
			
			loss = step_forward(model, optimizer, batch_x, batch_y)
			print_loss_total += loss
					
			if args.model == "bidirectional":
				loss = step_reverse(model, optimizer, batch_x, batch_y)
				print_loss_total += float(loss)

		if args.use_dataset_B:
			for n_i in range(num_iter):
				try:
					extra_data = next(extra_iter) 
				except StopIteration:
					extra_iter = iter(extra_loader)
					extra_data = next(extra_iter)
				if args.unigram_sampling:
					batch_x, batch_y = tempFromPairs(lang, extra_data,
														args.max_length, start=True, sample=True)
				else:
					batch_x, batch_y = tempFromPairs(lang, extra_data,
														args.max_length, start=True)
				loss = step_forward(model, optimizer, batch_x, batch_y)
				print_loss_total += loss

				if args.model == "bidirectional":
					loss = step_reverse(model, optimizer, batch_x, batch_y)
					print_loss_total += float(loss)

		step = step + args.step_add
		lr = min([step**(-0.5), step/(args.warmup_steps**1.5)]) / (args.d_model**0.5)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	return print_loss_total, step, extra_iter, lr

# --------------------------------Testing--------------------------------

def test(model, test_loader, lang, args):
	start = time.time()

	save_file = get_name(args)
	if os.path.exists(save_file):
		checkpoint = torch.load(save_file)
		model.load_state_dict(checkpoint['model_state_dict'])
		print("loaded model from "+save_file)
		print("greedy search BLEU score: %.1f" %(checkpoint['bleu']))
		checkpoint = []
	else:
		print("no model file found")
		exit(0)

	model.eval()
	print_beam(args)
	curr_bleu = beamEval(model, test_loader, lang, args)
	print(timeDiff(start))



# Beam search
def beamEval(model, loader, lang, args):
	max_length = args.max_length
	beam_width = args.beam_width
	rp = args.rp
	rpl = args.rpl
	mc = args.mc
	trans_file = "out.dev"
	if os.path.exists(trans_file):
		os.remove(trans_file)
	f = open(trans_file, "w+")
	
	for batch_x, batch_y in loader:
		batch_size = batch_x.size()[0]
		source_msk = (batch_x != PAD_token).unsqueeze(1)
		encoder_outputs = model.encoder(model.embed(batch_x), source_msk)

		for b in range(batch_size):
			out = torch.zeros(1, max_length).type_as(batch_x)
			out[0,0] = SOS_token
			node = BeamNode(out, 0, 1)
			nodes = PriorityQueue()
			flag = 0
			nodes.put((-node.eval(), flag, node))
			end_nodes = PriorityQueue()

			while not nodes.empty():
				new_nodes = PriorityQueue()
				for i_width in range(beam_width):
					if nodes.empty():
						continue
					score, i, node = nodes.get(False)			
					if i_width == 0:
						best_score = score
					else:
						if score > best_score / rp:
							break				
					seq = node.seq
					log = node.logProb
					n = node.length
					
					if n==max_length or (seq[0,n-1]==EOS_token and n>2):
						end_nodes.put((score, i, node))
						continue

					target_msk = np.triu(np.ones((1, n, n)), k=1).astype('uint8')
					target_msk = Variable(torch.from_numpy(target_msk)==0).to(DEVICE)
					out = F.softmax(model.out(model.decoder(model.embed(seq[:,:n].to(DEVICE)),
													target_msk, encoder_outputs[b,:,:], source_msk[b,:])),dim=2)
					topv, topi = out.data.topk(beam_width)
					out_candids = topi[:,n-1,:].squeeze()
					probs = topv[:,n-1,:].squeeze()
					
					if n == 1:
						local_width = beam_width
					else:
						local_width = min(mc, beam_width)
					for next_w in range(local_width):
						new_seq = seq.clone()
						new_seq[0, n] = out_candids[next_w]
						new_node = BeamNode(new_seq, log+math.log(float(probs[next_w])), n+1)
						score = -new_node.eval()					
						if next_w == 0:
							local_best_score = score
						else:
							if score > local_best_score / rpl:
								break				
						flag += 1
						new_nodes.put((score, flag, new_node))
				nodes = new_nodes
			
			_, _, best_node = end_nodes.get()
			best_seq = best_node.seq.squeeze()

			ignore = [EOS_token, PAD_token, UNK_token, SOS_token]
			sent = []
			for word in best_seq:
				if word not in ignore:
					sent.append(int(word))
				if word == EOS_token:
					break

			sent = nltk.word_tokenize(lang.decodeSentence(sent))
			sent = "".join([token+" " for token in sent])
			f.write(sent[:-1])
			f.write('\n')
			
	f.close()
	out = subprocess.check_output(["bash", "./compute_bleu.sh"], close_fds=True)
	out = str(out).split(' ')
	bleu = out[2]
	print('Test BLEU score '+bleu)
	return float(bleu)



# Greedy search
def evaluate(model, loader, lang, max_length):
	trans_file = "out.dev"
	if os.path.exists(trans_file):
		os.remove(trans_file)
	f = open(trans_file, "w+")

	for batch_x, batch_y in loader:
		batch_size = batch_x.size()[0]
		source_msk = (batch_x != PAD_token).unsqueeze(1)
		encoder_outputs = model.encoder(model.embed(batch_x), source_msk)
		outputs = torch.zeros(batch_size, max_length).type_as(batch_x).to(DEVICE)
		tmp = torch.LongTensor([[SOS_token] * batch_size]).squeeze()
		outputs[:,0] = tmp

		for i in range(1, max_length):
			target_msk = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
			target_msk = Variable(torch.from_numpy(target_msk)==0).to(DEVICE)
			out = model.out(model.decoder(model.embed(outputs[:,:i]), 
											target_msk, encoder_outputs, source_msk))
			topv, topi = out.data.topk(1)
			outputs[:,i] = topi[:,i-1,:].squeeze()
		
		for b in range(batch_size):
			ignore = [EOS_token, PAD_token, UNK_token, SOS_token]
			sent = []
			for word in outputs[b]:
				if word not in ignore:
					sent.append(int(word))
				if word == EOS_token:
					break

			sent = nltk.word_tokenize(lang.decodeSentence(sent))
			sent = "".join([token+" " for token in sent])
			f.write(sent[:-1])
			f.write('\n')
			
	f.close()
	out = subprocess.check_output(["bash", "./compute_bleu.sh"], close_fds=True)
	out = str(out).split(' ')
	bleu = out[2]
	print('Test BLEU score '+bleu)
	return float(bleu)

