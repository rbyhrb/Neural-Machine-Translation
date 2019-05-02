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
from nltk.translate.bleu_score import sentence_bleu
import copy
import warnings
from adabound import AdaBound
import nltk
import os, sys, subprocess
from queue import PriorityQueue
#import gc

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('punkt')


class Embedder(nn.Module):
	def __init__(self, vocab_size, n_hidden):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, n_hidden)
	def forward(self, x):
		return self.embedding(x)


class PositionalEncoder(nn.Module):
	def __init__(self, n_hidden, max_length):
		super().__init__()
		self.n_hidden = n_hidden
		
		# create constant 'pe' matrix with values dependant on 
		# pos and i
		pe = torch.zeros(max_length, n_hidden)
		for pos in range(max_length):
			for i in range(0, n_hidden, 2):
				pe[pos, i] = \
				math.sin(pos / (10000 ** ((2 * i)/n_hidden)))
				pe[pos, i + 1] = \
				math.cos(pos / (10000 ** ((2 * (i + 1))/n_hidden)))
				
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		# make embeddings relatively larger
		x = x * math.sqrt(self.n_hidden)
		#add constant to embedding
		seq_len = x.size(1)
		x = x + Variable(self.pe[:,:seq_len], requires_grad=False).to(DEVICE)
		return x

class MultiHeadAttention(nn.Module):
	def __init__(self, n_heads, n_hidden, dropout_p):
		super().__init__()
		self.n_hidden = n_hidden
		self.d_k = n_hidden // n_heads
		self.h = n_heads
		
		self.q_linear = nn.Linear(n_hidden, n_hidden)
		self.v_linear = nn.Linear(n_hidden, n_hidden)
		self.k_linear = nn.Linear(n_hidden, n_hidden)
		self.dropout = nn.Dropout(dropout_p)
		self.out = nn.Linear(n_hidden, n_hidden)
	
	def forward(self, q, k, v, mask=None):
		
		bs = q.size(0)
		
		# perform linear operation and split into h heads		
		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
		
		# transpose to get dimensions bs * h * sl * n_hidden
		k = k.transpose(1,2)
		q = q.transpose(1,2)
		v = v.transpose(1,2)

		# calculate attention using function we will define next
		scores = attention(q, k, v, self.d_k, mask, self.dropout)
		
		# concatenate heads and put through final linear layer
		concat = scores.transpose(1,2).contiguous()\
		.view(bs, -1, self.n_hidden)
		
		output = self.out(concat)
		return output

def attention(q, k, v, d_k, mask=None, dropout=None):
	scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		mask = mask.unsqueeze(1)
		scores = scores.masked_fill(mask==0, -1e9)
	scores = F.softmax(scores, dim=-1)
	if dropout is not None:
		scores = dropout(scores)	
	output = torch.matmul(scores, v)
	return output

class FeedForward(nn.Module):
	def __init__(self, input_size, hidden, dropout_p):
		super().__init__() 
		# We set d_ff as a default to 2048
		self.linear_1 = nn.Linear(input_size, hidden)
		self.dropout = nn.Dropout(dropout_p)
		self.linear_2 = nn.Linear(hidden, input_size)
	def forward(self, x):
		x = self.dropout(F.relu(self.linear_1(x)))
		x = self.linear_2(x)
		return x

class Norm(nn.Module):
	def __init__(self, n_hidden, eps=1e-6):
		super().__init__()
	
		self.size = n_hidden
		# create two learnable parameters to calibrate normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.zeros(self.size))
		self.eps = eps
	def forward(self, x):
		norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / \
					(x.std(dim=-1, keepdim=True) + self.eps) + self.bias
		return norm

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
	def __init__(self, n_hidden, n_heads, dropout_p, n_ff_hidden):
		super().__init__()
		self.norm_1 = Norm(n_hidden)
		self.norm_2 = Norm(n_hidden)
		self.attn = MultiHeadAttention(n_heads, n_hidden, dropout_p)
		self.ff = FeedForward(n_hidden, n_ff_hidden, dropout_p)
		self.dropout_1 = nn.Dropout(dropout_p)
		self.dropout_2 = nn.Dropout(dropout_p)
		
	def forward(self, x, mask):
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.ff(x2))
		return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
	def __init__(self, n_hidden, n_heads, dropout_p, n_ff_hidden):
		super().__init__()
		self.norm_1 = Norm(n_hidden)
		self.norm_2 = Norm(n_hidden)
		self.norm_3 = Norm(n_hidden)
		
		self.dropout_1 = nn.Dropout(dropout_p)
		self.dropout_2 = nn.Dropout(dropout_p)
		self.dropout_3 = nn.Dropout(dropout_p)
		
		self.attn_1 = MultiHeadAttention(n_heads, n_hidden, dropout_p)
		self.attn_2 = MultiHeadAttention(n_heads, n_hidden, dropout_p)

		self.ff = FeedForward(n_hidden, n_ff_hidden, dropout_p)

	def forward(self, x, e_outputs, src_mask, trg_mask):
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
		x2 = self.norm_3(x)
		x = x + self.dropout_3(self.ff(x2))
		return x

# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
	def __init__(self, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden):
		super().__init__()
		self.n_layers = n_layers
		self.dropout = nn.Dropout(dropout_p)
		self.pe = PositionalEncoder(n_hidden, max_length)
		self.layers = get_clones(EncoderLayer(n_hidden, n_heads, dropout_p, n_ff_hidden), n_layers)
		self.norm = Norm(n_hidden)
	def forward(self, src, mask):
		x = self.dropout(self.pe(src))
		for i in range(self.n_layers):
			x = self.layers[i](x, mask)
		return self.norm(x)


class Decoder(nn.Module):
	def __init__(self, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden):
		super().__init__()
		self.n_layers = n_layers
		self.dropout = nn.Dropout(dropout_p)
		self.pe = PositionalEncoder(n_hidden, max_length)
		self.layers = get_clones(DecoderLayer(n_hidden, n_heads, dropout_p, n_ff_hidden), n_layers)
		self.norm = Norm(n_hidden)
	def forward(self, trg, e_outputs, src_mask, trg_mask):
		x = self.dropout(self.pe(trg))
		for i in range(self.n_layers):
			x = self.layers[i](x, e_outputs, src_mask, trg_mask)
		return self.norm(x)



class Transformer_shared(nn.Module):
	def __init__(self, vocab_size, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden):
		super().__init__()
		self.embed = Embedder(vocab_size, n_hidden)
		self.encoder = Encoder(n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden)
		self.decoder = Decoder(n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden)
		self.out = nn.Linear(n_hidden, vocab_size)
	def forward(self, src, trg, src_mask, trg_mask):
		e_outputs = self.embed(src)
		e_outputs = self.encoder(e_outputs, src_mask)
		d_outputs = self.embed(trg)
		d_outputs = self.decoder(d_outputs, e_outputs, src_mask, trg_mask)
		output = self.out(d_outputs)
		return output

def create_masks(batch_x, batch_y):
	source_msk = (batch_x != PAD_token).unsqueeze(1)
	target_msk = (batch_y != PAD_token).unsqueeze(1)
	size = batch_y.size(1) # get seq_len for matrix
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
		loss = loss.masked_select(non_pad_mask).mean()  # average later
	else:
		loss = F.cross_entropy(pred, gold, ignore_index=PAD_token)
	return loss

def train(model, n_epoch, pairs, ext_loader, test_loader, target_lang, max_length, lr, from_scratch):
	start = time.time()
			
	#optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
	optimizer = AdaBound(model.parameters(), lr=lr, final_lr=0.1)

	if not os.path.exists(SAVE_PATH):
		os.mkdir(SAVE_PATH)
	save_file = os.path.join(SAVE_PATH, SAVE_NAME)

	step = 0.0

	if not from_scratch:
		if os.path.exists(save_file):
			checkpoint = torch.load(save_file)
			model.load_state_dict(checkpoint['model_state_dict'])
			lr = checkpoint['lr']
			step = checkpoint['step']
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
			print("load successful!")
			checkpoint = []
		else:
			print("load unsuccessful!")
	best_val_bleu = 0
	ext_iter = iter(ext_loader)
	for epoch in range(n_epochs):

		model.train()
		print_loss_total = 0.0
		v_pairs = variablesFromPairs(lang, pairs, MAX_LENGTH, start=True, sample=True)
		train_loader = torch.utils.data.DataLoader(v_pairs, 
			batch_size=batch_size, shuffle=True)
		for batch_x, batch_y in train_loader:
			'''
			targets_y = batch_y[:, 1:].contiguous().view(-1)
			inputs_y = batch_y[:, :-1]
			src_mask, trg_mask = create_masks(batch_x, inputs_y)
			preds = model(batch_x, inputs_y, src_mask, trg_mask)
			optimizer.zero_grad()
			loss = cal_loss(preds.view(-1, preds.size(-1)), targets_y)		
			print_loss_total += float(loss)
			loss.backward()
			optimizer.step()

			
			try:
				ext_data = next(ext_iter) 
			except StopIteration:
				# StopIteration is thrown if dataset ends
 				# reinitialize data loader 
				ext_iter = iter(ext_loader)
				ext_data = next(ext_iter)
			batch_x, batch_y = tempFromPairs(lang, ext_data,
				MAX_LENGTH, start=True, sample=True)
			targets_y = batch_y[:, 1:].contiguous().view(-1)
			inputs_y = batch_y[:, :-1]
			src_mask, trg_mask = create_masks(batch_x, inputs_y)
			preds = model(batch_x, inputs_y, src_mask, trg_mask)
			optimizer.zero_grad()
			loss = cal_loss(preds.view(-1, preds.size(-1)), targets_y)		
			print_loss_total += float(loss)
			loss.backward()
			optimizer.step()
			
			'''
			step = step + STEP_ADD
			lr = min([step**(-0.5), step/(warmup_steps**1.5)]) / (n_hidden**0.5)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
		print('%s (epoch: %d %d%%)' % (timeSince(start, (epoch+1)/n_epochs),
					epoch, (epoch+1)/n_epochs*100))
		print('total loss: %f'%(print_loss_total))
		model.eval()

		curr_bleu = beamEval(model, test_loader, target_lang, max_length, 3, 2, 1)
		# print('%s' % (timeSince(start, (epoch+1)/n_epochs)))
		# curr_bleu = beamEval(model, test_loader, target_lang, max_length, 3,1,0.9)
		# print('%s' % (timeSince(start, (epoch+1)/n_epochs)))
		# curr_bleu = beamEval(model, test_loader, target_lang, max_length, 4,1,0.9)
		# print('%s' % (timeSince(start, (epoch+1)/n_epochs)))
		# curr_bleu = beamEval(model, test_loader, target_lang, max_length, 5, 1,0.9)
		# print('%s' % (timeSince(start, (epoch+1)/n_epochs)))
		# curr_bleu = beamEval(model, test_loader, target_lang, max_length, 6,1,0.9)
		# print('%s' % (timeSince(start, (epoch+1)/n_epochs)))
		# curr_bleu = beamEval(model, test_loader, target_lang, max_length, 7,1,0.9)
		print('%s' % (timeSince(start, (epoch+1)/n_epochs)))
		exit(0)
		

		if curr_bleu > best_val_bleu:
			best_val_bleu = curr_bleu
			torch.save({
									'model_state_dict': model.state_dict(),
									'optimizer_state_dict': optimizer.state_dict(),
									'lr': lr,
									'step': step,
									}, save_file)
			print("checkpoint saved!")
		print()

class BeamNode(object):
	def __init__(self, seq, logProb, length):
		self.seq = seq
		self.logProb = logProb
		self.length = length
	def eval(self, alpha=1.0):
		reward = 0
		return self.logProb / (float(self.length-1)+1e-6)

def beamEval(model, loader, lang, max_length, beam_width, n_candidate,cut_per):
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
			nodes = PriorityQueue() # queue to store candidate 
			flag = 0
			nodes.put((-node.eval(), flag, node))
			q_size = 1
			end_nodes = PriorityQueue()
			esize = 0
			done = False # sign for break 
			while True:
				path_extension = len (list(nodes.queue)) # extenstion loop 
				if path_extension == 0: # empty nodes, all into end_queue
					break
				mid_nodes = PriorityQueue() # queue to store all results each extenstion
				for path in range(path_extension):
					score, i, node = nodes.get(False)
					seq = node.seq
					log = node.logProb
					n = node.length
				
					if n==max_length or (seq[0,n-1]==EOS_token and n>2):
						end_nodes.put((score, i, node))

						esize += 1
						if esize >= n_candidate:
							done = True
							break
						else:
							continue  # next node extenstion

					target_msk = np.triu(np.ones((1, n, n)), k=1).astype('uint8')
					target_msk = Variable(torch.from_numpy(target_msk)==0).to(DEVICE)
					out = F.softmax(model.out(model.decoder(model.embed(seq[:,:n].to(DEVICE)),
												encoder_outputs[b,:,:], source_msk[b,:], target_msk)),dim=2)
					topv, topi = out.data.topk(beam_width)
					out_candids = topi[:,n-1,:].squeeze()
					probs = topv[:,n-1,:].squeeze()

					for next_w in range(beam_width):
						new_seq = seq.clone()
						new_seq[0, n] = out_candids[next_w]
						new_node = BeamNode(new_seq, log+np.log(float(probs[next_w])), n+1)
						score = -new_node.eval()
						flag += 1
						mid_nodes.put((score, flag, new_node))
						q_size += 1

				if done:  # enough candidates in end_queue 
					break
					
				results = []
				for q in range(beam_width):
					results.append(mid_nodes.get(False)) # top k candidates

				for p in range(len(results)):
					if results[p][0] > cut_per * results[0][0]:   #prune  
						nodes.put(results[p])

			if esize == 0:
				_, _, best_node = nodes.get(False)
			else:
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



SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
vocab_size = 32000
MAX_LENGTH = 120
STEP_ADD = 1
n_hidden = 512
n_ff_hidden = 2048
n_heads = 8
n_layers = 4
dropout_p = 0.1
n_epochs = 50
lr = 1e-4
batch_size = 32

from_scratch = False
warmup_steps = 4000
SAVE_PATH = 'checkpoints'
SAVE_NAME = 'transformer_sample_encoding-32k.pkl'

pairs = readLangs('iwslt16_en_de/train.en', 'iwslt16_en_de/train.de', reverse=True)
test_pairs = readLangs('iwslt16_en_de/dev.en', 'iwslt16_en_de/dev.de', reverse=True)
ext_pairs = readLangs('de-en/europarl-v7.de-en.en', 'de-en/europarl-v7.de-en.de', reverse=True)
lang = Lang('u32.model', vocab_size)
# pairs = pairs[0:200]
# test_pairs = test_pairs[0:300]
file_name = "en.dev"
if os.path.exists(file_name):
	os.remove(file_name)
f = open(file_name,"w+")
for i in range(len(test_pairs)):
	f.write(test_pairs[i][1])
	f.write('\n')
f.close()


test_pairs = variablesFromPairs(lang, test_pairs, MAX_LENGTH, start=True)
ext_loader = torch.utils.data.DataLoader(ext_pairs, 
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_pairs, 
	batch_size=batch_size, shuffle=False)
print("Finished loading data.")

model = Transformer_shared(lang.size, n_hidden, n_layers, n_heads, dropout_p, MAX_LENGTH, n_ff_hidden).to(DEVICE)

for p in model.parameters():
	if p.dim() > 1:
		nn.init.xavier_uniform_(p)

print('Training starts.')

train(model, n_epochs, pairs, ext_loader, test_loader, lang, MAX_LENGTH, lr, from_scratch)

