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
import os, sys

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
		self.pe = PositionalEncoder(n_hidden, max_length)
		self.layers = get_clones(EncoderLayer(n_hidden, n_heads, dropout_p, n_ff_hidden), n_layers)
		self.norm = Norm(n_hidden)
	def forward(self, src, mask):
		x = self.pe(src)
		for i in range(self.n_layers):
			x = self.layers[i](x, mask)
		return self.norm(x)


class Decoder(nn.Module):
	def __init__(self, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden):
		super().__init__()
		self.n_layers = n_layers
		self.pe = PositionalEncoder(n_hidden, max_length)
		self.layers = get_clones(DecoderLayer(n_hidden, n_heads, dropout_p, n_ff_hidden), n_layers)
		self.norm = Norm(n_hidden)
	def forward(self, trg, e_outputs, src_mask, trg_mask):
		x = self.pe(trg)
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

''' Calculate cross entropy loss, apply label smoothing if needed. '''
def cal_loss(preds, targets_y, smoothing=True):

    if smoothing:
        eps = 0.1
        n_class = preds.size(1)
        one_hot = torch.zeros_like(preds).scatter(1, targets_y.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(preds, dim=1)
        
        non_pad_mask = targets_y.ne(PAD_token)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        loss = F.cross_entropy(preds, targets_y, ignore_index=PAD_token)
    
    return loss
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def train(model, n_epochs, train_loader, test_loader, target_lang, max_length, lr, from_scratch):
	start = time.time()
			
	#optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
	optimizer = AdaBound(model.parameters(), lr=lr, final_lr=0.1)

	if not os.path.exists(SAVE_PATH):
		os.mkdir(SAVE_PATH)
	save_file = os.path.join(SAVE_PATH, SAVE_NAME)

	if not from_scratch:
		if os.path.exists(save_file):
			checkpoint = torch.load(save_file)
			model.load_state_dict(checkpoint['model_state_dict'])
			lr = checkpoint['lr']
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
			print("load successful!")
			checkpoint = []
		else:
			print("load unsuccessful!")
	step = 0.0
	best_val_bleu = 0
	for epoch in range(n_epochs):

		model.train()
		print_loss_total = 0

		for batch_x, batch_y in train_loader:
			
			targets_y = batch_y[:, 1:].contiguous().view(-1)
			inputs_y = batch_y[:, :-1]
			src_mask, trg_mask = create_masks(batch_x, inputs_y)
			preds = model(batch_x, inputs_y, src_mask, trg_mask)

			optimizer.zero_grad()
			'''add label smoothing'''
			loss = cal_loss(preds.view(-1, preds.size(-1)), targets_y, smoothing=True)
			'''end'''
			print_loss_total += loss
			loss.backward()
			optimizer.step()
			step = step + 0.5
			lr = min([step**(-0.5), step/(warmup_steps**1.5)]) / (n_hidden**0.5)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		print('%s (epoch: %d %d%%)' % (timeSince(start, (epoch+1)/n_epochs),
					epoch, (epoch+1)/n_epochs*100))
		print('total loss: %f'%(float(print_loss_total)))
		model.eval()
		curr_bleu = evaluate(model, test_loader, target_lang, max_length)

		if curr_bleu > best_val_bleu:
			best_val_bleu = curr_bleu
			torch.save({
									'model_state_dict': model.state_dict(),
									'optimizer_state_dict': optimizer.state_dict(),
									'lr': lr,
									}, save_file)
			print("checkpoint saved!")
		print()

def evaluate(model, loader, lang, max_length):

	total = 0
	score = 0.0
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
			out = model.out(model.decoder(model.embed(outputs[:,:i]), encoder_outputs, source_msk, target_msk))
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
			y = [int(word) for word in batch_y[b] if word not in ignore]
			sent = nltk.word_tokenize(lang.decodeSentence(sent))
			y = [nltk.word_tokenize(lang.decodeSentence(y))]
			score += sentence_bleu(y, sent)
			total += 1
	print('Test BLEU score '+str(score/total))
	return score/total


SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
vocab_size = 36000
MAX_LENGTH = 100
n_hidden = 512
n_ff_hidden = 2048
n_heads = 8
n_layers = 6
dropout_p = 0.2
n_epochs = 50
lr = 2e-4
batch_size = 32
from_scratch = True
warmup_steps = 4000
SAVE_PATH = 'checkpoints'
SAVE_NAME = 'transformer_shared.pkl'

lang, pairs, test_pairs = readLangs('iwslt16_en_de/train.en', 'iwslt16_en_de/train.de', 'iwslt16_en_de/dev.en', 'iwslt16_en_de/dev.de', 'm.model', vocab_size, reverse=False)
pairs = pairs[0:2]
test_pairs = test_pairs[0:1]
# test_pairs = test_pairs[1:3000]
pairs = variablesFromPairs(lang, pairs, MAX_LENGTH, start=True)
test_pairs = variablesFromPairs(lang, test_pairs, MAX_LENGTH, start=True)
train_loader = torch.utils.data.DataLoader(pairs, 
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_pairs, 
	batch_size=batch_size, shuffle=False)

model = Transformer_shared(lang.size, n_hidden, n_layers, n_heads, dropout_p, MAX_LENGTH, n_ff_hidden).to(DEVICE)

for p in model.parameters():
	if p.dim() > 1:
		nn.init.xavier_uniform_(p)

print('Training starts.')

train(model, n_epochs, train_loader, test_loader, lang, MAX_LENGTH, lr, from_scratch)

