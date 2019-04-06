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
from utils import *
from nltk.translate.bleu_score import sentence_bleu
import copy
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
	def __init__(self, vocab_size, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden):
		super().__init__()
		self.n_layers = n_layers
		self.embed = Embedder(vocab_size, n_hidden)
		self.pe = PositionalEncoder(n_hidden, max_length)
		self.layers = get_clones(EncoderLayer(n_hidden, n_heads, dropout_p, n_ff_hidden), n_layers)
		self.norm = Norm(n_hidden)
	def forward(self, src, mask):
		x = self.embed(src)
		x = self.pe(x)
		for i in range(self.n_layers):
			x = self.layers[i](x, mask)
		return self.norm(x)


class Decoder(nn.Module):
	def __init__(self, vocab_size, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden):
		super().__init__()
		self.n_layers = n_layers
		self.embed = Embedder(vocab_size, n_hidden)
		self.pe = PositionalEncoder(n_hidden, max_length)
		self.layers = get_clones(DecoderLayer(n_hidden, n_heads, dropout_p, n_ff_hidden), n_layers)
		self.norm = Norm(n_hidden)
	def forward(self, trg, e_outputs, src_mask, trg_mask):
		x = self.embed(trg)
		x = self.pe(x)
		for i in range(self.n_layers):
			x = self.layers[i](x, e_outputs, src_mask, trg_mask)
		return self.norm(x)



class Transformer(nn.Module):
	def __init__(self, src_vocab_size, trg_vocab_size, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden):
		super().__init__()
		self.encoder = Encoder(src_vocab_size, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden)
		self.decoder = Decoder(trg_vocab_size, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden)
		self.out = nn.Linear(n_hidden, trg_vocab_size)
	def forward(self, src, trg, src_mask, trg_mask):
		e_outputs = self.encoder(src, src_mask)
		d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
		output = self.out(d_output)
		return output

def create_masks(batch_x, batch_y):
	source_msk = (batch_x != PAD_token).unsqueeze(1)
	target_msk = (batch_y != PAD_token).unsqueeze(1)
	size = batch_y.size(1) # get seq_len for matrix
	nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
	nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(DEVICE)
	target_msk = target_msk & nopeak_mask
	return source_msk, target_msk

def train(model, n_epochs, train_loader, test_loader, target_lang, max_length, lr):
	start = time.time()

	optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

	for epoch in range(n_epochs):

		model.train()
		print_loss_total = 0

		for batch_x, batch_y in train_loader:
			
			targets_y = batch_y[:, 1:].contiguous().view(-1)
			inputs_y = batch_y[:, :-1]
			src_mask, trg_mask = create_masks(batch_x, inputs_y)
			preds = model(batch_x, inputs_y, src_mask, trg_mask)

			optim.zero_grad()
			loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
						targets_y, ignore_index=PAD_token)		
			print_loss_total += loss
			loss.backward()
			optim.step()

		print('%s (epoch: %d %d%%)' % (timeSince(start, (epoch+1)/n_epochs),
					epoch, (epoch+1)/n_epochs*100))
		print('total loss: %f'%(float(print_loss_total)))
		model.eval()
		evaluate(model, test_loader, target_lang, max_length)
		print()

def evaluate(model, loader, lang, max_length):

	total = 0
	score = 0.0
	for batch_x, batch_y in loader:

		batch_size = batch_x.size()[0]
		source_msk = (batch_x != PAD_token).unsqueeze(1)
		encoder_outputs = model.encoder(batch_x, source_msk)
		outputs = torch.zeros(batch_size, max_length).type_as(batch_x).to(DEVICE)
		tmp = torch.LongTensor([[SOS_token] * batch_size]).squeeze()
		outputs[:,0] = tmp

		for i in range(1, max_length):
			target_msk = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
			target_msk = Variable(torch.from_numpy(target_msk)==0).to(DEVICE)
			out = model.out(model.decoder(outputs[:,:i], encoder_outputs, source_msk, target_msk))
			topv, topi = out.data.topk(1)
			outputs[:,i] = topi[:,i-1,:].squeeze()

		
		for b in range(batch_size):
			ignore = [EOS_token, PAD_token]
			sent = []
			for word in outputs[b]:
				if word not in ignore:
					sent.append(lang.index2word[int(word)])
				if word == EOS_token:
					break;
			y = [[lang.index2word[int(word)] for word in batch_y[b] if word not in ignore]]
			score += sentence_bleu(y, sent)
			total += 1
	print('Test BLEU score '+str(score/total))


SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 80
n_hidden = 512
n_ff_hidden = 2048
n_heads = 8
n_layers = 6
dropout_p = 0.3
n_epochs = 50
lr = 1e-4
batch_size = 32

input_lang, output_lang, pairs, test_pairs = prepareData('iwslt16_en_de/train.en', 'iwslt16_en_de/train.de', 'iwslt16_en_de/dev.en', 'iwslt16_en_de/dev.de', reverse=True)
#pairs = pairs[0:10]
pairs = variablesFromPairs(input_lang, output_lang, pairs, MAX_LENGTH, start=True)
test_pairs = variablesFromPairs(input_lang, output_lang, test_pairs, MAX_LENGTH, start=True)
train_loader = torch.utils.data.DataLoader(pairs, 
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_pairs, 
	batch_size=batch_size, shuffle=False)
#print(input_lang.word2index)

model = Transformer(input_lang.n_words, output_lang.n_words, n_hidden, n_layers, n_heads, dropout_p, MAX_LENGTH, n_ff_hidden).to(DEVICE)

for p in model.parameters():
	if p.dim() > 1:
		nn.init.xavier_uniform_(p)

print('Training starts.')

train(model, n_epochs, train_loader, test_loader, output_lang, MAX_LENGTH, lr)

