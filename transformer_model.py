import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy

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


class DecoderLayer(nn.Module):
	def __init__(self, n_hidden, n_heads, dropout_p, n_ff_hidden, bi=True):
		super().__init__()
		self.norm_1 = Norm(n_hidden)
		self.norm_2 = Norm(n_hidden)
		self.norm_3 = Norm(n_hidden)
		
		self.dropout_1 = nn.Dropout(dropout_p)
		self.dropout_3 = nn.Dropout(dropout_p)
		
		self.attn_1 = MultiHeadAttention(n_heads, n_hidden, dropout_p)

		if bi:
			self.attn_2 = MultiHeadAttention(n_heads, n_hidden, dropout_p)
			self.dropout_2 = nn.Dropout(dropout_p)
			self.norm_3 = Norm(n_hidden)

		self.ff = FeedForward(n_hidden, n_ff_hidden, dropout_p)

	def forward(self, x, trg_mask, e_outputs=None, src_mask=None):
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
		x2 = self.norm_2(x)
		if e_outputs is not None:
			x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
			x2 = self.norm_3(x)
		x = x + self.dropout_3(self.ff(x2))
		return x

def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Decoder(nn.Module):
	def __init__(self, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden, bi=True):
		super().__init__()
		self.n_layers = n_layers
		self.dropout = nn.Dropout(dropout_p)
		self.pe = PositionalEncoder(n_hidden, max_length)
		self.layers = get_clones(DecoderLayer(n_hidden, n_heads, dropout_p, n_ff_hidden, bi=bi), n_layers)
		self.norm = Norm(n_hidden)
	def forward(self, trg, trg_mask, e_outputs=None, src_mask=None):
		x = self.dropout(self.pe(trg))
		for i in range(self.n_layers):
			x = self.layers[i](x, trg_mask, e_outputs, src_mask)
		return self.norm(x)



class Transformer(nn.Module):
	def __init__(self, vocab_size, n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden, bi=True):
		super().__init__()
		self.embed = Embedder(vocab_size, n_hidden)
		self.encoder = Decoder(n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden, bi=bi)
		self.decoder = Decoder(n_hidden, n_layers, n_heads, dropout_p, max_length, n_ff_hidden)
		self.out_rev = nn.Linear(n_hidden, vocab_size)
		self.out = nn.Linear(n_hidden, vocab_size)
	def forward(self, src, trg, src_mask, trg_mask, reverse=False):
		inputs = self.embed(src)
		if reverse:
			e_outputs = self.decoder(inputs, src_mask)
			d_outputs = self.embed(trg)
			d_outputs = self.encoder(d_outputs, trg_mask, e_outputs, src_mask)
			output = self.out_rev(d_outputs)
		else:
			e_outputs = self.encoder(inputs, src_mask)
			d_outputs = self.embed(trg)
			d_outputs = self.decoder(d_outputs, trg_mask, e_outputs, src_mask)
			output = self.out(d_outputs)
		return output
