from io import open
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch import optim
import torch.nn.functional as F
from utils import *
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers=1):
		super(EncoderRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		#self.embedding.weight.data.copy_(torch.eye(hidden_size))
		#self.embedding.weight.requires_grad = False
		
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, batch_size, hidden):
		embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
		output = embedded
		for i in range(self.n_layers):
			output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self, batch_size):
		result = Variable(torch.zeros(1, batch_size, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

class DecoderATTN(nn.Module):
	
	def __init__(self, hidden_size, output_size, max_length, dropout_p=0.2, n_layers=1):
		super(DecoderATTN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		#self.embedding.weight.data.copy_(torch.eye(hidden_size))
		#self.embedding.weight.requires_grad = False
		
		self.attn = nn.Linear(hidden_size * 2, max_length)
		self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
		self.dropout = nn.Dropout(dropout_p)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, batch_size, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
		embedded = self.dropout(embedded)
		attn_weights = F.softmax(
									self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(1),
									encoder_outputs.transpose(0,1))
		output = torch.cat((embedded[0], attn_applied.squeeze()), 1)
		output = self.attn_combine(output).unsqueeze(0)
		for i in range(self.n_layers):
			output = F.relu(output)
			output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden, attn_weights

	def initHidden(self, batch_size):
		return torch.zeros(1, batch_size, self.hidden_size, deive=DEVICE)

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
	
	batch_size = input_variable.size()[0]
	encoder_hidden = encoder.initHidden(batch_size)

	input_variable = Variable(input_variable.transpose(0, 1))
	target_variable = Variable(target_variable.transpose(0, 1))

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_variable.size()[0]
	target_length = target_variable.size()[0]

	encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size, device=DEVICE)

	loss = 0
	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(
			input_variable[ei], batch_size, encoder_hidden)
		encoder_outputs[ei] = encoder_output[0]

	decoder_input = torch.tensor([[SOS_token] * batch_size], device=DEVICE)
	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	#use_teacher_forcing = True

	if use_teacher_forcing:
		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, _ = decoder(
				decoder_input, batch_size, decoder_hidden, encoder_outputs)
			loss += criterion(decoder_output, target_variable[di])
			decoder_input = target_variable[di]  # Teacher forcing
			
	else:
		# Without teacher forcing: use its own predictions as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, _ = decoder(
				decoder_input, batch_size, decoder_hidden, encoder_outputs)

			topv, topi = decoder_output.data.topk(1)
			decoder_input = topi.detach().to(DEVICE)
			# decoder_input = Variable(torch.LongTensor([[ni]]))

			loss += criterion(decoder_output, target_variable[di])

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss / target_length

def trainIters(encoder, decoder, epochs, train_loader, test_loader, lang, max_length, learning_rate=0.001):
	start = time.time()

	encoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, encoder.parameters()),
								  lr=learning_rate)
	decoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, decoder.parameters()),
								  lr=learning_rate)

	# data loader
	criterion = nn.NLLLoss()

	for epoch in range(epochs):
		print_loss_total = 0
		for batch_x, batch_y in train_loader:
			loss = train(batch_x, batch_y, encoder,
						 decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)			
			print_loss_total += loss

		print('%s (epoch: %d %d%%)' % (timeSince(start, (epoch+1)/epochs),
					epoch, (epoch+1)/epochs*100))
		print('total loss: %f'%(float(print_loss_total)))
		evaluate(encoder, decoder, test_loader, lang, max_length)
		print()

def evaluate(encoder, decoder, loader, lang, max_length):

	total = 0
	score = 0.0
	for batch_x, batch_y in loader:

		batch_size = batch_x.size()[0]
		encoder_hidden = encoder.initHidden(batch_size)

		input_variable = Variable(batch_x.transpose(0, 1))
		target_variable = Variable(batch_y.transpose(0, 1))

		input_length = input_variable.size()[0]
		target_length = target_variable.size()[0]

		output = torch.LongTensor(target_length, batch_size)

		encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size).to(DEVICE)
		
		for ei in range(input_length):
			encoder_output, encoder_hidden, = encoder(
				input_variable[ei], batch_size, encoder_hidden)
			encoder_outputs[ei] = encoder_output[0]

		decoder_input = torch.LongTensor([[SOS_token] * batch_size]).to(DEVICE)
		decoder_hidden = encoder_hidden

		for di in range(target_length):
			decoder_output, decoder_hidden, _ = decoder(
				decoder_input, batch_size, decoder_hidden, encoder_outputs)
			topv, topi = decoder_output.data.topk(1)
			output[di] = topi.transpose(0,1)
			decoder_input = topi

		output = output.transpose(0,1)
		for di in range(output.size()[0]):
			ignore = [EOS_token, PAD_token]
			sent = [lang.index2word[int(word)] for word in output[di] if word not in ignore]
			y = [[lang.index2word[int(word)] for word in batch_y[di] if word not in ignore]]
			score += sentence_bleu(y, sent)
			total += 1
	print('Test BLEU score '+str(score/total))


use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 80
teacher_forcing_ratio = 0.5
hidden_size = 512
batch_size = 48
epochs = 20
lr = 0.01

input_lang, output_lang, pairs, test_pairs = prepareData('iwslt16_en_de/train.en', 'iwslt16_en_de/train.de', 'iwslt16_en_de/dev.en', 'iwslt16_en_de/dev.de', reverse=True)
#pairs = pairs[0:10]
pairs = variablesFromPairs(input_lang, output_lang, pairs, MAX_LENGTH)
test_pairs = variablesFromPairs(input_lang, output_lang, test_pairs, MAX_LENGTH)
train_loader = torch.utils.data.DataLoader(pairs, 
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_pairs, 
	batch_size=batch_size, shuffle=False)
#print(input_lang.word2index)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
decoder = DecoderATTN(hidden_size, output_lang.n_words, MAX_LENGTH).to(DEVICE)

print('Training starts.')
trainIters(encoder, decoder, epochs, train_loader, test_loader, output_lang, MAX_LENGTH, learning_rate=lr)

