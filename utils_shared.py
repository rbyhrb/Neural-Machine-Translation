import re
import torch
import time
import math
import random
import unicodedata
from io import open
import sentencepiece as spm

SOS_token = 0
EOS_token = 1
PAD_token = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
	def __init__(self, file_name, vocab_size):
		self.sp = spm.SentencePieceProcessor()
		self.sp.load(file_name)
		self.size = vocab_size

	def encodeSentence(self, sentence):
		return self.sp.encode_as_ids(sentence)

	def sampleSentence(self, sentence):
		return self.sp.sample_encode_as_ids(sentence, 20, 0.1)

	def decodeSentence(self, codes):
		return self.sp.decode_ids(codes)


def readLangs(lang1, lang2, reverse=False):
	print("Reading lines...")

	# Read the files and split into lines
	lines_lang1 = open('%s' % (lang1), encoding='utf-8').\
		read().strip().split('\n')
	lines_lang2 = open('%s' % (lang2), encoding='utf-8').\
		read().strip().split('\n')
	# Split every line into pairs and normalize
	pairs = []
	pairs = [[lines_lang1[i], lines_lang2[i]] for i in range(len(lines_lang1))]


	# Reverse pairs, make Lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]

	return pairs


def indexesFromSentence(lang, sentence, sample=False):
	if sample:
		return lang.sampleSentence(sentence)
	else:
		return lang.encodeSentence(sentence)


def variableFromSentence(lang, sentence, max_length, start=False, sample=False):
	indexes = indexesFromSentence(lang, sentence, sample)
	if start:
		if len(indexes) > max_length-2:
			indexes = indexes[0:max_length-2]
		indexes = [SOS_token] + indexes + [EOS_token]
		indexes.extend([PAD_token] * (max_length - len(indexes)))
	else:
		if len(indexes) > max_length-1:
			indexes = indexes[0:max_length-1]
		indexes.append(EOS_token)
		indexes.extend([PAD_token] * (max_length - len(indexes)))
	indexes = torch.LongTensor(indexes).to(DEVICE)
	return indexes

def variablesFromPairs(lang, pairs, max_length, start=False, sample=False):
	res = []
	for pair in pairs:
		input_variable = variableFromSentence(lang, pair[0], max_length, start, sample)
		target_variable = variableFromSentence(lang, pair[1], max_length, start, sample)
		res.append((input_variable, target_variable))
	return res

def tempFromPairs(lang, pairs, max_length, start=False, sample=False):
	x = torch.zeros([len(pairs[0]),max_length], dtype=torch.int64).to(DEVICE)
	y = torch.zeros([len(pairs[0]),max_length], dtype=torch.int64).to(DEVICE)
	for i in range(len(pairs[0])):
		input_variable = variableFromSentence(lang, pairs[0][i], max_length, start, sample)
		target_variable = variableFromSentence(lang, pairs[1][i], max_length, start, sample)
		x[i,:] = input_variable
		y[i,:] = target_variable
	return x,y
