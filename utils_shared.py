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

	def decodeSentence(self, codes):
		return self.sp.decode_ids(codes)


def readLangs(lang1, lang2, test_lang1, test_lang2, file_name, vocab_size, reverse):
	print("Reading lines...")

	# Read the files and split into lines
	lines_lang1 = open('%s' % (lang1), encoding='utf-8').\
		read().strip().split('\n')
	lines_lang2 = open('%s' % (lang2), encoding='utf-8').\
		read().strip().split('\n')
	# Split every line into pairs and normalize
	pairs = []
	pairs = [[lines_lang1[i], lines_lang2[i]] for i in range(len(lines_lang1))]

	# Read the files and split into lines
	lines_lang1 = open('%s' % (test_lang1), encoding='utf-8').\
		read().strip().split('\n')
	lines_lang2 = open('%s' % (test_lang2), encoding='utf-8').\
		read().strip().split('\n')
	test_pairs = []
	test_pairs = [[lines_lang1[i], lines_lang2[i]] for i in range(len(lines_lang1))]

	# Reverse pairs, make Lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		test_pairs = [list(reversed(p)) for p in test_pairs]
	lang = Lang(file_name, vocab_size)

	print("Finished building Vocabularies.")
	return lang, pairs, test_pairs


def indexesFromSentence(lang, sentence):
	return lang.encodeSentence(sentence)


def variableFromSentence(lang, sentence, max_length, start=False):
	indexes = indexesFromSentence(lang, sentence)
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

	result = torch.LongTensor(indexes).to(DEVICE)
	return result

def variablesFromPairs(lang, pairs, max_length, start=False):
	res = []
	for pair in pairs:
		input_variable = variableFromSentence(lang, pair[0], max_length, start)
		target_variable = variableFromSentence(lang, pair[1], max_length, start)
		res.append((input_variable, target_variable))
	return res
