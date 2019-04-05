import re
import torch
import time
import math
import random
import unicodedata
from io import open
import nltk
nltk.download('punkt')

SOS_token = 0
EOS_token = 1
PAD_token = 2
use_cuda = torch.cuda.is_available()

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS", 2: 'PAD'}
		self.n_words = 3  # Count SOS and EOS

	def addSentence(self, sentence):
		#for word in sentence.split(' '):
		for word in nltk.word_tokenize(sentence):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1


# Lowercase, trim
def normalizeString(s):
	s = s.lower().strip()
	#s = re.sub(r"([,:\".!\?])", r" \1", s)
	return s


def readLangs(lang1, lang2, reverse=False):
	print("Reading lines...")

	# Read the files and split into lines
	lines_lang1 = open('%s' % (lang1), encoding='utf-8').\
		read().strip().split('\n')
	lines_lang2 = open('%s' % (lang2), encoding='utf-8').\
		read().strip().split('\n')

	# Split every line into pairs and normalize
	pairs = []
	pairs = [[normalizeString(lines_lang1[i]), normalizeString(lines_lang2[i])] for i in range(len(lines_lang1))]

	# Reverse pairs, make Lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)

	return input_lang, output_lang, pairs


def prepareData(lang1, lang2, test1, test2, reverse=False):
	input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
	print("Read %s sentence pairs" % len(pairs))
	_, _, test_pairs = readLangs(test1, test2, reverse)
	print("Read %s validation pairs" % len(test_pairs))
	print("Counting words...")
	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	for pair in test_pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	print("Counted words:")
	print(input_lang.name, input_lang.n_words)
	print(output_lang.name, output_lang.n_words)
	return input_lang, output_lang, pairs, test_pairs


def indexesFromSentence(lang, sentence):
	#return [lang.word2index[word] for word in sentence.split(' ')]
	return [lang.word2index[word] for word in nltk.word_tokenize(sentence)]


def variableFromSentence(lang, sentence, max_length):
	indexes = indexesFromSentence(lang, sentence)
	if len(indexes) > max_length-1:
		indexes = indexes[0:max_length-1]
	indexes.append(EOS_token)
	indexes.extend([PAD_token] * (max_length - len(indexes)))
	result = torch.LongTensor(indexes)
	if use_cuda:
		return result.cuda()
	else:
		return result


def variablesFromPairs(input_lang, output_lang, pairs, max_length):
	res = []
	for pair in pairs:
		input_variable = variableFromSentence(input_lang, pair[0], max_length)
		target_variable = variableFromSentence(output_lang, pair[1], max_length)
		res.append((input_variable, target_variable))
	return res


def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	if percent != 0.0:
		es = s / (percent)
		rs = es - s
		return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
	else:
		return '%s (- -m -s)' % (asMinutes(s))
