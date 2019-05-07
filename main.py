#####################################################
#	Author: Boya Ren @ Umass
#	Functions: Main funtion
#	Last modified: May 01, 2019
#####################################################

import argparse
from utils_shared import *
from transformer_model import *
import warnings
import os, sys
import torch
import unicodedata
from io import open
import sentencepiece as spm
import torch.utils.data as Data
from train import *

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='')


# -----------------------------Model parameters-----------------------------
parser.add_argument('--vocab_size', dest='vocab_size', type=int, default=25, help='Size of shared vovabulary in k, like 25k, 32k.')
parser.add_argument('--max_length', dest='max_length', type=int, default=120, help='The maximum length of token sequences.')
parser.add_argument('--d_model', dest='d_model', type=int, default=512, help='Model dimension, i.e. the size of embedding.')
parser.add_argument('--n_ff_hidden', dest='n_ff_hidden', type=int, default=2048, help='Number of hidden units in the feed forward layers.')
parser.add_argument('--n_heads', dest='n_heads', type=int, default=8, help='Number of heads.')
parser.add_argument('--n_layers', dest='n_layers', type=int, default=6, help='Number of layers.')
parser.add_argument('--dropout_p', dest='dropout_p', type=float, default=0.1, help='Dropout probability.')
parser.add_argument('--model', dest='model', default='normal', help='Choose between normal and bidirectional.')


# ----------------------------Training pamrameters----------------------------
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--optimizer', dest='optimizer', default='adabound', help='Choose between adabound and adam.')
parser.add_argument('--phase', dest='phase', default='train', help='Choose between train or test')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=30, help='Number of epochs.')
parser.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=4000, help='Warmup steps in the learning rate allocator.')
parser.add_argument('--step_add', dest='step_add', type=float, default=1.0, help='Step incrementation in the learning rate allocator.')
parser.add_argument('--save_path', dest='save_path', default='checkpoints', help='The path where models are saved.')
parser.add_argument('--from_scratch', dest='from_scratch', type=bool, default=False, help='Whether or not to train the model from scratch.')
scratch_parser = parser.add_mutually_exclusive_group(required=False)
scratch_parser.add_argument('--from-scratch', dest='from_scratch', action='store_true')
scratch_parser.add_argument('--continue', dest='from_scratch', action='store_false')
parser.add_argument('--use_dataset_A', dest='use_dataset_A', type=bool, default=True, help='Whether or not to use the given dataset.')
a_parser = parser.add_mutually_exclusive_group(required=False)
a_parser.add_argument('--use_A', dest='use_dataset_A', action='store_true')
a_parser.add_argument('--not_use_A', dest='use_dataset_A', action='store_false')
parser.add_argument('--use_dataset_B', dest='use_dataset_B', type=bool, default=True, help='Whether or not to use the extra dataset.')
b_parser = parser.add_mutually_exclusive_group(required=False)
b_parser.add_argument('--use_B', dest='use_dataset_B', action='store_true')
b_parser.add_argument('--not_use_B', dest='use_dataset_B', action='store_false')
parser.add_argument('--unigram_sampling', dest='unigram_sampling', type=bool, default=True, help='Whether or not to use unigram sampling.')
sample_parser = parser.add_mutually_exclusive_group(required=False)
sample_parser.add_argument('--sampling', dest='unigram_sampling', action='store_true')
sample_parser.add_argument('--no_sampling', dest='unigram_sampling', action='store_false')
parser.add_argument('--ratio', dest='ratio', type=float, default=1, help='The ratio between extra set and given set.')


# --------------------------Beam search pamrameters--------------------------
parser.add_argument('--beam_width', dest='beam_width', type=int, default=5, help='Beam width.')
parser.add_argument('--rp', dest='rp', type=float, default=0.6, help='Relative threshold puning in beam search.')
parser.add_argument('--rpl', dest='rpl', type=float, default=0.02, help='Relative local threshold puning in beam search.')
parser.add_argument('--mc', dest='mc', type=int, default=3, help='Maximum candidates per node in beam search.')


# -----------------------------Dataset locations-----------------------------
parser.add_argument('--train_a_1', dest='train_a_1', default='iwslt16_en_de/train.de', help='Training set A language 1.')
parser.add_argument('--train_a_2', dest='train_a_2', default='iwslt16_en_de/train.en', help='Training set A language 2.')
parser.add_argument('--train_b_1', dest='train_b_1', default='de-en/europarl-v7.de-en.de', help='Training set B language 1.')
parser.add_argument('--train_b_2', dest='train_b_2', default='de-en/europarl-v7.de-en.en', help='Training set B language 2.')
parser.add_argument('--test_1', dest='test_1', default='iwslt16_en_de/dev.de', help='Test set language 1.')
parser.add_argument('--test_2', dest='test_2', default='iwslt16_en_de/dev.en', help='Test set language 2.')

args = parser.parse_args()

def main():
	if not args.use_dataset_A and not args.use_dataset_B:
		print("No training dataset is designated.")
		return

	# Preprocessing
	if args.unigram_sampling:
		model_name = "u"+str(args.vocab_size)
		model_type = "unigram"
	else:
		model_name = "m"+str(args.vocab_size)
		model_type = "bpe"
	if not os.path.exists(model_name+".model"):
		# Read the files and split into lines
		lines_lang1 = open(args.train_a_1, encoding='utf-8').read().strip().split('\n')
		lines_lang2 = open(args.train_a_2, encoding='utf-8').read().strip().split('\n')
		file_name = "combined.txt"
		if os.path.exists(file_name):
			os.remove(file_name)
		f = open(file_name,"w+")
		for i in range(len(lines_lang1)):
			f.write(lines_lang1[i])
			f.write('\n')
			f.write(lines_lang2[i])
			f.write('\n')
		f.close()
		cmd = "--input=combined.txt --model_prefix="+model_name+" --vocab_size="+str(args.vocab_size)+"000 --pad_id=2 --unk_id=3 --bos_id=0 --eos_id=1 --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] --normalization_rule_name=nmt_nfkc --model_type="+model_type
		spm.SentencePieceTrainer.train(cmd)



	# Loading data
	lang = Lang(model_name+".model", args.vocab_size*1000)
	pairs = readLangs(args.train_a_1, args.train_a_2)
	test_pairs = readLangs(args.test_1, args.test_2)
	#pairs = pairs[0:160]
	if args.use_dataset_B:
		if args.ratio < 2:
			batch_size = int(args.batch_size*args.ratio)
		else:
			batch_size = args.batch_size
		extra_pairs = readLangs(args.train_b_1, args.train_b_2)
		extra_loader = Data.DataLoader(extra_pairs, batch_size=batch_size, shuffle=True)
	else:
		extra_loader = None

	# I am doing this because sometimes I do not use the whole dataset for validation
	#test_pairs = test_pairs[0:30]
	file_name = "en.dev"
	if os.path.exists(file_name):
		os.remove(file_name)
	f = open(file_name,"w+")
	for i in range(len(test_pairs)):
		f.write(test_pairs[i][1])
		f.write('\n')
	f.close()

	test_pairs = variablesFromPairs(lang, test_pairs, args.max_length, start=True)
	test_loader = Data.DataLoader(test_pairs, batch_size=args.batch_size, shuffle=False)
	print("Finished loading data.")

	if args.model == "normal":
		bi = False
	else:
		bi = True

	# Setting up the model
	model = Transformer(lang.size, args.d_model, args.n_layers, args.n_heads,
					args.dropout_p, args.max_length, args.n_ff_hidden, bi=bi).to(DEVICE)
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	print('Training starts.')

	# Training and testing
	if args.phase == "train":
		train(model, test_loader, lang, args, pairs, extra_loader)
	test(model, test_loader, lang, args)
	

if __name__ == '__main__':
	main()
