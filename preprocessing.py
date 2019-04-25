import unicodedata
from io import open
import sentencepiece as spm
import os

# Read the files and split into lines
lines_lang1 = open('iwslt16_en_de/train.en', encoding='utf-8').\
	read().strip().split('\n')
lines_lang2 = open('iwslt16_en_de/train.de', encoding='utf-8').\
	read().strip().split('\n')
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

cmd = '--input=combined.txt --model_prefix=m32 --vocab_size=32000 --pad_id=2 --unk_id=3 --bos_id=0 --eos_id=1 --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] --normalization_rule_name=nmt_nfkc --model_type=bpe'
spm.SentencePieceTrainer.train(cmd)


