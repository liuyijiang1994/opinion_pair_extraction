log_path = "./output/logs"
max_pieces_length = 100
max_word_length = 90
# dataset = './data/2014LapStandard.pt'
dataset = './data/2014ResStandard.pt'
# dataset = './data/2015ResStandard.pt'
# bert_path = 'F:\\resources\pretrained_model\\bert-base-uncased'
bert_path = 'bert-base-uncased'
output_dir = "output/checkpoint"  # checkpoint和预测输出文件夹

train_batch_size = 32
test_batch_size = 32
gradient_accumulation_steps = 1
num_train_epochs = 30

device = 'cuda'

labelDic = {"O": 0, "B-T": 1, "I-T": 2, "B-P": 3, "I-P": 4}
deprel = ['cop', 'case', 'obl:npmod', 'mark', 'nmod:npmod', 'advmod', 'nummod', 'aux', 'goeswith', 'det', 'nmod',
          'obl:tmod', 'ccomp', 'fixed', 'nsubj', 'list', 'appos', 'nmod:tmod', 'obl', 'discourse', 'cc:preconj', 'root',
          'parataxis', 'nsubj:pass', 'cc', 'det:predet', 'advcl', 'flat', 'compound', 'punct', 'expl', 'nmod:poss',
          'orphan', 'obj', 'iobj', 'vocative', 'acl', 'acl:relcl', 'compound:prt', 'conj', 'csubj', 'csubj:pass',
          'reparandum', 'aux:pass', 'xcomp', 'amod']
pos = ['PAD', 'ADJ', 'AUX', 'PRON', 'PUNCT', 'VERB', 'ADV', 'NOUN', 'DET', 'CCONJ', 'PART', 'NUM', 'INTJ', 'SYM', 'ADP',
       'SCONJ', 'X', 'PROPN']
posDic = {p: idx for idx, p in enumerate(pos)}
deprelDic = {d: idx for idx, d in enumerate(deprel)}
