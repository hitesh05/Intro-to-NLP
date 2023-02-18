# -*- coding: utf-8 -*-
import regex as re
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import numpy as np
from collections import defaultdict
import sys

model_path = str(sys.argv[1])

class Tokeniser():
    def __init__(self):
        pass
    
    def convert_text_to_string(self):
        self.text = '\n'.join(self.text)
        self.text = re.sub(r'\n+', r' ', self.text)
    
    def lower_case(self):
        self.text = self.text.lower()
    
    def change_urls(self):
        self.text = re.sub(r'http[s]?\S*[\s | \n]', r'<URL>', self.text) # changing urls
        
    def change_hashtags(self):
        self.text = re.sub(r'\b#\w*[a-z0-9]+\w*', r'<HASHTAG>', self.text) # changing hashtags
        
    def change_mentions(self):
        self.text = re.sub(r'@(\w+)', r'<MENTION>', self.text) # changing mentions
        
    # def change_nums(self):
    #     self.text = re.sub(r'\d+', r'<NUM>', self.text)
    
    def remove_punctuations(self):
        self.text = re.sub(r'[^\w+^\.^\?^\!\s]', r'', self.text)
        return self.text
        
    def handle_special_cases(self):
        self.text = re.sub(r'(\w+)\'bout', r'\1 about', self.text)
        self.text=re.sub(r"won\'t","will not",self.text)
        self.text = re.sub(r'(\w+)\'t', r'\1 not', self.text)
        self.text = re.sub(r'(\w+)\'s', r'\1 is', self.text)
        self.text = re.sub(r'(\w+)\'re', r'\1 are', self.text)
        self.text = re.sub(r'(\w+)\'ll', r'\1 will', self.text)
        self.text = re.sub(r'(\w+)\'d', r'\1 would', self.text)
        self.text = re.sub(r'(\w+)\'ve', r'\1 have', self.text)
        self.text = re.sub(r'([iI])\'m', r'\1 am', self.text)
        
    def remove_stupid_fullstop(self):
        self.text=re.sub("Mr\s*\.","Mr",self.text)
        self.text=re.sub("Ms\s*\.","Ms",self.text)
        self.text=re.sub("Mrs\s*\.","Mrs",self.text)
        self.text=re.sub("Miss\s*\.","Miss",self.text)
        
    def remove_extra_spaces(self):
        self.text = re.sub(' +', ' ', self.text)
        
    def split_into_sentences(self):
        self.text = re.split('\w*\. | \w*\?', self.text)
        
    def remove_empty_chars(self):
        for i in self.text:
            if i == '':
                self.text.remove(i)
        
    def tokenise(self):
        self.convert_text_to_string()
        self.remove_stupid_fullstop()
        self.lower_case()
        self.change_urls()
        self.change_hashtags()
        self.change_mentions()
        self.change_nums()
    
    def modify_text(self, text):
        self.text = text
        self.tokenise()
        self.remove_punctuations()
        self.handle_special_cases()
        self.remove_extra_spaces()
        self.split_into_sentences()
        self.remove_empty_chars()
        return self.text

def spl(data):
  arr = re.findall("<SOS>(.*?)<EOS>", data)
  return arr

filenames = ["Jane.txt", "Ulysses.txt"]
path = filenames[0]
path = './datasets/a.txt'
f = open(path, 'r')
text = f.read()
# with open(path, 'r') as fp:
#   text = fp.readlines()

# tokeniser = Tokeniser()
# text = tokeniser.modify_text(text)
text = spl(text)
maxlen = 0
for seq in text:
      maxlen = max(maxlen, len(seq))
# maxlen = max([len(seq) for seq in text])
# maxlen

train, test, dev = list(), list(), list()
division = [0.7, 0.15, 0.15]
len_test = int(round(division[1]*len(text)))
np.random.seed(41)
idx = np.random.choice(len(text), len_test, replace=True)
sos = "<SOS> "
eos = " <EOS>"
for id in range(len(text)):
    if id in idx:
        test.append(text[id].strip())
    else:
      x = sos + text[id].strip() + eos
      train.append(x)

train = ' '.join(train)
arr = spl(train)
# print(len(arr))
train = list()
idx = np.random.choice(len(arr), len_test, replace=True)
for id in range(len(arr)):
    if id in idx:
        dev.append(arr[id].strip())
    else:
      x = sos + arr[id].strip() + eos
      train.append(x)

sent_arr = ' '.join(train)
# print(sent_arr)
train2 = spl(sent_arr)

def remove_null(l):
  l = list(filter(None, l))
  return l

vocabulary = defaultdict(int)
for ind, sentence in enumerate(train2):
  sentence = sentence.split(' ')
  sentence = remove_null(sentence)
  train2[ind] = sentence

for sent in train2:
  for word in sent:
    vocabulary[word] += 1
    
vocabulary = dict(sorted(vocabulary.items(), key = lambda item: item[1], reverse = True))
alt = defaultdict(int)
alt['<UNK>'] = 0
c = 0
for k, v in vocabulary.items():
      if c >= int(len(vocabulary) / 4):
            alt['<UNK>'] += v
      else:
            alt[k] = v
vocabulary = alt.copy()

# count = 0
# vocabulary['<UNK>'] = 0
# key_rem = list()
# for k,v in vocabulary.items():
#   if v <= 1:
#     vocabulary["<UNK>"] += v
#     key_rem.append(k)

# for k in key_rem:
#   vocabulary.pop(k)

# vocabulary['<UNK>']

# print(len(vocabulary))
# print(vocabulary)
# print(vocabulary['<UNK>'])
# print(list(vocabulary.keys()))

d_final = defaultdict(int)
# l = list(vocabulary.keys())
l = list(vocabulary)

for i in range(0, len(l), 1):
  d_final[l[i]] = i+1
# print(d_final)

tokenised_sequence = list()
arr = list()
for sentence in train2:
  arr = []
  for word in sentence:
    if d_final[word] == 0:
      arr.append(d_final['<UNK>'])
    else:
      arr.append(d_final[word])
  tokenised_sequence.append(arr)

# print(tokenised_sequence)

def pad_sentences(seq, max_len):
  x, y = list(), list()
  for ind, word in enumerate(seq):
    i = ind + 1
    l = max_len - i
    x_pad = [0]*(l)
    x_pad += seq[:ind]
    x.append(x_pad)
    y.append(word)

  return x, y

max_len = 0
x,y = list(), list()
# for i in tokenised_sequence:
#   max_len = max(max_len, len(i))

def ret_np(x, y):
      x = np.array(x)
      y = np.array(y) - 1
      return x,y

for seq in tokenised_sequence:
  x_w, y_w = pad_sentences(seq, maxlen)
  x += x_w
  y += y_w

x,y = ret_np(x,y)
# print(type(x), type(y))
l = len(vocabulary)
y = np.eye(l)[y] # ONE-HOT ENCODING

# model = Sequential()

# in_dim = l + 1
# out_dim = 5
# in_len = maxlen - 1

# model.add(Embedding(input_dim=in_dim, output_dim=out_dim, input_length=in_len))
# ans = 128
# model.add(LSTM(ans, recurrent_dropout=0.3))
# model.add(Dense(l, activation='softmax'))
# model.compile('rmsprop', 'categorical_crossentropy')

# ep = 4
# model.fit(x,y,epochs=ep)

# # !rm -rf "/content/model_name.h5"
# model.save('Jane.h5')

from keras.models import load_model

model = load_model(model_path)

final_dict = {}
for k,v in d_final.items():
  final_dict[k] = v

def conv_sent(sent):
  x = re.split(' ', sent)
  x = remove_null(x)
  arr = list()
  elem = -1
  try:
    elem = final_dict[i]
  except:
    elem = final_dict['<UNK>']
  arr.append(elem)
  # for i in x:
  #   try:
  #     arr.append(final_dict[i])
  #   except:
  #     arr.append(final_dict['<UNK>'])
  return arr

def get_prob(sent):
  tok = conv_sent(sent)
  x_t, y_t = pad_sentences(tok, maxlen)
  x_t,y_t = ret_np(x_t, y_t)
  p_pred = model.predict(x_t)
  log_p_sentence = 0
  for ind, prob in enumerate(p_pred):
    prob_word = prob[y_t[ind]]
    log_p_sentence += np.log(prob_word)

  return (np.exp(log_p_sentence))

# def get_perplexity(t):
#   base = 1 / get_prob(t)
#   p = 1 / len(t.split())
#   a = np.power(base,p)
#   return a

# def perp(a):
#   return min(a, 20000)

########## TEST #########
# avg_perp = 0.0
# out_path = "2020115003_test_neural_model_" + filenames[0]
# f = open(out_path, 'w')
# f = open(out_path, 'a')

sos = '<SOS> '
eos = ' <EOS>'
# # ref = "<SOS> removed <EOS>"
# # ref = re.sub('/s+', ' ', ref)
# # val = get_perplexity(ref)
# # print(val)
# for sent in test:
  # ref=sos
  # ref+=sent+eos
  # # ref = sent
  # ref=re.sub("/s+"," ",ref)
  # val=get_perplexity(ref)
#   val = perp(val)
#   avg_perp+=val
#   print(val)
#   s1 = ref + ' ' + str(val) + '\n'
#   f.write(s1)

# s = "Average Value: " + str(avg_perp / len(test)) + '\n'
# f.write(s)
# f.write("Average Value: ")
# f.write(str(avg_perp/len(test))+"\n")
# print("Average value: ")
# print(avg_perp/len(test))
# print(s)

######## TRAIN ###########
# avg_perp = 0.0
# out_path = "2020115003_train_neural_model_" + filenames[0]
# f = open(out_path, 'w')
# f = open(out_path, 'a')

# for sent in test:
#   ref=sos
#   ref+=sent+eos
#   ref=re.sub("/s+"," ",ref)
#   val=get_perplexity(ref)
#   val = perp(val)
#   avg_perp+=val
#   print(val)
#   s1 = ref + ' '
#   s2 = str(val) + '\n'
#   f.write(s1);f.write(s2)

# s = "Average Value: " + str(avg_perp / len(test)) + '\n'
# f.write(s)
# f.write("Average Value: ")
# f.write(str(avg_perp/len(test))+"\n")
# print("Average value: ")
# print(avg_perp/len(test))
# print(s)

while True:
  st = input("input sentence: ")
  ref = st
  ref=sos
  ref+=st+eos
  # ref = sent
  ref=re.sub("/s+"," ",ref)
  val=get_prob(ref)
  print('prob: ', val)