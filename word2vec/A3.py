## imports 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
import random
import json
import regex as re
import pprint
from collections import defaultdict
import sys
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import gensim.downloader as api
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)

# tokenise the text
class Tokeniser():
    def __init__(self):
        pass
    
    def convert_text_to_string(self):
        self.text = '\n'.join(self.text)
        self.text = re.sub(r'\n+', r'\.', self.text)
    
    def lower_case(self):
        self.text = self.text.lower()
    
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
        self.text = re.split('\w*\.\w* | \w*\?\w* | \w*\!\w*', self.text)
        # self.text = re.split('\w*\. | \w*\?', self.text)
        
    def remove_empty_chars(self):
        for i in self.text:
            if i == '':
                self.text.remove(i)
        
    def tokenise(self):
        self.convert_text_to_string()
        self.remove_stupid_fullstop()
        self.lower_case()
    
    def modify_text(self, text):
        self.text = text
        self.tokenise()
        self.remove_punctuations()
        self.handle_special_cases()
        self.remove_extra_spaces()
        self.split_into_sentences()
        self.remove_empty_chars()
        return self.text
    
# preprocessing text and creating vocab
class TextProcessing():
    def __init__(self, path, sentences, pad_val=0):
        self.path = path
        self.sentences = sentences
        self.vocab = set()
        self.corpus = set()
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.tokeniser = Tokeniser()
        self.pad_val = pad_val
        
    def load_data(self):
        self.chunks = pd.read_json(self.path, lines=True, chunksize=10000)
        
    def read_data(self):
        check = False
        for chunk in self.chunks:
            text = chunk['reviewText']
            for sentences in text:
                sentences = sentences.split('.')
                for sent in sentences:
                    self.corpus.add(sent)
                if len(self.corpus) > self.sentences:
                    check = True
                    break
            if check:
                self.corpus = list(self.corpus)
                break
            
        for i in self.corpus:
            if len(i.split()) < 2:
                self.corpus.remove(i)
        self.final_sentences = len(self.corpus)
        self.corpus = self.tokeniser.modify_text(self.corpus)
        self.corpus = [[self.start_token]+[word for word in sentence.split()]+[self.end_token] for sentence in self.corpus]
    
    def build_vocab(self):
        self.vocab = sorted(list(set([word for doc in self.corpus for word in doc])))
        self.word_to_idx = defaultdict(int)
        for doc in self.corpus:
            for word in doc:
                self.word_to_idx[word] += 1
        for k in list(self.word_to_idx.keys()):
            if ('!' in k):
                del self.word_to_idx[k]
            elif self.word_to_idx[k] == 1:
                self.word_to_idx['<UNK>'] += 1
                del self.word_to_idx[k]
        self.word_to_idx = dict(sorted(self.word_to_idx.items(), key=lambda x:x[0]))
        self.widx = self.word_to_idx.copy()
        self.word_to_idx = {word: i+1 for i,word in enumerate(list(self.word_to_idx.keys()))}
        self.word_to_idx['<pad>'] = self.pad_val
        self.idx_to_word = {i+1:word for i,word in enumerate(list(self.widx.keys()))}
        self.idx_to_word[0] = '<pad>'
        self.vocab = list(self.word_to_idx.keys())
        self.num_words = len(self.vocab)


    def indenture(self):
        self.load_data()
        self.read_data()
        self.build_vocab()
 
# performing method #1 svd
class SVD():
    def __init__(self, corpus, dict, vocab, num_words, embedding_dim=2, window_size=4):
        self.corpus = corpus
        self.word_to_idx = dict
        self.vocab = vocab
        self.num_words = num_words
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.unk_word = '<UNK>'
    
    def build_cooc_mat(self):
        self.cooc_mat = np.zeros((self.num_words, self.num_words), dtype=np.int32)
        print('building cooc matrix')
        for doc in self.corpus:
            doc_size = len(doc)
            for cur_doc_idx in range(doc_size):
                l = max(0, cur_doc_idx - self.window_size)
                r = min(doc_size - 1, cur_doc_idx + self.window_size)
                w = doc[cur_doc_idx]
                try:
                    dict_idx = self.word_to_idx[w]
                except:
                    dict_idx = self.word_to_idx[self.unk_word]
                outside_words = doc[l:cur_doc_idx]+doc[cur_doc_idx+1:r]
                for i in outside_words:
                    try:
                        i_idx = self.word_to_idx[i]
                    except:
                        i_idx = self.word_to_idx[self.unk_word]
                    self.cooc_mat[i_idx, dict_idx] += 1
    
    def cal_svd(self):
        print('calculating svd')
        svd_inst = TruncatedSVD(n_components=self.embedding_dim, n_iter=10)
        self.word_embeddings = svd_inst.fit_transform(self.cooc_mat)
        # U, s, Vt = np.linalg.svd(self.cooc_mat)
        # self.word_embeddings = U[:, :self.embedding_dim] * np.sqrt(s[:self.embedding_dim])
      
    def indenture(self):
        self.build_cooc_mat()  
        self.cal_svd()
  
class Datasets():
    def __init__(self, vocab, corpus, word_to_idx, idx_to_word, pad_val, window_size=2):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.corpus = corpus
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.window_size = window_size
        self.unk_word = '<UNK>'
        self.data = list()
        self.data_skip = list()
        self.pad_val = pad_val

    def padding(self):
        # CBOW
        self.context = [sent[0] for sent in self.data]
        self.words = np.array([sent[1] for sent in self.data])
        max_len = max(len(arr) for arr in self.context)
        for ind, arr in enumerate(self.context):
            x = max_len - len(arr)
            for _ in range(x):
                self.context[ind].append(self.pad_val)
        self.data.clear()
        for ind, i in enumerate(self.context):
            self.data.append((i, self.words[ind]))

        for ind, i in enumerate(self.data):
            self.data[ind] = (torch.tensor(self.data[ind][0]), torch.tensor(self.data[ind][1]))

        # SKIPGRAM
        for ind, i in enumerate(self.data_skip):
            self.data_skip[ind] = (torch.tensor(self.data_skip[ind][0]), torch.tensor(self.data_skip[ind][1]), torch.tensor(self.data_skip[ind][2]))   

    def create_cbow_dataset(self):
        len_corpus = len(self.corpus)
        for doc_ind, doc in enumerate(self.corpus):
            doc_size = len(doc)
            for cur_doc_idx in range(doc_size):
                l = max(0, cur_doc_idx - self.window_size)
                r = min(doc_size - 1, cur_doc_idx + self.window_size)
                w = doc[cur_doc_idx]
                try:
                    dict_idx = self.word_to_idx[w]
                except:
                    dict_idx = self.word_to_idx[self.unk_word]
                outside_words = doc[l:cur_doc_idx]+doc[cur_doc_idx+1:r+1]
                i_idx_arr = list()
                for i in outside_words:
                    try:
                        i_idx = self.word_to_idx[i]
                    except:
                        i_idx = self.word_to_idx[self.unk_word]
                    i_idx_arr.append(i_idx)
                self.data.append((i_idx_arr, dict_idx))

    def create_skipgram_dataset(self):
        len_corpus = len(self.corpus)
        for doc_ind, doc in enumerate(self.corpus):
            doc_size = len(doc)
            for cur_doc_idx in range(doc_size):
                l = max(0, cur_doc_idx - self.window_size)
                r = min(doc_size - 1, cur_doc_idx + self.window_size)
                w = doc[cur_doc_idx]
                try:
                    dict_idx = self.word_to_idx[w]
                except:
                    dict_idx = self.word_to_idx[self.unk_word]
                outside_words = doc[l:cur_doc_idx]+doc[cur_doc_idx+1:r+1]
                i_idx_arr = list()
                for i in outside_words:
                    try:
                        i_idx = self.word_to_idx[i]
                    except:
                        i_idx = self.word_to_idx[self.unk_word]
                    self.data_skip.append((dict_idx, i_idx, 1))
        
                # negative sampling
                rand_id_arr = set()
                while len(rand_id_arr) != 2:
                    ind = random.randint(1, len_corpus -1)
                    doc_ind_neg = (doc_ind + ind) % len_corpus
                    doc_neg = self.corpus[doc_ind_neg]
                    rand_id = random.randint(0, len(doc_neg) - 1)
                    word_neg = doc_neg[rand_id]
                    try:
                        ind_word_neg = self.word_to_idx[word_neg]
                    except:
                        ind_word_neg = self.word_to_idx[self.unk_word]
                    self.data_skip.append((dict_idx, ind_word_neg, 0))

    def work(self):
        self.create_cbow_dataset()
        # self.create_skipgram_dataset()
        self.padding()
       
class cbow_word2vec(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_val, window_size=2):
        super(cbow_word2vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=pad_val)
        self.linear1 = nn.Linear(2*window_size*embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.size(0),-1))
        hid = F.relu(self.linear1(embeds))
        out = self.linear2(hid)
        probs = F.log_softmax(out, dim=1)
        return probs

    def predict(self, idx):
        return self.embeddings(idx)
    
class skipgram_word2vec(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_val, window_size=2):
        super(skipgram_word2vec, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=pad_val)
        self.out_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=pad_val)
        self.pred = torch.tensor([])
        
    def forward(self, i, o, neg):
        in_embeds = self.in_embeddings(i)
        out_embeds = self.out_embeddings(o)
        
        self.pred = out_embeds
        pos_val = F.logsigmoid(torch.bmm(out_embeds, in_embeds.unsqueeze(2)).squeeze(2).sum(1))
        neg_embeds = self.out_embeddings(neg)
        neg_val = - F.logsigmoid(torch.bmm(neg_embeds, in_embeds.unsqueeze(2)).squeeze(2).sum(1))
        return - (pos_val + neg_val).mean()

    def predict(self, idx):
        return self.in_embeddings(idx)

class CBOW_model():
    def __init__(self):
        pass

    def train_cbow_word2vec(self, vocab_size, data, hidden_size=64, batch_size=32,lr=0.001, num_epochs=10, window_size=2, embedding_size=64):
        self.w2v = cbow_word2vec(vocab_size, embedding_size, hidden_size, PAD_VALUE).to(device)
        dataload = DataLoader(data, batch_size=batch_size, num_workers=0)
        loss_func = nn.NLLLoss().to(device)
        optim = torch.optim.Adam(params=self.w2v.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in tqdm(range(num_epochs), desc='epoch'):
            self.w2v.train()
            total_loss = 0
            total_acc = 0
            for ind, i in enumerate(dataload):
                context, target = map(lambda x:x.to(device), i)
                # context = i[0]
                # target = i[1]
                # context = Variable(torch.LongTensor(context)).to(device)
                # target = Variable(torch.LongTensor([target])).to(device)

                optim.zero_grad()
                pred = self.w2v(context)
                loss = loss_func(pred, target)

                loss.backward()
                optim.step()

                total_loss += loss.item()    
            print(f'\tEpoch {epoch + 1}\tTrain Loss: {total_loss}')
        torch.save(self.w2v.state_dict(), 'cbow_w2v.pth')

    def load_model(self):
        self.w2v.load_state_dict(
            torch.load(
                "/content/drive/MyDrive/cbow_w2v.pth", map_location=torch.device("cpu")
            )
        )
        return self.w2v

    def nearest_words(self,word, word_to_idx, idx_to_word, num=10):
        try:
            idx = word_to_idx[word]
        except:
            idx = word_to_idx['<UNK>']
        emb = self.w2v.predict(torch.tensor(idx).to(device)).unsqueeze(0)
        sims = torch.mm(emb, self.w2v.embeddings.weight.transpose(0,1)).squeeze(0)
        sims = (-sims).sort()[1][0: num+1]
        tops = list()
        for k in range(num):
            tops.append(idx_to_word[sims[k].item()])
        return tops  

class Skipgram_model():
    def __init__(self):
        pass  
    
    def train_skipgram_word2vec(self, vocab_size, data, pad_val=0, hidden_size=64, batch_size=32,lr=0.001, num_epochs=10, window_size=2, embedding_size=64):
        self.w2v = skipgram_word2vec(vocab_size, embedding_size, hidden_size, pad_val).to(device)
        dataload = DataLoader(data, batch_size=batch_size, num_workers=0)
        loss_func = nn.NLLLoss().to(device)
        optim = torch.optim.Adam(params=self.w2v.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in tqdm(range(num_epochs), desc='epoch'):
            self.w2v.train()
            total_loss = 0
            total_acc = 0
            for ind, i in enumerate(dataload):
                inp, out, neg = map(lambda x:x.to(device), i)

                optim.zero_grad()
                loss = self.w2v(inp, out, neg)
                loss.backward()
                optim.step()

                total_loss += loss.item()    
            print(f'\tEpoch {epoch + 1}\tTrain Loss: {total_loss}')
        torch.save(self.w2v.state_dict(), 'skipgram_w2v.pth')
        
    def load_model(self):
        self.w2v.load_state_dict(
            torch.load(
                "/content/drive/MyDrive/skipgram_w2v.pth", map_location=torch.device("cpu")
            )
        )
        return self.w2v

    def nearest_words(self,word, word_to_idx, idx_to_word, num=10):
        try:
            idx = word_to_idx[word]
        except:
            idx = word_to_idx['<UNK>']
        emb = self.w2v.predict(torch.tensor(idx).to(device)).unsqueeze(0)
        sims = torch.mm(emb, self.w2v.in_embeddings.weight.transpose(0,1)).squeeze(0)
        sims = (-sims).sort()[1][0: num+1]
        tops = list()
        for k in range(num):
            tops.append(idx_to_word[sims[k].item()])
        return tops
    
# visualising word embeddings
def plot(word_embeddings, word_to_idx, words):
    for word in words:
        try:
            idx = word_to_idx[word]
        except:
            idx = word_to_idx['<UNK>']
        x, y = word_embeddings[idx]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x,y,word,fontsize=9)
    plt.show()

# plotting embeddings according to tsne (method #3)
# ERROR
def plot_tsne(model, word_to_idx, words):
    word_vectors = []
    for word in words:
        try:
            idx = word_to_idx[word]
        except:
            idx = word_to_idx['<UNK>']
        emb = model.predict(torch.tensor(idx).to(device)).unsqueeze(0)
        word_vectors.append(emb.cpu().detach().numpy())
    word_vectors = np.array(word_vectors)
    print(word_vectors)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    word_vectors_tsne = tsne.fit_transform(word_vectors)
    for ind, word in enumerate(words):
        x, y = word_vectors_tsne[ind, :]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x,y,fontsize=9)
    plt.show()

# finding the n closest words according to the pretrained word2vec model and my program
def find_closest_words(word_embeddings, word_to_idx, idx_to_word, word, top_n=10):
    ### pretrained model ###
    pretrained_model = api.load('word2vec-google-news-300')
    pretrained_closest_words = pretrained_model.most_similar(word, topn=top_n)
    print(f"Closest words to {word} according to word2vec:")
    for i in pretrained_closest_words:
        print(i, end=' ')
    
    ### my program ###  
    try:
        idx = word_to_idx[word]
    except:
        idx = word_to_idx['<UNK>']
    word_vector = word_embeddings[idx]
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(word_embeddings)
    dist, ind = knn.kneighbors(word_vector.reshape(1,-1), n_neighbors=11)
    closest_words = []
    for i in ind.flatten()[1:]:
        closest_words.append(idx_to_word[i])
    print(f"Closest words to {word} according to my program:")
    for i in closest_words:
        print(i, end = ' ')

if __name__ == '__main__':   
    path = '/media/hitesh/DATA/IIIT-H/3rd_year/INLP/A3/reviews_Movies_and_TV.json'       
    sentences = 45000
    textProcesser = TextProcessing(path, sentences)
    textProcesser.indenture()
    
    ### PART 1: SVD
    # svd = SVD(textProcesser.corpus, textProcesser.word_to_idx, textProcesser.vocab, textProcesser.num_words)
    # svd.indenture()
    # scaler = StandardScaler()
    
    # word_embeddings = svd.word_embeddings
    word_to_idx = textProcesser.word_to_idx
    idx_to_word = textProcesser.idx_to_word
    # word_embeddings = scaler.fit_transform(word_embeddings)
    test_words = ["funny", "hilarious", "good", "bad", "king", "beggar"]
    # plot(word_embeddings, word_to_idx, test_words)
    
    ### PART 2: WORD2VEC
    datasets = Datasets(textProcesser.vocab, textProcesser.corpus, word_to_idx, idx_to_word,PAD_VALUE)
    datasets.work()
    
    # CBOW
 
    # SKIPGRAM
    
        
    ### PART 3: TSNE
    # plot_tsne(word_embeddings, word_to_idx, test_words)
    
    ### PART 4: 10 CLOSEST WORDS
    # find_closest_words(word_embeddings, word_to_idx, idx_to_word, "titanic")