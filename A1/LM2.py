import re
from collections import defaultdict
import math
import sys
import numpy as np

make_scores = True

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
    
    def modify_text(self, text):
        self.text = text
        self.tokenise()
        self.remove_punctuations()
        self.handle_special_cases()
        self.remove_extra_spaces()
        self.split_into_sentences()
        self.remove_empty_chars()
        return self.text
    
class N_Gram():
    def __init__(self, n_gram, text):
        self.n_gram = n_gram
        self.text = text
        self.d_final = []
        for i in range(self.n_gram):
            self.d_final.append(defaultdict(int))
        
    def make_dict(self, n, sentence):
        sentence = re.split(' ', sentence)
        sentence = [i for i in sentence if i]
        l = sentence[0:n]
        x = len(sentence)
        
        d = defaultdict(int)
        s = ' '.join(map(str, l))
        d[s]+=1
    
        for i in range(1, x - n + 1):
            l.pop(0)
            l.append(sentence[i+n-1])
            s = ' '.join(map(str, l))
            d[s]+=1   
        return d
    
    def merge_dict(self, d, n):
        for k, v in d.items():
            self.d_final[n-1][k]+=v
               
    def get_dict(self, sentence):
        # appending SOS and EOS
        s1 = '<SOS> '
        eos = '<EOS>'
        
        for n in range(1, self.n_gram+1):
            sos = s1 * (n - 1) if n > 1 else s1
            sentence_mod = '{}{}{}'.format(sos, sentence, eos)
            d = self.make_dict(n, sentence_mod)
            self.merge_dict(d, n)
        
    def build_LM(self):
        for sentence in self.text:
            self.get_dict(sentence)

        return self.d_final
    
class KneserNey():
    def __init__(self, n_gram, d_final):
        self.n_init = n_gram - 1
        self.n = n_gram - 1
        self.final_dict = d_final
        self.sentence = list()
        self.ans = list()
        self.d = 0.75
        self.prob = 0
        self.unk_threshold = 10
        
    def handle_zero(self):
        if self.n == 1:
            return self.handle_unk()
        else:
            return 1e-6
        
    def first_term(self, key):
        if self.n == self.n_init:
            key2 = key + ' ' + self.word # whole string
            numer = max((self.final_dict[self.n][key2] - self.d), 0)
            denom = self.final_dict[self.n - 1][key]
            if denom == 0:
                return self.handle_zero()
            res = numer / denom
            return res

        # conti count
        key2 = key + ' ' + self.word
        key2 = key2.split(' ')
        if len(key2) == 2:
            self.n = 2
        elif len(key2) == 3:
            self.n = 3
            
        numer = len(dict(filter(lambda item: key2 == item[0].split(' ')[1:], self.final_dict[self.n].items())))
        denom = len(dict(filter(lambda item: key == item[0].split(' ')[2:], self.final_dict[self.n].items())))
        if denom == 0:
            return self.handle_zero()
        res = numer / denom
        return res
    
    def lamda_term(self, key):
        c_w = self.final_dict[self.n - 1][key]
        if c_w == 0:
            return self.handle_zero()
        search_key = key.split(' ') # list
        final_word_types = len(dict(filter(lambda item: search_key == item[0].split(' ')[:-1], self.final_dict[self.n].items())))
        res = (self.d / c_w) * (final_word_types)
        return res
        
    
    def conti_term(self, key):
        key = key.split(' ')[-1]
        denom = len(self.final_dict[self.n].values())
        numer = len(dict(filter(lambda item: key == item[0].split(' ')[-1], self.final_dict[self.n].items())))
        if numer == 0:
            numer = self.handle_zero()
        x = numer / denom
        return x
    
    def handle_unk(self):
        denom = sum(self.final_dict[0].values())
        numer = 0
        for k,v in self.final_dict[0].items():
            if v < self.unk_threshold:
                numer += v
        res = numer / denom
        return res

    def smooth(self, sentence):
        self.sentence = sentence.split(' ')
        x = self.sentence[:-1]
        self.word = self.sentence[-1]
        self.sentence = x
            
        for i in range(self.n_init):
            self.n = i+1
            x = self.n_init - i - 1
            key = self.sentence[x:]
            key = ' '.join(map(str, key))
            if self.final_dict[self.n - 1][key] == 0:
                self.ans.append(0)
                continue
            if i == 0:
                res = self.conti_term(key)
                self.ans.append(res)
            else:
                f_term = self.first_term(key)
                l_term = self.lamda_term(key)
                res = f_term + (l_term*self.ans[i-1])
                self.ans.append(res)
        self.prob = self.ans[self.n_init - 1]
        if self.prob == 0:
            self.prob = self.handle_unk()
            
        return self.prob

 
class WittenBell():
    def __init__(self, n_gram, d_final):
        self.n_init = n_gram - 1
        self.n = n_gram - 1
        self.final_dict = d_final
#         self.sentence = sentence.split(' ')
        self.prob = 0
        self.unk_threshold = 10
        
    def handle_zero(self):
        if self.n == 1:
            return self.handle_unk()
        else:
            return 1e-6
        
    def handle_unk(self):
        denom = sum(self.final_dict[0].values())
        numer = 0
        for k,v in d_final[0].items():
            if v < self.unk_threshold:
                numer += v
        res = numer / denom
        return res
    
    def smooth_2(self, n_val, sent):
        self.n = n_val
        key = ' '.join(map(str, sent))
        if self.n == 0:
            key = self.word
            numer = self.final_dict[self.n][key]
            if numer == 0:
                res = 1 / len(self.final_dict[self.n])
                return res
            denom = sum(self.final_dict[self.n].values())
            res = numer / denom
            return res
        
        search_key = key.split(' ') # list
        d = dict(filter(lambda item: search_key == item[0].split(' ')[:-1], self.final_dict[self.n].items()))
        numer = len(d)
        denom = sum(d.values())
        denom2 = denom + numer
        if denom2 == 0:
            res = 1 / len(self.final_dict[self.n])
            return res
        lambda_term = numer / denom2
        key2 = key + ' ' + self.word
        x = self.final_dict[self.n][key2]
        pml = x / denom
        
        return (1 - lambda_term)*pml + (lambda_term)*(self.smooth_2(self.n - 1, sent[1:]))
    
    def smooth(self, sentence):
        self.sentence = sentence.split(' ')
        x = self.sentence[:-1]
        self.word = self.sentence[-1]
        self.sentence = x
        self.prob = self.smooth_2(self.n_init, self.sentence)
        return self.prob
    
class Evaluation():
    def __init__(self, n_gram, d_final):
        self.n = n_gram
        self.prob = 1.0
        self.final_dict = d_final
        self.witten_bell = WittenBell(self.n, self.final_dict)
        self.kneser_ney = KneserNey(self.n, self.final_dict)
        self.perplexities = list()
        self.avg_perplexity = 0
        
    def write_to_file(self):
        with open(self.output_path, 'w', encoding="utf-8") as f:
            f.write("Average Perplexity: " + str(self.avg_perplexity) + "\n")
            f.write(self.perplexities)
        
    def get_perpelexity(self):
        base = 1/self.prob
        exp = 1/len(self.sentence)
        return math.pow(base, exp)
    
    def get_probabilities(self, sentence):
        sentence=re.sub("\s+"," ",sentence)
        s1 = '<SOS> '
        eos = '<EOS>'
        sos = s1 * (self.n - 1) if self.n > 1 else s1
        sentence = '{}{}{}'.format(sos, sentence, eos)
        # print(sentence)
        sentence = sentence.split(' ')
        sentence = [i for i in sentence if i]
        self.sentence = sentence
        l = sentence[0:self.n]
        x = len(sentence)
        test_str = ' '.join(map(str, l))
        test = list()
        self.prob = 1.0
        if self.smoothing_type == 'w':
            self.prob *= self.witten_bell.smooth(test_str)
        else:
            self.prob *= self.kneser_ney.smooth(test_str)
    
        for i in range(1, x - self.n + 1):
            l.pop(0)
            l.append(sentence[i+self.n-1])
            test_str = ' '.join(map(str, l))
            if self.smoothing_type == 'w':
                self.prob *= self.witten_bell.smooth(test_str)
            else:
                self.prob *= self.kneser_ney.smooth(test_str)
        
    
    def evaluate(self, text, smoothing_type, output_path):
        self.text = text
        self.smoothing_type = smoothing_type
        self.output_path = output_path
        print(self.smoothing_type)
        for ind, sentence in enumerate(self.text):
            print(ind)
            if len(sentence) == 0:
                continue
            self.get_probabilities(sentence)
            prp = self.get_perpelexity()
            self.sentence = ' '.join(map(str, self.sentence))
            self.perplexities.append(self.sentence + '\t' + str(prp) + '\n')
            self.avg_perplexity += prp
        self.avg_perplexity = self.avg_perplexity / len(self.text)
        

def preprocessing(path, n):
    with open(path, 'r') as fp:
        text = fp.readlines()

    tokeniser = Tokeniser()
    text = tokeniser.modify_text(text)
    
    train_text, test_text = list(), list()
    np.random.seed(37)
    idx = np.random.choice(len(text), 1000, replace=False)
    for id in range(len(text)):
        if id in idx:
            test_text.append(text[id])
        else:
            train_text.append(text[id])

    n_gram = n
    n_gram_model = N_Gram(n_gram,train_text)
    d_final = n_gram_model.build_LM()

    # remove empty strings
    for d_type in range(4):
        empty_keys = [k for k,v in d_final[d_type].items() if not v]
        for k in empty_keys:
            d_final[d_type].remove(k)

    for i in d_final[0].keys():
        if i == '':
            del d_final[0][i]
            break
        
    return text, train_text, test_text, d_final


def get_scores(n_gram):
    filenames = ['Pride and Prejudice - Jane Austen', 'Ulysses - James Joyce']
    dir = "scores"
    roll_num = "2020115003"
    l_models = ["LM1", "LM2", "LM3", "LM4"]
    types = ["test-perplexity", "train-perplexity"]
    out_path = str()
    smoothing_types = ['w', 'k']
    
    ## PRIDE AND PREJUDICE:
    path = './datasets/' + filenames[0] + '.txt'
    text, train_text, test_text, d_final = preprocessing(path, n_gram)
    # print(len(train_text), len(test_text), test_text)
    eval = Evaluation(n_gram, d_final)
    
    ################## 1 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[0] + "_" + types[0] + ".txt"
    eval.evaluate(test_text, smoothing_types[1], out_path) # kneser ney
    
    ################## 2 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[0] + "_" + types[1] + ".txt"
    # eval.evaluate(train_text, smoothing_types[1], out_path) # kneser ney
    
    ################## 3 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[1] + "_" + types[0] + ".txt"
    # eval.evaluate(test_text, smoothing_types[0], out_path) # witten bell 
    
    ################## 4 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[1] + "_" + types[1] + ".txt"
    # eval.evaluate(train_text, smoothing_types[0], out_path) # witten bell 
    
    ## ULYSSES:
    path = './datasets/' + filenames[1] + '.txt'
    text, train_text, test_text, d_final = preprocessing(path, n_gram)
    # print(len(train_text), len(test_text), test_text)
    eval = Evaluation(n_gram, d_final)
    
    ################## 5 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[2] + "_" + types[0] + ".txt"
    # eval.evaluate(test_text, smoothing_types[1], out_path) # kneser ney
    
    ################## 6 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[2] + "_" + types[1] + ".txt"
    # eval.evaluate(train_text, smoothing_types[1], out_path) # kneser ney
    
    ################## 7 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[3] + "_" + types[0] + ".txt"
    # eval.evaluate(test_text, smoothing_types[0], out_path) # witten bell 
    
    ################## 8 #####################
    out_path = dir + "/" + roll_num + "_" + l_models[3] + "_" + types[1] + ".txt"
    # eval.evaluate(train_text, smoothing_types[0], out_path) # witten bell
     
if __name__ == '__main__':
    n_gram = 4
    if make_scores:
        get_scores(n_gram)
    exit(0)
        
    smoothing_type = str(sys.argv[1])
    if smoothing_type != 'w' and smoothing_type != 'k':
        print("Incorrect smoothing type. Use 'w' or 'k'")
        exit(0)
    path = sys.argv[2]
        
    sentence = str(input("Input Sentence: "))
    text, train_text, test_text, d_final = preprocessing(path, n_gram)
    witten_bell = WittenBell(n_gram, d_final)
    kneser_ney = KneserNey(n_gram, d_final)
    
    tokeniser = Tokeniser()
    test_str = list()
    test_str.append(sentence)
    test_str = tokeniser.modify_text(test_str)
    test_str = ' '.join(map(str, test_str))
    sentence = test_str.split(' ')
    
    l = sentence[0:n_gram]
    x = len(sentence)
    test_str = ' '.join(map(str, l))
    prob = 1.0
    test = [prob]
    if smoothing_type == 'w':
        prob *= witten_bell.smooth(test_str)
    else:
        prob *= kneser_ney.smooth(test_str)
    test.append(prob)

    for i in range(1, x - n_gram + 1):
        l.pop(0)
        l.append(sentence[i+n_gram-1])
        test_str = ' '.join(map(str, l))
        if smoothing_type == 'w':
            prob *= witten_bell.smooth(test_str)
        else:
            prob *= kneser_ney.smooth(test_str)
        test.append(prob)
        
            
    print(prob)