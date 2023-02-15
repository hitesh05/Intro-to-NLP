import re
from collections import defaultdict

class Tokeniser():
    def __init__(self, text):
        self.text = text
        
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
    
    def modify_text(self):
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
  
# handle the case where length of sentence is less than 3
# class KneserNey():
#     def __init__(self, n_gram, d_final, sentence):
#         self.n_init = n_gram - 1
#         self.n = n_gram - 1
#         self.final_dict = d_final
#         self.sentence = sentence.split(' ')
#         self.ans_dict = dict()
#         self.d = 0
#         self.prob = 0
        
#     def first_term(self, key):
#         if (self.n == self.n_init):
#             key2 = key + ' ' + self.word # whole string
#             numer = max((self.final_dict[self.n][key2] - self.d), 0)
#             denom = self.final_dict[self.n - 1][key]
#             res = numer / denom
#             return res
        
#         # conti count
#         key2 = key + ' ' + self.word
#         key2 = key2.split(' ')
#         x = 0
# #         print(key2, key)
#         if len(key2) == 2:
#             self.n = 2
#         elif len(key2) == 3:
#             self.n = 3
            
#         numer = len(dict(filter(lambda item: key2 == item[0].split(' ')[1:], self.final_dict[self.n].items())))
#         denom = len(dict(filter(lambda item: key == item[0].split(' ')[-1], self.final_dict[self.n].items())))
# #         print(numer, denom)
#         res = numer / denom
#         return res
    
#     def lamda_term(self):
#         if self.d == 0:
#             return 0
#         key = ' '.join(map(str, self.sentence))
#         c_w = self.final_dict[self.n - 1][key]
#         search_key = self.sentence # list
#         final_word_types = len(dict(filter(lambda item: search_key == item[0].split(' ')[:-1], self.final_dict[self.n].items())))
#         res = (self.d / c_w) * (final_word_types)
#         return res
        
    
#     def conti_term(self, key):
#         key = key.split(' ')[-1]
#         denom = sum(self.final_dict[self.n].values())
#         numer = len(dict(filter(lambda item: key == item[0].split(' ')[-1], self.final_dict[self.n].items())))
#         x = numer / denom
#         return x
    
#     def smooth(self):
#         x = self.sentence[:-1]
#         self.word = self.sentence[-1]
#         self.sentence = x
        
#         while(self.n > 0):
#             key = ' '.join(map(str, self.sentence))
#             if self.final_dict[self.n - 1][key] > 0:
#                 break
#             self.n -= 1
#             self.sentence = self.sentence[1:]
            
#         if self.n == self.n_init:
#             self.d = 0
#         else:
#             self.d = 0.75
#         # print(self.n)
        
#         if self.n > 0:
#             key = ' '.join(map(str, self.sentence))
#             f_term = self.first_term(key)
#             l_term = self.lamda_term()
#             c_term = self.conti_term(key)
# #             print(f_term, l_term, c_term)
#             self.prob = f_term + (l_term*c_term)
#         else:
#             denom = sum(self.final_dict[0].values())
#             numer = 0
#             for k,v in d_final[0].items():
#                 if v < 5:
#                     numer += v
#             self.prob = numer / denom
            
#         return self.prob

class KneserNey():
    def __init__(self, n_gram, d_final, sentence):
        self.n_init = n_gram - 1
        self.n = n_gram - 1
        self.final_dict = d_final
        self.sentence = sentence.split(' ')
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
#         if (self.n == self.n_init):
        key2 = key + ' ' + self.word # whole string
        numer = max((self.final_dict[self.n][key2] - self.d), 0)
        denom = self.final_dict[self.n - 1][key]
        if denom == 0:
            return self.handle_zero()
        res = numer / denom
        return res
        
        # conti count
#         key2 = key + ' ' + self.word
#         key2 = key2.split(' ')
#         x = 0
# #         print(key2, key)
#         if len(key2) == 2:
#             self.n = 2
#         elif len(key2) == 3:
#             self.n = 3
            
#         numer = len(dict(filter(lambda item: key2 == item[0].split(' ')[1:], self.final_dict[self.n].items())))
#         denom = len(dict(filter(lambda item: key == item[0].split(' ')[-1], self.final_dict[self.n].items())))
#         if denom == 0:
#             return self.handle_zero()
#         res = numer / denom
#         return res
    
    def lamda_term(self, key):
#         key  = self.sentence.split(' ')
#         key = ' '.join(map(str, self.sentence))
#         print(key)
        c_w = self.final_dict[self.n - 1][key]
        if c_w == 0:
            return self.handle_zero()
#         print(c_w, 'cw')
        search_key = key.split(' ') # list
        final_word_types = len(dict(filter(lambda item: search_key == item[0].split(' ')[:-1], self.final_dict[self.n].items())))
        res = (self.d / c_w) * (final_word_types)
        return res
        
    
    def conti_term(self, key):
        key = key.split(' ')[-1]
        denom = len(self.final_dict[self.n].values())
#         denom =1
        numer = len(dict(filter(lambda item: key == item[0].split(' ')[-1], self.final_dict[self.n].items())))
        if numer == 0:
            numer = self.handle_zero()
        x = numer / denom
#         print('x',x)
        return x
    
    def handle_unk(self):
        denom = sum(self.final_dict[0].values())
        numer = 0
        for k,v in d_final[0].items():
            if v < self.unk_threshold:
                numer += v
        res = numer / denom
        return res

    def smooth(self):
        x = self.sentence[:-1]
        self.word = self.sentence[-1]
        self.sentence = x
        
#         check = False
#         while(self.n > 0):
#             key = ' '.join(map(str, self.sentence))
#             if self.final_dict[self.n - 1][key] > 0:
#                 check = True
#                 break
#             self.n -= 1
#             self.sentence = self.sentence[1:]
            
        for i in range(self.n_init):
            self.n = i+1
            x = self.n_init - i - 1
            key = self.sentence[x:]
            key = ' '.join(map(str, key))
#             print(key)
            if self.final_dict[self.n - 1][key] == 0:
                self.ans.append(0)
                continue
            if i == 0:
                res = self.conti_term(key)
                self.ans.append(res)
            else:
                f_term = self.first_term(key)
                l_term = self.lamda_term(key)
#                 print(l_term)
                res = f_term + (l_term*self.ans[i-1])
                self.ans.append(res)
        self.prob = self.ans[self.n_init - 1]
#         print(self.ans)
        if self.prob == 0:
            self.prob = self.handle_unk()
        
#         if self.n > 0:
#             key = ' '.join(map(str, self.sentence))
#             f_term = self.first_term(key)
#             l_term = self.lamda_term()
#             c_term = self.conti_term(key)
# #             print(f_term, l_term, c_term)
#             self.prob = f_term + (l_term*c_term)
#         else:
#             denom = sum(self.final_dict[0].values())
#             numer = 0
#             for k,v in d_final[0].items():
#                 if v < 5:
#                     numer += v
#             self.prob = numer / denom
            
        return self.prob
    
class WittenBell():
    def __init__(self, n_gram, d_final, sentence):
        self.n_init = n_gram - 1
        self.n = n_gram - 1
        self.final_dict = d_final
        self.sentence = sentence.split(' ')
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
                numer = self.handle_zero()
            denom = sum(self.final_dict[self.n].values())
            res = numer / denom
            return res
        
        numer = self.final_dict[self.n - 1][key]
        if numer == 0:
                numer = self.handle_zero()
        key2 = key + ' ' + self.word
        x = self.final_dict[self.n][key2]
        pml = x / numer
        endings = self.final_dict[self.n - 1][key]
        if endings == 0:
            endings = self.handle_zero()
        lambda_term = numer / (max(1, numer+endings))
        
        return lambda_term*pml + (1 - lambda_term)*(self.smooth_2(self.n - 1, sent[1:]))
    
    def smooth(self):
        x = self.sentence[:-1]
        self.word = self.sentence[-1]
        self.sentence = x
        self.prob = self.smooth_2(self.n_init, self.sentence)
        return self.prob   

if __name__ == '__main__':    
    filenames = ['Pride and Prejudice - Jane Austen', 'Ulysses - James Joyce']
    path = './datasets/' + filenames[0] + '.txt'
    with open(path, 'r') as fp:
        text = fp.readlines()
        
    tokeniser = Tokeniser(text)
    text = tokeniser.modify_text()

    n_gram = 4
    n_gram_model = N_Gram(n_gram,text)
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

    # for kneser ney pass only the last 4 words of the sentence. Correct using placeholders in case length is not sufficient.
    test_str = 'or views of such'
    kneser = KneserNey(n_gram, d_final, test_str)
    prob = kneser.smooth()
    print(prob)

    witten = WittenBell(n_gram, d_final, test_str)
    prob = witten.smooth()
    print(prob)