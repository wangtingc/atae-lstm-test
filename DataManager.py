import numpy as np
import theano

class Sentence(object):
    """docstring for sentence"""
    def __init__(self, content, target, rating, grained):
        self.content, self.target = content.lower(), target
        self.solution = np.zeros(grained, dtype=theano.config.floatX)
        self.senlength = len(self.content.split(' '))
        try:
            self.solution[int(rating)+1] = 1
        except:
            exit()
    def stat(self, target_dict, wordlist, grained=3):
        data, data_target, i = [], [], 0
        solution = np.zeros((self.senlength, grained), dtype=theano.config.floatX)
        for word in self.content.split(' '):
            data.append(wordlist[word])
            try:
                pol = Lexicons_dict[word]
                solution[i][pol+1] = 1
            except:
                pass
            i = i+1
        for word in self.target.split(' '):
            data_target.append(wordlist[word])
        return {'seqs': data, 'target': data_target, 'solution': np.array([self.solution]), 'target_index': self.get_target(target_dict)}
    def get_target(self, dict_target):
        return dict_target[self.target]

class DataManager(object):
    def __init__(self, dataset, grained=3):
        self.fileList = ['train', 'test', 'dev']
        self.origin = {}
        for fname in self.fileList:
            data = []
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                #for i in range(len(sentences)/3):
                for i in range(int(len(sentences) / 3)):
                    content, target, rating = sentences[i*3].strip(), sentences[i*3+1].strip(), sentences[i*3+2].strip()
                    sentence = Sentence(content, target, rating, grained)
                    data.append(sentence)
            self.origin[fname] = data
        self.gen_target()

    def gen_word(self):
        wordcount = {}
        def sta(sentence):
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1
            for word in sentence.target.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1

        for fname in self.fileList:
            for sent in self.origin[fname]:
                sta(sent)
        #words = wordcount.items()
        words = list(wordcount.items())
        words.sort(key=lambda x:x[1], reverse=True)
        self.wordlist = {item[0]:index+1 for index, item in enumerate(words)}
        return self.wordlist

    def gen_target(self, threshold=5):
        self.dict_target = {}
        for fname in self.fileList:
            for sent in self.origin[fname]:
                #if self.dict_target.has_key(sent.target):
                if sent.target in self.dict_target:
                    self.dict_target[sent.target] = self.dict_target[sent.target] + 1
                else:
                    self.dict_target[sent.target] = 1
        i = 0
        #for (key,val) in self.dict_target.items():
        for (key, val) in list(self.dict_target.items()):
            if val < threshold:
                self.dict_target[key] = 0
            else:
                self.dict_target[key] = i
                i = i + 1
        return self.dict_target
    
    def gen_data(self, grained=3):
        self.data = {}
        for fname in self.fileList:
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.dict_target, self.wordlist))
        return self.data['train'], self.data['dev'], self.data['test']

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        list_seledted = ['']
        line = ''
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                #if mdict.has_key(tmp[0]):
                if tmp[0] in mdict:
                    list_seledted.append(line.strip())
        list_seledted[0] = str(len(list_seledted)-1) + ' ' + str(len(line.strip().split())-1)
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))
