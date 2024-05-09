import numpy as np
import torch
import torch.nn as nn

from polyglot.mapping import Embedding
from torchtext.vocab import Vectors, GloVe
from polyglot.downloader import downloader

class Word_embedding:
    def __init__(self, emb_dim=300, lang='en', embedding='polyglot'):
        '''
        Initializing word embedding
        Parameter:
        emb_dim = (int) embedding dimension for word embedding
        '''
        if embedding == 'glove':
            # *GloVE
            glove = GloVe('6B', dim=emb_dim)
            self.embedding_vectors = glove.vectors
            self.stoi = glove.stoi
            self.itos = glove.itos
        elif embedding == 'word2vec':
            # *word2vec
            word2vec = Vectors('GoogleNews-vectors-negative300.bin.gz.txt')
            self.embedding_vectors = word2vec.vectors
            self.stoi = word2vec.stoi
            self.itos = word2vec.itos
        elif embedding == 'polyglot':
            # *Polyglot
            print(lang)
            downloader.download("embeddings2.en")
            polyglot_emb = Embedding.load('embeddings2/%s/embeddings_pkl.tar.bz2' % lang)
            self.embedding_vectors = torch.from_numpy(polyglot_emb.vectors)
            self.stoi = polyglot_emb.vocabulary.word_id
            self.itos = [polyglot_emb.vocabulary.id_word[i] for i in range(len(polyglot_emb.vocabulary.id_word))]
        elif embedding == 'dict2vec':
            word2vec = Vectors('dict2vec-vectors-dim100.vec')
            self.embedding_vectors = word2vec.vectors
            self.stoi = word2vec.stoi
            self.itos = word2vec.itos
        self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors, freeze=True, sparse=True)
        self.emb_dim = self.embedding_vectors.size(1)
        
    def __getitem__(self, index):
        return (torch.tensor([index], dtype=torch.long), self.word_embedding(torch.tensor([index])).squeeze())

    def __len__(self):
        return len(self.itos)
    
    def update_weight(self, weight):
        new_emb = Vectors(weight)
        self.embedding_vectors = new_emb.vectors
        self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors, freeze=True, sparse=True)
        self.emb_dim = self.embedding_vectors.size(1)
        self.stoi = new_emb.stoi
        self.itos = new_emb.itos

    def word2idx(self, c):
        return self.stoi[c]

    def idx2word(self, idx):
        return self.itos[int(idx)]

    def idxs2sentence(self, idxs):
        return ' '.join([self.itos[int(i)] for i in idxs])

    def sentence2idxs(self, sentence):
        word = sentence.split()
        return [self.stoi[w] for w in word]

    def idxs2words(self, idxs):
        '''
        Return tensor of indexes as a sentence
        
        Input:
        idxs = (torch.LongTensor) 1D tensor contains indexes
        '''
        idxs = idxs.squeeze()
        sentence = [self.itos[int(idx)] for idx in idxs]
        return sentence

    def get_word_vectors(self):
        return self.word_embedding

class Word_embedding_test:
    def __init__(self, emb_dim=300):
        '''
        Initializing word embedding
        Parameter:
        emb_dim = (int) embedding dimension for word embedding
        '''
        self.embedding = './.vector_cache/GoogleNews-vectors-negative300.bin.gz-'
        self.stoi = "./.vector_cache/stoi.txt"
        self.emb_dim = 0

        with open('%s%d.txt' % (self.embedding, 0), encoding='utf-8') as fp:
            for line in fp:
                entry = line.split(' ')
                self.emb_dim = len(entry) - 1
        
    def __getitem__(self, index):
        file_id = index // 100000
        file_line = index % 100000
        vector = None
        with open('%s%d.txt' % (self.embedding, file_id), encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                if i == file_line:
                    entry = line.split(' ')
                    vector = np.array(entry[1:], dtype=np.float32)
                elif i > file_line:
                    break
        return (torch.tensor([file_id, file_line], dtype=torch.long), torch.tensor(vector))

    def __len__(self):
        with open(self.stoi, encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def idx2word(self, file_idx, file_line):
        with open('%s%d.txt' % (self.embedding, file_idx), encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                if i == file_line:
                    entry = line.split(' ')
                    return(entry[0])
                elif i > file_line:
                    break
    
    # def idxs2sentence(self, idxs):
    #     return ' '.join([self.itos[int(i)] for i in idxs])

    # def sentence2idxs(self, sentence):
    #     word = sentence.split()
    #     return [self.stoi[w] for w in word]

    def idxs2words(self, idxs):
        '''
        Return tensor of indexes as a sentence
        
        Input:
        idxs = (torch.LongTensor) 1D tensor contains indexes
        '''
        return [self.idx2word(idx, line) for idx, line in idxs]

    # def get_word_vectors(self):
    #     return self.word_embedding