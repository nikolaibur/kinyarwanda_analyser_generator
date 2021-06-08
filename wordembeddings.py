import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterator

from encoder import Encoding

class Embedding(nn.Module):
    # the embedding module

    def __init__(self, symset: Iterator, emb_size: int):
        # initialize the embedding
        # symset: Iterator - symbols to embed
        # emb_size: integer - size of the word embeddings

        # using the super method to initalize the super class (nn.Module)
        super(Embedding, self).__init__()
        # store embedding size and symset
        self.symset = symset
        self.emb_size = emb_size
        # create the encoding for the symset
        self.encoding = Encoding(symset)
        # create the embedding (embedding(symset)-size, embedding dimension)
        self.emb = nn.EmbeddingBag(self.encoding.sym_amount, self.emb_size, mode="sum")

    
    def forward(self, word: Iterator):
        # embeds a word by summing the symbols embeddings
        # word: Iterator - containing all symbols of the word

        # create a list, that stores the encoded features
        enc_feats = []
        # fill the list with the encoded features
        for feat in word:
            # skip unknows features
            try:
                enc_feats.append(self.encoding.encode(feat))
            except KeyError:
                pass
        if len(enc_feats) > 0:
            # create a tensor for the features sum. Use view(1,-1) to make sure the tensor is created with only one row (like a vector), the -1 is used because we don't know the number of collumns in advance
            word_tensor = torch.LongTensor(enc_feats).view(1,-1)
            # embed the tensor, use "[0]" to deinterleave the tensor
            return self.emb(word_tensor)[0]
        else:
            return torch.zeros(self.emb_size)


def ngrams(size: int, word: str) -> Iterator[str]:
    # the ngrams method splits a word(string) into features(smaller strings) corresponding to the given feature size
    # size: integer - size of the feaures
    # word: String - word to calculate the ngrams of

    # iterate over the word
    for n in range(len(word)-size+1):
        # yield the ngram
        yield(word[n:n+size])