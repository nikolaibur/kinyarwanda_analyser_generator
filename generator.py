import torch
import torch.nn as nn
from typing import Iterator

from gen_decodeencode import rnnEncoder, rnnDecoder
from encoder import Encoding
from data import DataSet
# TODO: Test with my word embeddings. atm nn.Embedding is used 


class Generator(nn.Module):
    # The generator for generating words from the given morphological information
    
    def __init__ (self, inp_enc: Encoding, target_enc: Encoding, emb_size: int, hid_dim: int, endsym: str):
        # initialize the generator module
        # inp_enc: Encoding - for the input (morphological information), target_enc: Encoding - for the target vocab
        # emb_size: integer - size of the word embeddings, hid_dim: integer - number of hidden dimensions
        # endsym: String - the symbol marking the end of a word
        
        # using the super method to initalize the super class (nn.Module)
        super(Generator, self).__init__()
        # store encoders and sizes of the encoded vocabularies
        self.inp_enc = inp_enc
        self.target_enc = target_enc
        inp_voc_size = inp_enc.sym_amount
        target_voc_size = target_enc.sym_amount
        # create word embedding for the input vocab
        self.emb = nn.Embedding(inp_voc_size+1, emb_size, inp_voc_size)
        # create the rnnEncoder and rnnDecoder
        self.encoder = rnnEncoder(emb_size, hid_dim)
        self.decoder = rnnDecoder(hid_dim, emb_size, target_voc_size, target_enc.encode(endsym))

    
    def forward(self, inp_ten, out_ten):
        # apply encode decode to calculate scores
        l_hid_state, l_cell_state = self.encoder(self.emb(inp_ten))
        return self.decoder(l_hid_state, l_cell_state, out_ten)

    
    @torch.no_grad()
    def generate(self, morph_inf):
        # generate the tensor for the output word
        # morph_inf: the morphological information to generate the word from

        # get all the encodings
        morph_enc = enc_list(morph_inf, self.inp_enc)
        # create an embedding for all of the info
        morph_emb = self.emb(morph_enc)
        # use the rnnEncoder
        hid_state, cell_state = self.encoder(morph_emb)
        # use the rnnDecoder
        char_enc = self.decoder(hid_state, cell_state)
        # decode the character encodings using the target decoder (containing all characters)
        return [self.target_enc.decode(e) for e in char_enc]


def enc_list(morph_info:tuple, enc: Encoding):
    # Adds all the encodings of a tuple of morphological info to a list and returns it
    # morph_info: tuple - containing all of the morpholical info
    # enc: Encoding - to encode the morphological info

    # create the list to store the encodings
    enc_info = []
    # iterate over the tuple
    for e in morph_info:
        # add each encoding to the list
        enc_info.append(enc.encode(e))
    # return the list of encodings
    return enc_info


def get_vocabs(data: DataSet):
    # gets the input(morph infos) and target(chars) vocab from a data set and returns them
    # data: DataSet - the data set to extract the vocabs from

    # create lists for input and target vocab
    target_vocab = []
    input_vocab = []
    # iterate over each token in the data
    for tok in data:
        # add all characters to the target vocab
        for char in tok[0]:
            # check if char is already part of the vocab
            if char.lower() not in target_vocab:
                # add new char
                target_vocab.append(char.lower())
        # add all morphological values to the input vocab
        for e in tok[1:]:
            # check if morphological value is already part of the vocab
            if e not in input_vocab:
                # add new value
                input_vocab.append(e)
    # return input and target vocab
    return input_vocab, target_vocab


def create_pairs(data:DataSet, endsym: str):
    # create pairs of target words+endsymbol and the morphological info (input)
    return [(tok[0]+endsym, [inf for inf in tok[1:]]) for tok in data]


def create_enc_pairs(data:list, inp_enc: Encoding, target_enc: Encoding):
    # create encoded pairs from pairs
    return [(torch.tensor(target_enc.seq_encode(tp[0])), torch.tensor([inp_enc.encode(inf) for inf in tp[1]])) for tp in data]