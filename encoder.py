# This file contains the encoder which is able to en- and decode symbols.
from typing import Iterator

class Encoding:
    # Encoding Object that stores the encoded symbols and is able to en- and decode

    def __init__(self, symset):
        # Initialize the Encoding
        # symset: The set of symbols to encode

        # store the symset
        self.symset = symset
        # store the amount of symbols to encode
        self.sym_amount = len(list(symset))
        # create encoding and decoding dictionary
        self.sym_to_id = {}
        self.id_to_sym = {}
        # fill the dictionaries
        for id, sym in enumerate(symset):
            self.sym_to_id[sym] = id
            self.id_to_sym[id] = sym

    
    def encode(self, sym: str):
        # encoding method (input a symbol and get the corresponding id)
        # sym: String - the symbol one wants to encode

        # try to return the encoding
        try:
            return self.sym_to_id[sym]
        # catch a KeyError, in case the symbol is not part of the encoding
        except KeyError:
            # add the symbol to en- and decoding dictionaries
            self.sym_to_id[sym] = len(self.sym_to_id)
            self.id_to_sym[len(self.sym_to_id)-1] = sym
            # return the encoding
            return self.sym_to_id[sym]


    def seq_encode(self, seq: list):
        # encoding method for a sequence of symbols
        # seq: List - list of symbols to encode
        
        # encode every symbol in the sequence and return the list
        return [self.sym_to_id[sym] for sym in seq]

    
    def decode(self, id: int):
        # decoding method (input an id and get the corresponding symbol)
        # id: Integer - the value to be decoded
        
        # try to return the decoding
        try:
            return self.id_to_sym[id]
        # catch a KeyError, in case the id is not part of the encoding
        except KeyError:
            # return the first decoding in the dictionary
            return self.id_to_sym[0]