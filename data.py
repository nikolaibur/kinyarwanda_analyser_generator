# This file contains the functions, classes and variables required to import and store data
# TODO: Changes with the new data file?

from typing import NamedTuple, Iterator
from random import shuffle
import torch.utils.data as data
from csv import reader

# Input word
Word = str

# Lexeme
Lexeme = str

# Person
Person = str

# Number
Number = str

# Tense
Tense = str

# Voice
Voice = str

# Mood
Mood = str

# Extension
Extension = str

# Gloss
Gloss = str

# Aspect
Aspect = str


#Token --> Word with all its morphological Information in one Tuple
class Token(NamedTuple):
    word: Word
    lex: Lexeme
    pers: Person
    num: Number
    ten: Tense
    voi: Voice
    md: Mood
    ext: Extension
    gl: Gloss
    asp: Aspect


def load_tokens(path: str) -> Iterator[Token]:
    # function that loads the data (or all of the tokens into an iterator to be more precise)
    # path: String - path to the data one wants to load

    # read the file
    with open(path, "r") as token_file:
        # check if it is the raw data file and if so skip wordOld
        if path == "kinyarwandaVerbsExtensionsSylJuliaTestCombo.csv":
            last_n = 10
        else:
            last_n = 9
        # create the csv reader
        csv_reader = reader(token_file, delimiter=";")
        # read file line by line
        for tok in csv_reader:
            # try to create the token
            try:
                # store each value in lower case
                word = tok[0].lower()
                lex = tok[1].lower()
                pers = tok[2].lower()
                num = tok[3].lower()
                ten = tok[4].lower()
                voi = tok[5].lower()
                md = tok[6].lower()
                ext = tok[7].lower()
                gl = tok[8].lower()
                asp = tok[last_n].lower()
                # yield the token
                yield Token(word,lex,pers,num,ten,voi,md,ext,gl,asp)
            # skip the step if the imported line is empty
            except IndexError:
                continue


class DataSet(data.IterableDataset):
    # Data set as object to store the data

    def __init__(self, path: str):
        # initialize DataSet
        # path: String - path to the data set one wants to create

        # store all the tokens in a list
        self.data_set = list(load_tokens(path))

    def __iter__(self):
        # iteration method for the data set

        # iterate over each token in the data list
        for tok in self.data_set:
            # yield the token
            yield tok      

    def shuffle_data(self):
        # used to shuffle the data in the DataSet
        shuffle(self.data_set)

    def remove_item(self, id):
        # used to remove a item from a list by index
        del(self.data_set[id])