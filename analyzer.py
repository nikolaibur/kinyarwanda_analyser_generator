# This file contains the analyzer
import torch
import torch.nn as nn
import torch.nn.functional as fnc
from typing import Iterator

from encoder import Encoding
from wordembeddings import Embedding, ngrams
from network import Network
from data import DataSet

class IndependentAnalyzer(nn.Module):
    # The Idependent Analyzer tries to classifies each morhological information independently from each other

    def __init__(self, data: DataSet, emb_size: int, hiddim: int, ngram: int):
        # initialize the IndependentAnlayzer
        # data: DataSet Object - the data in that set is going to be classified
        # emb_size: Integer - the size of the word embeddings
        # hiddim: Integer - the number of hidden dimensions in the neural networks
        # ngram: Integer - the size of the ngrams

        # using the super method to initalize the super class (nn.Module)
        super(IndependentAnalyzer, self).__init__()
        # store the parameters
        self.data = data
        self.emb_size = emb_size
        self.hiddim = hiddim
        self.ngram = ngram
        # create the embedding
        self.word_feat = self.features(self.class_extract(self.data, 0))
        self.emb = Embedding(self.word_feat, self.emb_size)
        # get all of the different classes to classify using the reduced_classes and class_extract method
        self.lex = self.reduced_classes(self.class_extract(self.data, 1))
        self.pers = self.reduced_classes(self.class_extract(self.data, 2))
        self.num = self.reduced_classes(self.class_extract(self.data, 3))
        self.ten = self.reduced_classes(self.class_extract(self.data, 4))
        self.voi = self.reduced_classes(self.class_extract(self.data, 5))
        self.md = self.reduced_classes(self.class_extract(self.data, 6))
        self.ext = self.reduced_classes(self.class_extract(self.data, 7))
        self.gl = self.reduced_classes(self.class_extract(self.data, 8))
        self.asp = self.reduced_classes(self.class_extract(self.data, 9))

        # create an encoding for all of the reduced classes
        self.enc_lex = Encoding(self.lex)
        self.enc_pers = Encoding(self.pers)
        self.enc_num = Encoding(self.num)
        self.enc_ten = Encoding(self.ten)
        self.enc_voi = Encoding(self.voi)
        self.enc_md = Encoding(self.md)
        self.enc_ext = Encoding(self.ext)
        self.enc_gl = Encoding(self.gl)
        self.enc_asp = Encoding(self.asp)

        # create the Networks for each classification task
        self.netw_lex = Network(self.emb_size, self.hiddim, len(self.lex))
        self.netw_pers = Network(self.emb_size, self.hiddim, len(self.pers))
        self.netw_num = Network(self.emb_size, self.hiddim, len(self.num))
        self.netw_ten = Network(self.emb_size, self.hiddim, len(self.ten))
        self.netw_voi = Network(self.emb_size, self.hiddim, len(self.voi))
        self.netw_md = Network(self.emb_size, self.hiddim, len(self.md))
        self.netw_ext = Network(self.emb_size, self.hiddim, len(self.ext))
        self.netw_gl = Network(self.emb_size, self.hiddim, len(self.gl))
        self.netw_asp = Network(self.emb_size, self.hiddim, len(self.asp))


    def class_extract(self, data:DataSet, class_id: int) -> Iterator[str]:
        # this method extracts all possible values from one class to an Iterator (e.g. Voice)
        # data: DataSet object - the data to extract the class from
        # class_id: Integer - the id of the class one wants to extract (see data/Token for id)

        # iterate over each token in the dataset
        for tok in self.data.data_set:
            # yield the value for the class
            yield(tok[class_id])

    def features(self, words: Iterator[str]) -> Iterator[str]:
        # creates an iterator containing all features for a sequence of words
        # words: Iterator - sequence of words to extract the features from

        # iterate over each word in the iterator
        for word in words:
            # create an iterator for the features of the word
            feat_word = ngrams(self.ngram, word)
            # iterate over each feature in the created iterator
            for feat in feat_word:
                # yield each feature
                yield(feat)

    def reduced_classes(self, class_elm: Iterator[str]) -> list:
        # takes an iterator with all elements of a class, created in class_extract() and returns a list with all different class features only once, not multiple times
        # class_elm: Iterator containing every value for one class occuring in the dataset

        # create the list
        class_feat = []
        # iterate over each element in the Iterator
        for e in class_elm:
            # check if element is not in the list already
            if e not in class_feat:
                # if not, add it to the list
                class_feat.append(e)
        # return the final list
        return class_feat

    def forward(self, word: str):
        # The forward method calculates scores for every classification task as its own tensor and returns them as a tuple
        # word: String - word to calculate scores for each classification task for

        # create the embedding list
        embs = []
        # add all the embeddings
        embs.append(self.emb.forward(ngrams(self.ngram, word)))
        # stack the tensors together to get a single embedding tensor
        emb_sum = sum(embs)
        # calculate the scores using the each tasks neural network
        scores_lex = self.netw_lex.forward(emb_sum)
        scores_pers = self.netw_pers.forward(emb_sum)
        scores_num = self.netw_num.forward(emb_sum)
        scores_ten = self.netw_ten.forward(emb_sum)
        scores_voi = self.netw_voi.forward(emb_sum)
        scores_md = self.netw_md.forward(emb_sum)
        scores_ext = self.netw_ext.forward(emb_sum)
        scores_gl = self.netw_gl.forward(emb_sum)
        scores_asp = self.netw_asp.forward(emb_sum)
        # return all scores as a tuple
        return scores_lex, scores_pers, scores_num, scores_ten, scores_voi, scores_md, scores_ext, scores_gl, scores_asp


    def calc_results(self, probs: torch.Tensor, enc: Encoding) -> dict:
        # Clalculates the results for a task (probs mapped on each option for each task)
        # probs: Tensor - containing the probabilities for each value in the class
        # enc: Encoding object - encoding for the corresponding class

        # create the result dictionary
        results = {}
        # iterate over the probaability tensor using its length
        for id in range(len(probs)):
            # decodes the option for each id
            opt = enc.decode(id)
            # assigns the correspoding prob to the option
            results[opt] = probs[id]
        # return the result dictionary
        return results

    def calc_highest(self, results: dict) -> str:
        # calculates the option with the highest probability in a results dictionary and returns it
        # results: dictionary - containing the pobabilites for each value of a class created in calc_results

        # create string that stores the value with the highest probability
        highest = ""
        # iterate over the results dictionary
        for opt in results:
            # check if the string is empty
            if highest == "":
                # if so, assign the first value of the dictionary
                highest = opt
            # if not start to compare
            else:
                # check if the new value has a higher probability then the currently selected one
                if results[opt] > results[highest]:
                    # if so assign the new one to highest
                    highest = opt
        # return the highest value as a string
        return highest

    def pre_classifyer(self, word: str):
        # classifies the morphological information of a word and returns a tuple with all probabilities for each classification tasks as seperate dictionaries
        # word: String - word to be classified
        
        # use torch.no_grad() so the gradients are not influenced
        with torch.no_grad():
            # calculate the score tuple
            scores = self.forward(word)
            # make probabilities out of the scores by mapping them to a span from 0 to 1 using softmax for each task
            probs_lex = fnc.softmax(scores[0], dim=0)
            probs_pers = fnc.softmax(scores[1], dim=0)
            probs_num = fnc.softmax(scores[2], dim=0)
            probs_ten = fnc.softmax(scores[3], dim=0)
            probs_voi = fnc.softmax(scores[4], dim=0)
            probs_md = fnc.softmax(scores[5], dim=0)
            probs_ext = fnc.softmax(scores[6], dim=0)
            probs_gl = fnc.softmax(scores[7], dim=0)
            probs_asp = fnc.softmax(scores[8], dim=0)
            # calculate the result dictionaries for each probability dictionary
            result_lex = self.calc_results(probs_lex, self.enc_lex)
            result_pers = self.calc_results(probs_pers, self.enc_pers)
            result_num = self.calc_results(probs_num, self.enc_num)
            result_ten = self.calc_results(probs_ten, self.enc_ten)
            result_voi = self.calc_results(probs_voi, self.enc_voi)
            result_md = self.calc_results(probs_md, self.enc_md)
            result_ext = self.calc_results(probs_ext, self.enc_ext)
            result_gl = self.calc_results(probs_gl, self.enc_gl)
            result_asp = self.calc_results(probs_asp, self.enc_asp)
            # return the result dictionaries as a tuple
            return result_lex, result_pers, result_num, result_ten, result_voi, result_md, result_ext, result_gl, result_asp


    def classify(self, word:str):
        # only returns the most likely (ml) option for each classification task
        # word: String - word to be classified

        # calculate the results tuple for the word
        results = self.pre_classifyer(word)
        # calculate the value with the highest probability for each classification task
        ml_lex = self.calc_highest(results[0])
        ml_pers = self.calc_highest(results[1])
        ml_num = self.calc_highest(results[2])
        ml_ten = self.calc_highest(results[3])
        ml_voi = self.calc_highest(results[4])
        ml_md = self.calc_highest(results[5])
        ml_ext = self.calc_highest(results[6])
        ml_gl = self.calc_highest(results[7])  
        ml_asp = self.calc_highest(results[8])
        # return every most likely option as a tuple
        return ml_lex, ml_pers, ml_num, ml_ten, ml_voi, ml_md, ml_ext, ml_gl, ml_asp