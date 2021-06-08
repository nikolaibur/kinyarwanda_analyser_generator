# This file contains the loss and the accuracy functions

import torch
import torch.nn as nn

from encoder import Encoding
from wordembeddings import Embedding
from data import DataSet
from analyzer import IndependentAnalyzer


def calc_loss(scores:torch.Tensor, correct_id: int) -> torch.Tensor:
    # calculates the loss between a scores tensor and the correct id(option)
    # scores: Tensor - containing the calculated scores
    # correct_id: integer - id of the target value

    # return the loss
    return nn.CrossEntropyLoss()(scores.view(1,-1), torch.LongTensor([correct_id]))


def total_loss(data: list, analyzer) -> torch.Tensor:
    # calculates the total loss of the analyzer on the dataset
    # data: list: containing the data to calculate the total loss from
    # analyzer: the analyzer to analyze the dada

    #initatate loss tensor
    loss_sum = torch.Tensor([0.0])
    # iterate over the token tuple in the data
    for (word, lex, pers, num, ten, voi, md, ext, gl, asp) in data:
        # encode the ids for the correct values
        lex_id = analyzer.enc_lex.encode(lex)
        pers_id = analyzer.enc_pers.encode(pers)
        num_id = analyzer.enc_num.encode(num)
        ten_id = analyzer.enc_ten.encode(ten)
        voi_id = analyzer.enc_voi.encode(voi)
        md_id = analyzer.enc_md.encode(md)
        ext_id = analyzer.enc_ext.encode(ext)
        gl_id = analyzer.enc_gl.encode(gl)
        asp_id = analyzer.enc_asp.encode(asp)
        # calculate scores that the analyzer predicts
        scores = analyzer.forward(word)
        # add on the total loss
        loss_sum += calc_loss(scores[0], lex_id)
        loss_sum += calc_loss(scores[1], pers_id)
        loss_sum += calc_loss(scores[2], num_id)
        loss_sum += calc_loss(scores[3], ten_id)
        loss_sum += calc_loss(scores[4], voi_id)
        loss_sum += calc_loss(scores[5], md_id)
        loss_sum += calc_loss(scores[6], ext_id)
        loss_sum += calc_loss(scores[7], gl_id)
        loss_sum += calc_loss(scores[8], asp_id)
    # return the total loss
    return loss_sum


def accuracy(data: DataSet, analyzer) -> float:
    # calculates the accuracy of the model by comparing the predicted with the goal morphological info
    # data: DataSet object - the data to calculate the accuracy of
    # analyzer: nn.Module - the analyzer to analyze the data

    # initialize integers for correct and total predictions
    cor, tot = 0, 0
    # iterate over each token in the data
    for tok in data:
        # classify the morph info unsing the analyzer
        res = analyzer.classify(tok[0])
        # check if the result equals the target
        if res == tok[1:]:
            # if so increase the value of correct predictions
            cor +=1
        # increase the total number of predictions
        tot += 1
    # return correct predictions devided by the total number of predictions
    return float(cor)/float(tot)


def single_accuracy(data: DataSet, analyzer) -> tuple:
    # calculates the accuracy for each classification task to get a more differentiated overview
    # data: DataSet object - the data to calculate the accuracy of
    # analyzer: nn.Module - the analyzer to analyze the data

    # initialize the integers for correct and total predictions
    cor_lex, cor_pers, cor_num, cor_ten, cor_voi, cor_md, cor_ext, cor_gl, cor_asp, tot = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # iterate over each token in the data
    for tok in data:
        # classify the morph info unsing the analyzer
        res = analyzer.classify(tok[0])
        # check if the predicted value equals the corresponding target value and increase it if that is the case
        if res[0] == tok[1]:
            cor_lex +=1
        if res[1] == tok[2]:
            cor_pers +=1
        if res[2] == tok[3]:
            cor_num +=1
        if res[3] == tok[4]:
            cor_ten +=1
        if res[4] == tok[5]:
            cor_voi +=1
        if res[5] == tok[6]:
            cor_md +=1
        if res[6] == tok[7]:
            cor_ext +=1
        if res[7] == tok[8]:
            cor_gl +=1
        if res[8] == tok[9]:
            cor_asp +=1
        # increase the total number of predictions
        tot += 1
    # calculate the accuracies
    acc_lex = float(cor_lex)/float(tot)
    acc_pers = float(cor_pers)/float(tot)
    acc_num = float(cor_num)/float(tot)
    acc_ten = float(cor_ten)/float(tot)
    acc_voi = float(cor_voi)/float(tot)
    acc_md = float(cor_md)/float(tot)
    acc_ext = float(cor_ext)/float(tot)
    acc_gl = float(cor_gl)/float(tot)
    acc_asp = float(cor_asp)/float(tot)
    # return the accuracies
    return acc_lex, acc_pers, acc_num, acc_ten, acc_voi, acc_md, acc_ext, acc_gl, acc_asp


def list_average(liste: list) -> float:
    # calculates the average of all values in a list
    # liste: list - containing the values to caculate the average of

    # initialize the integer for the sum of all values in the list
    total = 0
    # iterate over each value in the list
    for e in liste:
        # add to total
        total += e
    # return the average
    return float(total)/float(len(liste))


def tup_list_average(liste: list) -> tuple:
    # calculates averages for a list of tuples
    # liste: list - containing the tuples

    # initialize the integers for the totals
    tot_lex, tot_pers, tot_num, tot_ten, tot_voi, tot_md, tot_ext, tot_gl, tot_asp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # iterate over each tuple in the list
    for tp in liste:
        # add value to the corresponding total
        tot_lex += tp[0]
        tot_pers += tp[1]
        tot_num += tp[2]
        tot_ten += tp[3]
        tot_voi += tp[4]
        tot_md += tp[5]
        tot_ext += tp[6]
        tot_gl += tp[7]
        tot_asp += tp[8]
    # calculate the averages
    av_lex = tot_lex/float(len(liste))
    av_pers = tot_pers/float(len(liste))
    av_num = tot_num/float(len(liste))
    av_ten = tot_ten/float(len(liste))
    av_voi = tot_voi/float(len(liste))
    av_md = tot_md/float(len(liste))
    av_ext = tot_ext/float(len(liste))
    av_gl = tot_gl/float(len(liste))
    av_asp = tot_asp/float(len(liste))
    # return the averages
    return av_lex, av_pers, av_num, av_ten, av_voi, av_md, av_ext, av_gl, av_asp