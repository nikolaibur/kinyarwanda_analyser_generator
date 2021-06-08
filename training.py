# This file contains the training method for the analyzer network
import torch
import torch.nn as nn
import torch.nn.functional as fnc
from torch.optim import Adam
from typing import Iterator
import random
import math

from data import DataSet
from analyzer import IndependentAnalyzer
from loss import calc_loss, total_loss, accuracy


def train(trainset: DataSet, devset: DataSet, analyzer: IndependentAnalyzer, lr: float, rr: int, en: int, mbs:int):
    # This method is used for training the analyzer
    # trainset = The dataset containing the trainings data
    # devset = The dataset containing the development data
    # analyzer = The Analyzer used for classification
    # lr = the learning rate
    # rr = the report rate
    # en = the number of epochs
    # mbs = the size of the mini batches used

    # Create pytorchs optimizer Adam
    opt = Adam(analyzer.parameters(), lr)
    # calculate the number of updates per epoch
    upd_num = math.ceil(len(trainset.data_set)/mbs)
    # use gradient descent to optimize the gradients
    for e in range(en):
        for n in range(upd_num):
            # create the mini batch
            min_bat = random.sample(trainset.data_set, mbs)
            # calculate the total loss
            tot_loss = total_loss(min_bat, analyzer)
            # determine gradients
            tot_loss.backward()
            # optimize
            opt.step()
            # clear the gradients so they can be reused
            opt.zero_grad()

        # Reporting for every other epoch (depends on the reporting rate)
        if (e+1) % rr == 0:
            with torch.no_grad():
                msg = ("epoch {n}: "
                "total loss training: {lt}, accuracy training: {at}"
                "total loss development: {ld}, accuracy development: {ad}")
                print(msg.format(
                    n = e+1,
                    lt = round(total_loss(trainset, analyzer).item(), 3),
                    at = round(accuracy(trainset, analyzer), 3),
                    ld = round(total_loss(devset, analyzer).item(), 3),
                    ad = round(accuracy(devset, analyzer), 3))
                )