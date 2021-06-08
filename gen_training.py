# Training for the generator
# TODO: Merge trainings into one file?

import torch
import torch.nn as nn
from torch.optim import Adam

from generator import Generator


def gen_loss(gen: Generator, gen_res, target_res):
    # calculates the loss for a pair of the encoded result of the generator and the encoding of the target result
    # gen: Generator - the generator to generate the words
    # gen_res: the result of the generator
    # target_res: the targeted result
    
    # create loss function using CrossEntropyLoss
    loss = nn.CrossEntropyLoss()
    # calculate the scores
    scores = gen(gen_res, target_res)
    # return the loss
    return loss(scores, target_res)


def gen_tot_loss(gen: Generator, data:list):
    # calculates the total loss from a generator applied to da list of data
    # gen: Generator
    # data: list - list containing the data pairs

    # initialize the total loss
    tot_loss = torch.Tensor([0.0])
    # iterate over all of the pairs in the training data
    for x, y in data:
        # calculate the loss for the current pair
        pair_loss = gen_loss(gen, y, x)
        # add pair loss to the total loss
        tot_loss += pair_loss
    # return the total loss
    return tot_loss
        

def gen_acc(gen: Generator, data: list):
    # calculates the accuracy of the generator
    # gen: Generator
    # data: list - list containing the data pairs

    # initialize the integers for correct and total word generations
    cor, tot = 0, 0
    # iterate over every pair
    for x, y in data:
        # generate the most likely word
        gen_x = torch.argmax(gen(y, x), dim=1)
        # check if generated and targeted result are equal
        if torch.eq(gen_x, x).all():
            # if so increase correct generations by one
            cor += 1
        # increate total generations by one
        tot += 1
    # return the proportion of correct generations
    return float(cor) / float(tot)


def gen_train(train_data: list, dev_data: list, gen: Generator, loss, acc, lr: float, rr: int, en: int):
    # training method for the generator
    # train_data: list - data for the generator to train on, dev_data: list - development data, gen: Generator
    # loss: loss function, acc: accuracy funtion
    # lr: float - learning rate, rr: integer - report rate, en: integer - number of epochs

    # initialize the Adam optimizer
    opt = Adam(gen.parameters(), lr)
    # training epochs
    for epoch in range(en):
        # enable training mode
        gen.train()
        # store the loss
        tot_loss = 0
        # iterate over all of the pairs in the training data
        for x, y in train_data:
            # calculate the loss for the current pair
            pair_loss = loss(gen, y, x)
            # add pair loss to the total loss
            tot_loss += pair_loss
            # calculate gradients with back-propagation
            pair_loss.backward()
            # update parameters via optimizer using the gradients
            opt.step()
            # clear the gradients
            opt.zero_grad()
        # ---- REPORTING -----
        if (epoch+1) % rr == 0:
            # no gradients for reporting
            with torch.no_grad():
                # calculate the accuracy on the trainings set
                train_acc = acc(gen, train_data)
                # calculate the accuracy on the development set
                dev_acc = acc(gen, dev_data)
                print("Training Accuracy: ", train_acc, ", Development Accuracy: ", dev_acc)