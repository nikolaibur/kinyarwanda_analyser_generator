# This file contains the functions, classes and variables to split data into three sets (training, development and test)
# The content of this file is not needed in the main process of the analyzer, but is necessary to split the set in preperation. It is more of a support tool for myself.
# TODO: filter out wordOld and Tense2 before exporting the three sets

import random
from data import DataSet
import csv


def split_data(path: str):
    # This funtions loads the data and splits it into the three sets
    # path: String - path to the main data file

    # load data into DataSet
    data = DataSet(path)
    # remove the datas header
    data.remove_item(0)
    # shuffle DataSets data
    data.shuffle_data()
    # calculate splitpoints (e.g. train:0.6, dev:0.2, test:0.2)
    trainsplit = calc_splitpoint(data.data_set,0.6)
    devsplit = calc_splitpoint(data.data_set,0.8)
    testsplit = calc_splitpoint(data.data_set,1)
    # split DataSet list into three lists with a defined size
    trainl = split_list(data.data_set, 0, trainsplit)
    devl = split_list(data.data_set, trainsplit, devsplit)
    testl = split_list(data.data_set, devsplit, testsplit)
    # return a tuple with the three lists
    return trainl, devl, testl


def export_set(xlist: list, path: str):
    # Exports a DataSet to a csv file
    # xlist: list - containing data to export


    # open the file, where the data is stored
    with open(path, "w", newline="") as file:
        # create the csv.wirter
        csv_writer = csv.writer(file, delimiter=";")
        # iterate over every token in the list
        for tok in xlist:
            # write one token per row
            csv_writer.writerow(tok)


def split_list(xlist: list, id1: int, id2: int):
    # Function that cutts out a part of a list between two ids and returns the cut out part
    # xlist: list - containing the data
    # id1: integer - splitpoint 1, id2: integer - splitpoint 2
    
    # calculate the list between the splitpoints
    slist = xlist[id1:id2]
    # return the list
    return slist


def calc_splitpoint(xlist: list, perc: float):
    # Function that calculates a splitting point from a percentage value
    # xlist: list - containing the data
    # perc: float - percentage value to calculate the splitpoint from

    # check if the percentage value is given as a float between 0 and 1
    assert 0 <= perc <= 1
    # calulate the length of the list
    listlen = len(xlist)
    # calculate the splitpoint
    splitpoint = listlen * perc
    # return the splitpoint as an integer
    return int(splitpoint)


# split data into three sets and export them
trainset, devset, testset = split_data("kinyarwandaVerbsExtensionsSylJuliaTestCombo.csv")
export_set(trainset, "training.csv")
export_set(devset, "development.csv")
export_set(testset, "test.csv")