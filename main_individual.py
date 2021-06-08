# Imports
import torch

from data import DataSet
from analyzer import IndependentAnalyzer
from loss import single_accuracy, tup_list_average
from training import train

# import the data sets
trainset = DataSet("training.csv")
devset = DataSet("development.csv")
testset = DataSet("test.csv")

# print the sizes of the data sets
print("Training Set Size: ", len(trainset.data_set))
print("Development Set Size: ", len(devset.data_set))
print("Test Set Size: ", len(testset.data_set))
print("_____________________________\n")

# Assign all sizes and values needed
ngram_size = 3
epoch_num = 10
report_rate = epoch_num+1 #no report while training
learn_rate = 0.01
mini_batch_size = 256

# try each set of parameters a certain amout of time to rely less on coincedence
try_num = 3

# try a variaty of different embedding sizes and hidden dimension sizes
emb_sizes = [100]
hid_dims = [100]

# store the number of runs
run = 1

# run the analyer for all combination of embedding and hidden dimension sizes
for emb_size in emb_sizes:
    for hid_dim in hid_dims:
        # print the parameters
        print("Embedding Size: ", emb_size)
        print("Hidden Dimensions: ", hid_dim, "\n")
        # create stores for loss and accuracy values to calculate an average
        training_accuracies, development_accuracies = [], []
        test_accuracies = []
        # initiate the try
        for t in range(try_num):
            # create the analyzer for the try
            analyzer = IndependentAnalyzer(trainset, emb_size, hid_dim, ngram_size)
            # train the analyzer
            train(trainset, devset, analyzer, learn_rate, report_rate, epoch_num, mini_batch_size)
            # print the results per try and fill the average stores
            with torch.no_grad():
                # append the stores
                training_accuracies.append(single_accuracy(trainset, analyzer))
                development_accuracies.append(single_accuracy(devset, analyzer))
                test_accuracies.append(single_accuracy(testset, analyzer))
        # calculate the average accuracys 
        av_acc_tr = tup_list_average(training_accuracies)
        av_acc_dev = tup_list_average(development_accuracies)
        av_acc_test = tup_list_average(test_accuracies)
        # print the average result for each run
        msg = ("average run {n}: \n"
                "Lexeme -> accuracy training: {at1}, accuracy development: {ad1}, accuracy test: {ats1}\n"
                "Person -> accuracy training: {at2}, accuracy development: {ad2}, accuracy test: {ats2}\n"
                "Number -> accuracy training: {at3}, accuracy development: {ad3}, accuracy test: {ats3}\n"
                "Tense -> accuracy training: {at4}, accuracy development: {ad4}, accuracy test: {ats4}\n"
                "Voice -> accuracy training: {at5}, accuracy development: {ad5}, accuracy test: {ats5}\n"
                "Mood -> accuracy training: {at6}, accuracy development: {ad6}, accuracy test: {ats6}\n"
                "Extension -> accuracy training: {at7}, accuracy development: {ad7}, accuracy test: {ats7}\n"
                "Glossary -> accuracy training: {at8}, accuracy development: {ad8}, accuracy test: {ats8}\n"
                "Aspect -> accuracy training: {at9}, accuracy development: {ad9}, accuracy test: {ats9}\n"
                "_________________________________________________________________________________\n")
        print(msg.format(
            n = run,
            at1 = round((av_acc_tr[0]), 3),
            at2 = round((av_acc_tr[1]), 3),
            at3 = round((av_acc_tr[2]), 3),
            at4 = round((av_acc_tr[3]), 3),
            at5 = round((av_acc_tr[4]), 3),
            at6 = round((av_acc_tr[5]), 3),
            at7 = round((av_acc_tr[6]), 3),
            at8 = round((av_acc_tr[7]), 3),
            at9 = round((av_acc_tr[8]), 3),
            ad1 = round((av_acc_dev[0]), 3),
            ad2 = round((av_acc_dev[1]), 3),
            ad3 = round((av_acc_dev[2]), 3),
            ad4 = round((av_acc_dev[3]), 3),
            ad5 = round((av_acc_dev[4]), 3),
            ad6 = round((av_acc_dev[5]), 3),
            ad7 = round((av_acc_dev[6]), 3),
            ad8 = round((av_acc_dev[7]), 3),
            ad9 = round((av_acc_dev[8]), 3),
            ats1 = round((av_acc_test[0]), 3),
            ats2 = round((av_acc_test[1]), 3),
            ats3 = round((av_acc_test[2]), 3),
            ats4 = round((av_acc_test[3]), 3),
            ats5 = round((av_acc_test[4]), 3),
            ats6 = round((av_acc_test[5]), 3),
            ats7 = round((av_acc_test[6]), 3),
            ats8 = round((av_acc_test[7]), 3),
            ats9 = round((av_acc_test[8]), 3)
        ))
        # increase the number of runs by 1 after every run
        run += 1