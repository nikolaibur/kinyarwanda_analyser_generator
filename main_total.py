import torch

from data import DataSet
from analyzer import IndependentAnalyzer
from loss import total_loss, accuracy, list_average
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
epoch_num = 8
report_rate = epoch_num+1 #for no report while training 
learn_rate = 0.01
mini_batch_size = 256

# try each set of parameters a certain amout of time to rely less on coincedence
try_num = 3

# try a variaty of different embedding sizes and hidden dimension sizes
emb_sizes = [100]
hid_dims = [25, 50, 100] #!!!! auf bestes Ergebnis ändern

# store the number of runs
run = 1

# run the analyer for each combination of embedding and hidden dimension size
for emb_size in emb_sizes:
    for hid_dim in hid_dims:
        # print the parameters
        print("Embedding Size: ", emb_size)
        print("Hidden Dimensions: ", hid_dim, "\n")
        # create stores for loss and accuracy values to calculate an average
        training_losses, training_accuracies, development_losses, development_accuracies = [], [], [], []
        test_losses, test_accuracies = [], []
        # initiate the try
        for t in range(try_num):
            # create the analyzer for the try
            analyzer = IndependentAnalyzer(trainset, emb_size, hid_dim, ngram_size)
            # train the analyzer
            train(trainset, devset, analyzer, learn_rate, report_rate, epoch_num, mini_batch_size)
            # print the results per try and fill the average stores
            with torch.no_grad():
                # append the stores
                training_losses.append(total_loss(trainset, analyzer).item())
                training_accuracies.append(accuracy(trainset, analyzer))
                development_losses.append(total_loss(devset, analyzer).item())
                development_accuracies.append(accuracy(devset, analyzer))
                test_losses.append(total_loss(testset, analyzer).item())
                test_accuracies.append(accuracy(testset, analyzer))
                # print the latest result
                msg = ("try {n}: "
                "total loss training: {lt}, accuracy training: {at}, "
                "total loss development: {ld}, accuracy development: {ad}, "
                "total loss test: {lts}, accuracy test: {ats}")
                print(msg.format(
                    n = t+1,
                    lt = round(training_losses[-1], 3),
                    at = round(training_accuracies[-1], 3),
                    ld = round(development_losses[-1], 3),
                    ad = round(development_accuracies[-1], 3),
                    lts = round(test_losses[-1], 3),
                    ats = round(test_accuracies[-1], 3))
                )      
        # print the average result for each run
        msg = ("average run {n}: "
                "total loss training: {lt}, accuracy training: {at}, "
                "total loss development: {ld}, accuracy development: {ad}, "
                "total loss test: {lts}, accuracy test: {ats} \n"
                "_________________________________________________________________________________\n")
        print(msg.format(
            n = run,
            lt = round(list_average(training_losses), 3),
            at = round(list_average(training_accuracies), 3),
            ld = round(list_average(development_losses), 3),
            ad = round(list_average(development_accuracies), 3),
            lts = round(list_average(test_losses), 3),
            ats = round(list_average(test_accuracies), 3))
        )
        # increase the number of runs by 1 after every run
        run += 1