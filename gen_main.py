import torch

from data import DataSet
from encoder import Encoding
from generator import Generator, get_vocabs, create_enc_pairs, create_pairs
from gen_training import gen_acc, gen_loss, gen_train, gen_tot_loss
from loss import list_average


# Dateset
train_set = DataSet("training.csv")
dev_set = DataSet("development.csv")
test_set = DataSet("test.csv")

# print the sizes of the data sets
print("Training Set Size: ", len(train_set.data_set))
print("Development Set Size: ", len(dev_set.data_set))
print("Test Set Size: ", len(test_set.data_set))
print("_____________________________\n")

# ´get Training Vocab
train_input_vocab, train_target_vocab = get_vocabs(train_set)

# Choose an endsymbol and add it to the target vocab
endsym = "#"
train_target_vocab.append(endsym)

# create target and input encoding
input_enc, target_enc = Encoding(train_input_vocab), Encoding(train_target_vocab)

# transform data into list of (word,[inf]) pairs
train_pairs = create_pairs(train_set, endsym)
dev_pairs = create_pairs(dev_set, endsym)
test_pairs = create_pairs(test_set, endsym)

# create endcoded pairs
train_list = create_enc_pairs(train_pairs, input_enc, target_enc)
dev_list = create_enc_pairs(dev_pairs, input_enc, target_enc)
test_list = create_enc_pairs(test_pairs, input_enc, target_enc)

# create a list of various embedding sizes and hidden dimensions 
emb_sizes = [30]
hid_dims = [100]

# set the training parameters
learn_rate = 0.001
epoch_num = 10
report_rate = epoch_num+1 #no report while training

# set the number of tries for each combination of embedding and hidden dimension size
tries = 3

# store number of total runs
run = 1

# try all combinations of embedding and hidden dimension sizes
for emb_size in emb_sizes:
    for hid_dim in hid_dims:

        # print the current embedding size and hidden dimensions
        print("Embedding Size: ", emb_size, ", Hidden Dimensions: ", hid_dim)

        # create lists storing the results, to calculate an average later
        training_loss, development_loss = [], []
        training_acc, development_acc = [], []
        test_loss, test_acc = [], []

        # run each try
        for t in range(tries):
            # initialize Generator
            generator = Generator(input_enc, target_enc, emb_size, hid_dim, endsym)
            # Training
            gen_train(train_list, dev_list, generator, gen_loss, gen_acc, learn_rate, report_rate, epoch_num)
            # save to lists (with torch.no_grad)
            with torch.no_grad():
                # loss
                training_loss.append(gen_tot_loss(generator, train_list).item())
                development_loss.append(gen_tot_loss(generator, dev_list).item())
                test_loss.append(gen_tot_loss(generator, test_list).item())
                # accuracy
                training_acc.append(gen_acc(generator, train_list))
                development_acc.append(gen_acc(generator, dev_list))
                test_acc.append(gen_acc(generator, test_list))
                # print result of the current try
                msg = ("try {n}: "
                "total loss training: {lt}, accuracy training: {at}, "
                "total loss development: {ld}, accuracy development: {ad}, "
                "total loss test: {lts}, accuracy test: {ats}")
                print(msg.format(
                    n = t+1,
                    lt = round(training_loss[-1], 3),
                    at = round(training_acc[-1], 3),
                    ld = round(development_loss[-1], 3),
                    ad = round(development_acc[-1], 3),
                    lts = round(test_loss[-1], 3),
                    ats = round(test_acc[-1], 3))
                )
        # print runs average results
        msg = ("average run {n}: "
                "total loss training: {lt}, accuracy training: {at}, "
                "total loss development: {ld}, accuracy development: {ad}, "
                "total loss test: {lts}, accuracy test: {ats}\n"
                "_________________________________________________________________________________\n")
        print(msg.format(
            n = run,
            lt = round(list_average(training_loss), 3),
            at = round(list_average(training_acc), 3),
            ld = round(list_average(development_loss), 3),
            ad = round(list_average(development_acc), 3),
            lts = round(list_average(test_loss), 3),
            ats = round(list_average(test_acc), 3))
        )
        # increase number of runs
        run += 1