# RNN Decoder and RNN Encoder
import torch
import torch.nn as nn

class rnnEncoder(nn.Module):
    # The RNN Encoder takes the encoded morphological infos and returns a fixed sized tensor for all of them united

    def __init__(self, emb_size: int, outdim: int):
        # initialize the RNN-Encoder
        # emb_size: Integer - the size of the word embeddings
        # outdim: Integer - size of the output dimension

        # using the super method to initalize the super class (nn.Module)
        super(rnnEncoder, self).__init__()
        # initialize the hidden state and cell state for the rnn
        self.init_hid = nn.Parameter(torch.randn(outdim).unsqueeze(0))
        self.init_cell = nn.Parameter(torch.randn(outdim).unsqueeze(0))
        # create the computation cell (LSTM)
        self.comp_lstm = nn.LSTMCell(emb_size, outdim)

    
    def forward(self, stacked_info):
        # use the LSTM-Cell to calculate the encoded tensor from the stacked morphological information tensor
        # stacked_info: The input tensor, containing all of the morphological information
        
        # assign the hidden and cell state
        hid_state = self.init_hid
        cell_state = self.init_cell
        # apply the computation cell on every tensor in the stacked_info tensor and calculate the new hidden and cell states
        for tens in stacked_info:
            hid_state, cell_state = self.comp_lstm(tens.unsqueeze(0), (hid_state, cell_state))
        # return a tuple of the final hidden and cell states
        return (hid_state.squeeze(0), cell_state.squeeze(0))


class rnnDecoder(nn.Module):
    # The RNN Decoder takes it and returns the predicted word

    
    def __init__(self, state_size:int, emb_size: int, voc_size: int, endsym_id: int):
        # initialize the RNN-Decoder
        # voc_size = size of the targeted vocabulary, endsym_id = id of the symbold that marks the end of each word (has tobe part of the vocabulary)

        # using the super method to initalize the super class (nn.Module)
        super(rnnDecoder, self).__init__()
        #store vocabulary size and endsymbol id
        self.voc_size = voc_size
        self.endsym_id = endsym_id
        # create an embedding for the target vocabulary (ngrams)
        self.emb = nn.Embedding(voc_size+1, emb_size, voc_size)
        # create the computation cell (LSTM)
        self.comp_lstm = nn.LSTMCell(emb_size+state_size*2, state_size)
        # initialize the parameters for the scoring function
        self.X = nn.Parameter(torch.randn(voc_size, state_size))
        self.Y = nn.Parameter(torch.randn(voc_size, emb_size))
        self.Z = nn.Parameter(torch.randn(voc_size, state_size*2))

   
    def next(self, l_hid_state, l_cell_state, hid_state, cell_state, last_char):
        # Calculate the scores for the next character
        # l_hid_state and l_cell_state: the last hidden and cell state
        # hid_state and cell_state: the current hidden and cell state
        # last_char: the id of the last predicted character

        # embed the last predicted character
        lc_emb = self.emb(last_char)
        # computation cell requires a vector consisting of the embedding vector + the two last states
        emb_vec = torch.cat((lc_emb, l_hid_state, l_cell_state))
        # apply the computation cell
        hid_state, cell_state = self.comp_lstm(emb_vec.unsqueeze(0), (hid_state, cell_state))
        # calculate the scores with the parameters created in the init function (@ is used for matrix multiplication)
        scores = (self.X @ hid_state.squeeze(0)) + (self.Y @ lc_emb) + self.Z @ torch.cat((l_hid_state, l_cell_state))
        # return the scores and the current states
        return hid_state, cell_state, scores
    
    
    def forward(self, l_hid_state, l_cell_state, target_ids):
        # calculates scores for each target character
        # l_hid_state and l_cell_state: the last hidden and cell state
        # target_ids: contains all ids of the target vocab
        
        # initialize hidden and cell state
        hid_state = l_hid_state.unsqueeze(0)
        cell_state = l_cell_state.unsqueeze(0)
        #initialize the output list, which is stacked later
        output = []
        # precict all scores
        for target_id in [target_ids[-1]] + list(target_ids[:-1]):
            hid_state, cell_state, scores = self.next(l_hid_state, l_cell_state, hid_state, cell_state, target_id)
            output.append(scores)
        return torch.stack(output)

    
    def decode(self, l_hid_state, l_cell_state, max_it: int = 25):
        # Decode a sequence of target ids (for characters) for an input of morphological information
        # l_hid_state and l_cell_state: the last hidden and cell state
        # max_it = maximum number of iterations

        # initialize hidden and cell state
        hid_state = l_hid_state.unsqueeze(0)
        cell_state = l_cell_state.unsqueeze(0)
        # initialize output
        output = []
        # set the last predicted char to the end marker (because after that there are no more predictions needed)
        last_char = torch.tensor(self.endsym_id)
        # initialize iteration counter
        count_it = 0
        # generate characters (max number = max-id)
        while count_it < max_it:
            # calculate the next charachters scores
            hid_state, cell_state, scores = self.next(l_hid_state, l_cell_state, hid_state, cell_state, last_char)
            # predict the next character
            last_char = torch.argmax(scores)
            # check if the end marker occurs (break the loop if it does)
            if last_char == self.endsym_id:
                break
            # append the output by the predicted character
            output.append(last_char)
            # count the step
            count_it += 1
        # return the stacked output
        return torch.stack(output)