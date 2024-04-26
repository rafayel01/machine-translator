
import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        self.emb_layer = nn.Embedding(self.output_size, self.emb_size)
        # self.lstm = nn.LSTM(self.emb_size, self.decoder_hidden_size)
        # self.linear = nn.Sequential(nn.Linear(self.decoder_hidden_size, self.output_size),
        #                             nn.LogSoftmax(dim=1))
        # self.dropout = nn.Dropout(p=dropout)
        # print(f"{encoder_hidden_size = }, {decoder_hidden_size = }, {output_size = }")
        if self.model_type == "RNN":

            self.recurrent_layer = nn.RNN(emb_size, decoder_hidden_size, batch_first=True)

        elif self.model_type == "LSTM":

            self.recurrent_layer = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)

        self.linear_layer = nn.Linear(decoder_hidden_size, output_size)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
        # print(f"Decoder 1: {input = }, {input.shape = }")
        input = input.reshape(-1, 1)
        # print(f"Decoder 1: {hidden = }, {hidden.shape = }")
        # print(f"{self.output_size = }, {self.emb_size = }")
        emb = self.dropout(self.emb_layer(input))
        # print(f"Decoder 2: {hidden = }, {hidden.shape = }")
        output, hidden = self.recurrent_layer(emb, hidden)
        output = self.logsoftmax(self.linear_layer(output))
        output = output.squeeze(1)

        # print(f"{output = }, {hidden = }")
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
