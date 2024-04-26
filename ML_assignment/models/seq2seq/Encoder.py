

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        print(f"{self.input_size = }")
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        self.emb_layer = nn.Embedding(input_size, emb_size)
        if model_type == "RNN":
            self.recurrent_layer = nn.RNN(input_size=self.emb_size, hidden_size=self.encoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.recurrent_layer = nn.LSTM(self.emb_size, self.encoder_hidden_size, batch_first=True)

        self.linear_layers = nn.Sequential(nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size))

        self.dropout = nn.Dropout(dropout)

        self.tanh = nn.Tanh()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the weights coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################
        # h0 = torch.zeros(1, input.shape[0], self.encoder_hidden_size)
        # print(f"{input.shape = }, {self.input_size = }, {self.emb_size = }")
        emb = self.emb_layer(input)
        output = self.dropout(emb)
        if self.model_type == "RNN":
            output, hidden = self.recurrent_layer(output)
            hidden = self.tanh(self.linear_layers(hidden))
        elif self.model_type == "LSTM":
            output, (hidden_st, cell_st) = self.recurrent_layer(output)
            hidden_st = self.tanh(self.linear_layers(hidden_st))
            cell_st = self.tanh(cell_st)
            hidden = (hidden_st, cell_st)
        else:
            raise ValueError("model_type must be RNN or LSTM")
        # print(f"Encoder: {hidden = }, {hidden.shape = }")
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # Do not apply any linear layers/Relu for the cell state when model_type is #
        # LSTM before returning it.                                                 #
        # print(f"Last) {output = },\n {hidden = }")
        return output, hidden

