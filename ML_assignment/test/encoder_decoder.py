

import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
RANDOM_SEED = 0


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)


def unit_test_values(testcase):
    if testcase == 'encoder':
        expected_out = torch.FloatTensor([[[-0.7773, -0.2031]],
                                          [[-0.4129, -0.1802]],
                                          [[0.0599, -0.0151]],
                                          [[-0.9273, 0.2683]],
                                          [[0.6161, 0.5412]]])
        expected_hidden = torch.FloatTensor([[[0.4912, -0.6078],
                                              [0.4912, -0.6078],
                                              [0.4985, -0.6658],
                                              [0.4932, -0.6242],
                                              [0.4880, -0.7841]]])
        return expected_out, expected_hidden

    if testcase == 'decoder':
        expected_out = torch.FloatTensor([[-2.1507, -1.6473, -3.1772, -3.2119, -2.6847, -2.1598, -1.9192, -1.8130,
                                           -2.6142, -3.1621],
                                          [-2.0260, -2.0121, -3.2508, -3.1249, -2.4581, -1.8520, -2.0798, -1.7596,
                                           -2.6393, -3.2001],
                                          [-2.1078, -2.2130, -3.1951, -2.7392, -2.1194, -1.8174, -2.1087, -2.0006,
                                           -2.4518, -3.2652],
                                          [-2.7016, -1.1364, -3.0247, -2.9801, -2.8750, -3.0020, -1.6711, -2.4177,
                                           -2.3906, -3.2773],
                                          [-2.2018, -1.6935, -3.1234, -2.9987, -2.5178, -2.1728, -1.8997, -1.9418,
                                           -2.4945, -3.1804]])
        expected_hidden = torch.FloatTensor([[[-0.1854, 0.5561],
                                              [-0.4359, 0.1476],
                                              [-0.0992, -0.3700],
                                              [0.9429, 0.8276],
                                              [0.0372, 0.3287]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor([[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
                                            -2.1898],
                                           [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
                                            -2.3045]]])
        return expected_out


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
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
        self.emb_layer = nn.Embedding(self.input_size, self.emb_size)
        if model_type == "RNN":
            self.recurrent_layer = nn.RNN(input_size=self.emb_size, hidden_size=self.encoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.recurrent_layer = nn.LSTM(self.emb_size, self.encoder_hidden_size, batch_first=True, dropout=dropout)

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
        emb = self.emb_layer(input)
        output = self.dropout(emb)
        if self.model_type == "RNN":
            output, hidden = self.recurrent_layer(output)
            hidden = self.tanh(self.linear_layers(hidden))
        elif self.model_type == "LSTM":
            output, hidden = self.recurrent_layer(output)
            hidden = self.tanh(hidden)


        # print(f"MY TEST: {self.tanh(self.linear_layers(torch.FloatTensor([[[-0.7773, -0.2031]], [[-0.4129, -0.1802]], [[0.0599, -0.0151]],[[-0.9273, 0.2683]],[[0.6161, 0.5412]]])))}")

        # #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # Do not apply any linear layers/Relu for the cell state when model_type is #
        # LSTM before returning it.                                                 #
        print(f"Last) {output = },\n {hidden = }")
        return output, hidden

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

        if self.model_type == "RNN":

            self.recurrent_layer = nn.RNN(emb_size, decoder_hidden_size, batch_first=True)

        elif self.model_type == "LSTM":

            self.recurrent_layer = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)

        self.linear_layer = nn.Linear(decoder_hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

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
        emb = self.dropout(self.emb_layer(input))
        output, hidden = self.recurrent_layer(emb, hidden)
        output = self.linear_layer(output)

        print(f"{output = }, {hidden = }")
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden


set_seed_nb()
i, n, h = 10, 4, 2

encoder = Encoder(i, n, h, h)
x_array = np.random.rand(5,1) * 10
x = torch.LongTensor(x_array)
out, hidden = encoder.forward(x)

expected_out, expected_hidden = unit_test_values('encoder')

print('Close to out: ', expected_out.allclose(out, atol=1e-4))
print('Close to hidden: ', expected_hidden.allclose(hidden, atol=1e-4))


i, n, h =  10, 2, 2
decoder = Decoder(h, n, n, i)
x_array = np.random.rand(5, 1) * 10
x = torch.LongTensor(x_array)
_, enc_hidden = unit_test_values('encoder')
out, hidden = decoder.forward(x,enc_hidden)

expected_out, expected_hidden = unit_test_values('decoder')

print('Close to out: ', expected_out.allclose(out, atol=1e-4))
print('Close to hidden: ', expected_hidden.allclose(hidden, atol=1e-4))