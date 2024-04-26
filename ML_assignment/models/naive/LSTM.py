
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   You also need to include correct activation functions                      #
        ################################################################################

        self.concat_size = input_size + hidden_size

        # i_t: input gate

        self.W_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_xi = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t: the forget gate

        self.W_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_xf = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t: the cell gate

        self.W_xg = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_xg = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t: the output gate

        self.W_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_xo = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################

        if init_states is None:
            h_t, c_t = torch.zeros(x.shape[0], self.hidden_size), torch.zeros(x.shape[0], self.hidden_size)
        else:
            h_t, c_t = init_states

        seq_size = x.shape[1]

        for i in range(seq_size):
            x_t = x[:, i, :]
            i_t = self.sigmoid(torch.matmul(x_t, self.W_xi) + self.b_xi + torch.matmul(h_t, self.W_hi) + self.b_hi)
            f_t = self.sigmoid(torch.matmul(x_t, self.W_xf) + self.b_xf + torch.matmul(h_t, self.W_hf) + self.b_hf)
            g_t = self.tanh(torch.matmul(x_t, self.W_xg) + self.b_xg + torch.matmul(h_t, self.W_hg) + self.b_hg)
            o_t = self.sigmoid(torch.matmul(x_t, self.W_xo) + self.b_xo + torch.matmul(h_t, self.W_ho) + self.b_ho)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)