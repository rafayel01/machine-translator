

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # This should take 1-2 lines.                                                #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        
        self.word_emb = nn.Embedding(input_size, self.word_embedding_dim)
        positional_encodings = torch.zeros(max_length, hidden_dim)
        positions = torch.arange(0, max_length).unsqueeze(1)
        even = torch.arange(0, hidden_dim, step=2).float()
        positional_encodings[:, 0::2] = torch.sin(positions / 10000**(even / hidden_dim))
        positional_encodings[:, 1::2] = torch.cos(positions / 10000**(even / hidden_dim))
        self.positional_encodings =  positional_encodings.unsqueeze(0).to(device)

        ### For learnable positional encoding ###
        #self.positional_encodings = nn.Parameter(torch.randn(max_length, self.word_embedding_dim))
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.ffl_1 = nn.Linear(hidden_dim, dim_feedforward) 
        self.ffl_2 = nn.Linear(dim_feedforward, hidden_dim)
        self.relu = nn.ReLU()

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final_linear_layer = nn.Linear(hidden_dim, output_size)
        self.final_softmax = nn.Softmax()
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        inputs = inputs.to(self.device)
        outputs = self.embed(inputs)
        outputs = self.multi_head_attention(outputs)
        outputs = self.feedforward_layer(outputs)
        outputs = self.final_layer(outputs)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        embeddings = None
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        # print("word emb shape: ", self.word_emb.shape, "Pos enc: ", self.positional_encoding.shape)
        # print("Input before: ", type(inputs))
        # print("INPUT shape before = ", inputs.shape)
        #inputs = inputs * self.hidden_dim ** (0.5)
        # print("Input after: ", type(inputs))
        batch_size, seq_length = inputs.size()
        # print("INPUT shape after = ", inputs.shape)
        # print("self.word_emb(input): ", self.word_emb(inputs), "\nself.positional_encoding", self.positional_encodings)
        scale = torch.sqrt(torch.FloatTensor([inputs.size(0)])).to(self.device)
        # print("scale device: ", scale.get_device())
        # inputs = inputs.to(self.device)
        # print("input device: ", inputs.get_device())
        embeddings = self.word_emb(inputs) / scale
        
        # print("Embeddings: ", {embeddings, embeddings.shape})
        # print("Positional emb: ", self.positional_encodings, self.positional_encodings.shape)
        #embeddings = embeddings * self.hidden_dim ** (0.5)
        # print("after word embeddings: ", embeddings.shape)
        # print("POS emb: ", self.positional_encodings.size(), "\nPOS2 Emb: ", self.positional_encodings[:, :seq_length, :].size())
        embeddings += self.positional_encodings
        # embeddings += self.learnable_positional_embedings[:inputs.shape[1], :].unsqueeze(0)
        # print("after sumarize embeddings: ", embeddings.shape)
        # print("Shape EMB: ", {embeddings.shape})
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        query_1 = self.q1(inputs)
        key_1 = self.k1(inputs)
        value_1 = self.v1(inputs)
        query_2 = self.q2(inputs)
        key_2 = self.k2(inputs)
        value_2 = self.v2(inputs)
        head_1 = query_1.matmul(key_1.transpose(1, 2)) / self.dim_k ** 0.5
        head_1 = self.softmax(head_1)
        head_1 = head_1.matmul(value_1)
        head_2 = query_2.matmul(key_2.transpose(1, 2)) / self.dim_k ** 0.5
        head_2 = self.softmax(head_2)
        head_2 = head_2.matmul(value_2)
        concat = torch.cat((head_1, head_2), dim=-1)
        addition = self.attention_head_projection(concat) + inputs
        outputs = self.norm_mh(addition)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        ffn = self.ffl_2(self.relu(self.ffl_1(inputs)))
        addition_layer = ffn + inputs
        outputs = self.norm_mh(addition_layer)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = self.final_softmax(self.final_linear_layer(inputs))
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True