

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
        # # print(f"{self.word_emb = }, {self.word_emb.shape = }")
        # # self.pos_emb = torch.zeros(self.max_length, self.hidden_dim)
        # self.position = torch.arange(max_length).unsqueeze(1)

        # self.positional_encoding = torch.zeros(1, max_length, self.word_embedding_dim)

        # _2i = torch.arange(0, self.word_embedding_dim, step=2).float()

        # # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        # self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / self.word_embedding_dim)))

        # # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        # self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / self.word_embedding_dim)))
        
        self.word_emb = nn.Embedding(input_size, self.word_embedding_dim)
        positional_encodings = torch.zeros(max_length, hidden_dim)
        positions = torch.arange(0, max_length).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim))
        _2i = torch.arange(0, hidden_dim, step=2).float()
        print("_2i SHAPE: ", _2i.shape)
        print("Pos emb shape: ", positional_encodings.shape)
        # for i in range(hidden_dim):
        #     if i % 2 == 0:


        positional_encodings[:, 0::2] = torch.sin(positions / 10000**(_2i / hidden_dim))
        positional_encodings[:, 1::2] = torch.cos(positions / 10000**(_2i / hidden_dim))
        self.positional_encodings =  positional_encodings.unsqueeze(0)
        # self.positional_encodings = torch.arange(max_length).unsqueeze(1).repeat(1, hidden_dim).float().unsqueeze(0)

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
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        
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
        outputs = None
        
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
        scale = torch.sqrt(torch.FloatTensor([inputs.size(0)]))
        embeddings = self.word_emb(inputs) / scale
        print("Embeddings: ", {embeddings, embeddings.shape})
        print("Positional emb: ", self.positional_encodings, self.positional_encodings.shape)
        #embeddings = embeddings * self.hidden_dim ** (0.5)
        # print("after word embeddings: ", embeddings.shape)
        # print("POS emb: ", self.positional_encodings.size(), "\nPOS2 Emb: ", self.positional_encodings[:, :seq_length, :].size())
        embeddings += self.positional_encodings
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
        outputs = None
        query_1 = self.q1(inputs)
        key_1 = self.k1(inputs)
        value_1 = self.v1(inputs)
        print("query 1: ", query_1.shape)
        print("key 1: ", key_1.shape)
        print("value 1: ", value_1.shape)
        query_2 = self.q2(inputs)
        key_2 = self.k2(inputs)
        value_2 = self.v2(inputs)
        print("input shape: ", inputs.shape)
        print("key 1 shape: ", key_1.shape)
        print("query_1 shape: ", query_1.shape)
        head_1 = query_1.matmul(key_1.transpose(1, 2)) / self.dim_k ** 0.5
        head_1 = self.softmax(head_1)
        print("Q * K: ", head_1.shape)
        head_1 = head_1.matmul(value_1)
        print("after matmul V: ", head_1.shape)
        head_2 = query_2.matmul(key_2.transpose(1, 2)) / self.dim_k ** 0.5
        head_2 = self.softmax(head_2)
        head_2 = head_2.matmul(value_2)
        print("head_1: ", head_1.shape)
        print("head_2: ", head_2.shape)
        outputs = torch.cat((head_2, head_1), dim=-1)
        add = self.attention_head_projection(outputs) + inputs
        outputs = self.norm_mh(add)
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
        outputs = None
        
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
        outputs = None
                
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