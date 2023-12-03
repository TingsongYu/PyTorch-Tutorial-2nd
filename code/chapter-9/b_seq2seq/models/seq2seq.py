# -*- coding:utf-8 -*-
"""
@file name  : seq2seq.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-30
@brief      : seq2seq 模型构建
"""
import os
import sys
import random
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(EncoderLSTM, self).__init__()

        # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
        self.hidden_size = hidden_size

        # Number of layers in the lstm
        self.num_layers = num_layers

        # Regularization parameter
        self.dropout = nn.Dropout(p)
        self.tag = True

        # Shape --------------------> (5376, 300) [input size, embedding dims]
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    # Shape of x (26, 32) [Sequence_length, batch_size]
    def forward(self, x):
        # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]
        embedding = self.dropout(self.embedding(x))

        # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]
        # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size, hidden_size]
        outputs, (hidden_state, cell_state) = self.LSTM(embedding)

        return hidden_state, cell_state


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):
        super(DecoderLSTM, self).__init__()

        # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
        self.hidden_size = hidden_size

        # Number of layers in the lstm
        self.num_layers = num_layers

        # Size of the one hot vectors that will be the output to the encoder (English Vocab Size)
        self.output_size = output_size

        # Regularization parameter
        self.dropout = nn.Dropout(p)

        # Shape --------------------> (5376, 300) [input size, embedding dims]
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

        # Shape -----------> (1024, 4556) [embedding dims, hidden size, num layers]
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        # x.shape == batch_size]
        x = x.unsqueeze(0)  # x.shape == [1, batch_size]

        # Shape -----------> (1, 32, 300) [1, batch_size, embedding dims]
        embedding = self.dropout(self.embedding(x))

        # Shape --> outputs (1, 32, 1024) [1, batch_size , hidden_size]
        # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size] (passing encoder's hs, cs - context vectors)
        outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))

        # Shape --> predictions (1, 32, 4556) [ 1, batch_size , output_size]
        predictions = self.fc(outputs)

        # Shape --> predictions (32, 4556) [batch_size , output_size]
        predictions = predictions.squeeze(0)

        return predictions, hidden_state, cell_state


class Seq2Seq(nn.Module):
    def __init__(self, Encoder_LSTM, Decoder_LSTM, device):
        super(Seq2Seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM
        self.device = device

    def forward(self, source, target, tfr=0.5):
        # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
        batch_size = source.shape[1]

        # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
        target_len = target.shape[0]
        target_vocab_size = self.Decoder_LSTM.output_size

        # Shape --> outputs (14, 32, 5766)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
        hidden_state, cell_state = self.Encoder_LSTM(source)

        # Shape of x (32 elements)
        x = target[0]  # <bos> token

        for i in range(1, target_len):
            # output.shape ==  (bs, vocab_length)
            # hidden_state.shape == [num_layers, batch_size size, hidden_size]
            output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
            outputs[i] = output
            best_guess = output.argmax(1)  # 0th dimension is batch size, 1st dimension is word embedding
            x = target[
                i] if random.random() < tfr else best_guess  # Either pass the next word correctly from the dataset or use the earlier predicted word

        # Shape --> outputs (14, 32, 5766)
        return outputs


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Example usage
    SOURCE_vocab_len, TARGET_vocab_len = 8000, 10000
    input_dimensions = SOURCE_vocab_len
    output_dimensions = TARGET_vocab_len
    encoder_embedding_dimensions = 256
    decoder_embedding_dimensions = 256
    hidden_layer_dimensions = 512
    number_of_layers = 2
    encoder_dropout = 0.5
    decoder_dropout = 0.5

    input_size_encoder = SOURCE_vocab_len
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = 0.5

    encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                               hidden_size, num_layers, encoder_dropout).to(device)

    input_size_decoder = TARGET_vocab_len
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = 0.5
    output_size = TARGET_vocab_len
    decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                               hidden_size, num_layers, decoder_dropout, output_size).to(device)

    model = Seq2Seq(encoder_lstm, decoder_lstm, device).to(device)

    batch_size = 16
    seq_len = 50

    src_fake = torch.randint(0, SOURCE_vocab_len, size=(batch_size, seq_len))
    trg_fake = torch.randint(0, TARGET_vocab_len, size=(batch_size, seq_len))
    outputs = model(src_fake, trg_fake)

    print(outputs.shape)
