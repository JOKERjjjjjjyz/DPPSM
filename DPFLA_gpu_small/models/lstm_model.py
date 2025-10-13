# models/lstm_model.py
import torch
import torch.nn as nn
import json

class LSTMModel(nn.Module):
    def __init__(self, config_path):
        super(LSTMModel, self).__init__()
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

        vocab_size = self.config.get('vocab_size', 97)  # 默认97个字符
        hidden_size = self.config.get('hidden_size', 128)
        num_layers = self.config.get('layers', 2)  # 修改为两层 LSTM
        output_size = self.config.get('vocab_size', 97)  # 输出大小等于词汇表大小

        # 检查 embedding_layer
        self.embedding_layer = self.config.get('embedding_layer', True)
        if self.embedding_layer:
            embedding_size = self.config.get('embedding_size', 64)  # 默认 embedding 大小为 64
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            lstm_input_size = embedding_size
        else:
            self.embedding = None
            lstm_input_size = vocab_size

        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True)

        # 检查 dropouts
        self.dropout = nn.Dropout(self.config.get('dropout_ratio', 0.5)) if self.config.get('dropouts', False) else None

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 确保输入的 x 是字符索引序列，而不是 one-hot 编码
        if self.embedding_layer:
            x = self.embedding(x)  # 使用 embedding 层将字符索引转换为嵌入表示
        out, _ = self.lstm(x)  # LSTM 输入为 (batch_size, seq_len)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        if self.dropout:
            out = self.dropout(out)
        out = self.fc(out)
        return out