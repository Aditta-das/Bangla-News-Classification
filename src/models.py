import torch
import torch.nn as nn
import config


class MultiClassModel(nn.Module):
  def __init__(self, input_dim, n_layer, embedding_dim, hidden_dim, output_dim):
    super(MultiClassModel, self).__init__()
    self.input_dim = input_dim  
    self.hidden_dim = hidden_dim
    self.n_layer = n_layer
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.rnn1 = nn.LSTM(embedding_dim, hidden_dim, n_layer, batch_first=True)
    self.drop1 = nn.Dropout(0.5)
    self.rulu = nn.ReLU()
    self.rnn2 = nn.LSTM(hidden_dim, hidden_dim//2, n_layer, batch_first=True)
    self.drop2 = nn.Dropout(0.3)
    self.fc1 = nn.Linear(hidden_dim//2, output_dim)
    
  def forward(self, text):
    embedded = self.embedding(text)
    h0 = torch.zeros(self.n_layer, embedded.size(0), self.hidden_dim).requires_grad_().to(config.device)
    # Initialize cell state
    c0 = torch.zeros(self.n_layer, embedded.size(0), self.hidden_dim).requires_grad_().to(config.device)
    output, (hidden, cell) = self.rnn1(embedded, (h0.detach(), c0.detach()))

    output, (hidden, cell) = self.rnn2(output)
    hidden.squeeze_(0)
    output = self.fc1(hidden)
    return output

def collate_fn(data):
    text_list = []
    label_list = []
    for d in data:
      spec = d["inputs"].to(config.device)
      label = d["target"].to(config.device)
      processed_text = torch.tensor(spec.clone().detach(), dtype=torch.long)
      processed_text = processed_text.clone().detach()
      text_list.append(processed_text)
      label_list.append(label)
    spec = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0.)
    spec = spec.clone().detach()
    text = torch.tensor(spec)
    text = text.clone().detach()
    labels = torch.tensor(label_list)
    labels = labels.clone().detach()
    return text, labels