import torch

class ModelDataset:
  def __init__(self, texts, target):
    """
    :param texts: this is a numpy array
    :param targets: a vector, numpy array
    """
    self.texts = texts
    self.target = target

  def __len__(self):
    # return the length of dataset
    return len(self.texts)
  
  def __getitem__(self, idx):
    # return text and targets as tensor, idx is the index number 
    texts = self.texts[idx, :]
    targets = self.target[idx]
    return {
        "inputs": torch.tensor(texts, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long)
    }