import torch

dataset_path = "H:\\Abhishek Thakur\\Technometrics\\input\\new_train.csv"
path = "H:\\Abhishek Thakur\\Technometrics\\input"
csv_name = 'train_fold'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 64
TRAIN_BATCH = 32
TEST_BATCH = 16
N_LAYER = 1
EMBED_DIM = 32
HIDDEN_DIM = 256
EPOCHS = 2
learning_rate = 0.01
PATIENCE = 2
NUM_CLASSES = 7
