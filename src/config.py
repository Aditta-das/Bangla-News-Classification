import torch

dataset_path = "H:\\Abhishek Thakur\\Technometrics\\input\\new_train.csv" # main csv file
path = "H:\\Abhishek Thakur\\Technometrics\\input" # define path where fold dataset will store
csv_name = 'train_fold' # fold dataset name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device type
MAX_LEN = 512
TRAIN_BATCH = 32
TEST_BATCH = 16
N_LAYER = 1
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
learning_rate = 0.01
PATIENCE = 2
NUM_CLASSES = 7
FOLD = 5
model_save_path = "./models./best_model_at_epoch_0_fold_0.pth.tar" # add your save model path