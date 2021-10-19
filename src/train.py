from tensorflow.keras.preprocessing import text
import torch
import torch.nn as nn
import config
import engine
import tensorflow as tf
import dataset, models
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from string import punctuation
from engine import train_fold
from models import collate_fn
from engine import plot_loss

def bangla_word_tokenizer(sent):
  sentence = re.compile(r'[।\s+\|{}]+'.format(re.escape(punctuation)))
  word_listing_ = [i for i in sentence.split(str(sent)) if i]
  return word_listing_

def split_words_reviews(data):
    text = list(data['cleanText'].values)
    clean_text = []
    for t in text:
        clean_text.append(t.translate(str.maketrans('', '', punctuation)).lower().rstrip())
    tokenized = [bangla_word_tokenizer(x) for x in clean_text]
    all_text = []
    for tokens in tokenized:
        for t in tokens:
            all_text.append(t)
    return tokenized, set(all_text)

reviews, vocabs = split_words_reviews(train_fold)

def create_dictionaries(words):
    word_to_int_dict = {w:i+1 for i, w in enumerate(words)}
    int_to_word_dict = {i:w for w, i in word_to_int_dict.items()}
    return word_to_int_dict, int_to_word_dict

word_to_int_dict, int_to_word_dict = create_dictionaries(vocabs)

def save_model(state, filename):
    torch.save(state, filename)
    print("-> Model Saved")



def run(df, fold):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    test_df = df[df.kfold == fold].reset_index(drop=True)

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.cleanText.values.tolist())


    xtrain = tokenizer.texts_to_sequences(train_df.cleanText.values)
    xtest = tokenizer.texts_to_sequences(test_df.cleanText.values)

    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain, maxlen=config.MAX_LEN
    )

    xtest = tf.keras.preprocessing.sequence.pad_sequences(
        xtest, maxlen=config.MAX_LEN
    )

    train_dataset = dataset.ModelDataset(
        texts=xtrain,
        target=train_df.category.values 
    )

    test_dataset = dataset.ModelDataset(
        texts=xtest,
        target=test_df.category.values
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH,
        collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH,
        collate_fn=collate_fn
    )

    
    model = models.MultiClassModel(
        input_dim=len(int_to_word_dict)+1,
        n_layer=config.N_LAYER,
        embedding_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=7
    )
    model = model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=config.PATIENCE)

    best_accuracy = 0
    best_epoch = 0

    for epoch in range(config.EPOCHS):
        avg_train_acc, avg_train_loss, losses = engine.train(
            train_loader, 
            model,
            optimizer,
            device=config.device
        )
        
        print(f"Avarage Train ACC: {avg_train_acc} || Avarage Train Loss: {avg_train_loss}")
        plt.title("Losses")
        plot_loss(losses)

        avg_test_acc = engine.evaluate(
            test_loader,
            model,
            device=config.device
        )

        scheduler.step(avg_test_acc)

        print(f"Avarage Test Accuracy: {avg_test_acc}")

        if avg_train_acc > best_accuracy:
            best_accuracy = avg_train_acc
            best_epoch = epoch
            filename = f"best_model_at_epoch_{best_epoch}.pth.tar"
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_model(checkpoint, filename)

        


if __name__ == "__main__":
    df = pd.read_csv("H:\\Abhishek Thakur\\Technometrics\\input\\train_fold.csv")
    for i in range(5):
        run(df, i)
