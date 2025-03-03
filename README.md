# Next_Word_Prediction

Here I implemented a GRU (RNN) model in Pytorch which predicts some words after a given sentence. I achieved an accuracy of 83% on the Shakespeare Hamlet data.

The hyparameters that I used were:-

input_size = 128
hidden_size = 256
num_layers = 1
num_classes = vocab_size
learning_rate = 0.001
batch_size = 64
num_epochs = 60
sequence_length = 1

The data_preprocessing is done in the `data_preprocessing.py` file. The outputs of this file at various steps are shown in `data_preprocessing_outputs.ipynb` file.
The main model architecture has been implemented in the `model.py` file.
The model training and the predictions have been done in `main.ipynb` file.
