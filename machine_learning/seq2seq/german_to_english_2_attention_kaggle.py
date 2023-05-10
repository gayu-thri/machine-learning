# -*- coding: utf-8 -*-
"""german_to_english_2_kaggle.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1peJRQwbBdlgd5KfG06qZ5wye0ZMOJPc_

In this notebook, I've implemented the most basic version of Seq2Seq model (without attention mechanism), also know as encoder-decoder model. Most of the modeling and training part are referenced from part-I of this great tutorial series: https://github.com/bentrevett/pytorch-seq2seq.

But in this series, the preprocessed data is being used for training/evaluation (because pytorch's Multi30k class provides all the heavy lifting), so it's bit difficult to generalize the structure for custom dataset implementation. So in this notebook, I've implemented data preprocessing like tokenization, padding etc. from scratch using spacy and pure pytorch.

Here are some other references I've used:

* [Original research paper.](https://arxiv.org/pdf/1409.3215.pdf)
* [Creating custom dataset for NLP tasks](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/22635a65d8cf462aa44199357928e61c0ecda000/ML/Pytorch/more_advanced/image_captioning/get_loader.py)
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !python -m spacy download en
# # !python -m spacy download de
# !python -m spacy download de_core_news_sm

import os
import re
import time
import math
import random
import unicodedata

import numpy as np
import pandas as pd

from tqdm import tqdm

import spacy

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

# TODO:
# from machine_learning.utils import DATA_ROOT_DIR
deu_text_path = "/home/local/ZOHOCORP/gayathri-12052/learnings/machine-learning/machine_learning/data/seq2seq/german_english/archive/deu.txt"

SEED = 28

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

data_df = pd.read_csv(deu_text_path, sep="\t", usecols=[0, 1])
data_df.columns = ["en", "de"]
data_df.head()

data_df.shape
# 221532 en-de pairs

plt.figure(figsize=(12, 6))
plt.style.use("ggplot")
plt.subplot(1, 2, 1)
sns.distplot(data_df["en"].str.split().apply(len))
plt.title("Distribution of English sentences length")
plt.xlabel("Length")

plt.style.use("ggplot")
plt.subplot(1, 2, 2)
sns.distplot(data_df["de"].str.split().apply(len))
plt.title("Distribution of German sentences length")
plt.xlabel("Length")
plt.show()

# Use maximum lengths from both
seq_len_en = 20
seq_len_de = 20
MAX_LENGTH = 20

train_df, valid_df = train_test_split(
    data_df, test_size=0.1, shuffle=True, random_state=28
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print(f"Shape of train set: {train_df.shape}")
print(f"Shape of val set: {valid_df.shape}")

print(f'{"="*31}\nSample entries from the dataset\n{"="*31}')
for i in range(len(train_df) - 5, len(train_df)):
    print(
        f'ENGLISH:\n{train_df.iloc[i]["en"]},\nGERMAN:\n{train_df.iloc[i]["de"]}\n{"="*92}'
    )


class Vocabulary:
    def __init__(
        self, freq_threshold=2, language="en", preprocessor=None, reverse=False
    ):
        self.itos = {
            0: "<pad>",
            1: "<sos>",
            2: "<eos>",
            3: "<unk>",
        }  # integer to string/token
        self.stoi = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }  # string/token to integer
        self.tokenizer = spacy.load(language)
        self.freq_threshold = freq_threshold
        self.preprocessor = preprocessor
        self.reverse = reverse

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        if self.reverse:
            return [token.text.lower() for token in self.tokenizer.tokenizer(text)][
                ::-1
            ]
        return [token.text.lower() for token in self.tokenizer.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = len(self.itos)

        for sentence in sentence_list:
            # Preprocess the sentence using given preprocessor.
            if self.preprocessor:
                sentence = self.preprocessor(sentence)

            for word in self.tokenize(sentence):
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

                # only those words that are >= freq threshold
                # is added to the vocabulary
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()
    return w


# Commented out IPython magic to ensure Python compatibility.
# %%time
# Build vocab using training data
freq_threshold = 2
en_vocab = Vocabulary(
    freq_threshold=freq_threshold,
    language="en_core_web_sm",
    preprocessor=preprocess_sentence,
    reverse=False,
)
de_vocab = Vocabulary(
    freq_threshold=freq_threshold,
    language="de_core_news_sm",
    preprocessor=preprocess_sentence,
    reverse=True,
)

# print(len(train_df["en"].tolist()))
# print(len(train_df["de"].tolist()))

# build vocab for both english and german
en_vocab.build_vocabulary(train_df["en"].tolist())
de_vocab.build_vocabulary(train_df["de"].tolist())


class CustomTranslationDataset(Dataset):
    def __init__(self, df, en_vocab: Vocabulary, de_vocab: Vocabulary):
        super().__init__()
        self.df = df
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab

    def __len__(self):
        return len(self.df)

    def _get_numericalized(self, sentence, vocab: Vocabulary):
        """Numericalize given text using prebuilt vocab."""
        # With start and end of string tokens
        numericalized = [vocab.stoi["<sos>"]]
        numericalized.extend(vocab.numericalize(sentence))
        numericalized.append(vocab.stoi["<eos>"])
        return numericalized

    def __getitem__(self, index):
        # To get integer tensors of en and de for a particular index
        en_numericalized = self._get_numericalized(
            self.df.iloc[index]["en"], self.en_vocab
        )
        de_numericalized = self._get_numericalized(
            self.df.iloc[index]["de"], self.de_vocab
        )

        return torch.tensor(de_numericalized), torch.tensor(en_numericalized)


class CustomCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Pads tensors to the maximum length available in the whole dataset
        src = [item[0] for item in batch]
        src = pad_sequence(src, batch_first=False, padding_value=self.pad_idx)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return src, targets


BATCH_SIZE = 256

# Define dataset and dataloader
train_dataset = CustomTranslationDataset(train_df, en_vocab, de_vocab)
valid_dataset = CustomTranslationDataset(valid_df, en_vocab, de_vocab)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    collate_fn=CustomCollate(pad_idx=en_vocab.stoi["<pad>"]),
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    collate_fn=CustomCollate(pad_idx=en_vocab.stoi["<pad>"]),
)

fun_de = np.vectorize(lambda x: de_vocab.itos[x])
fun_en = np.vectorize(lambda x: en_vocab.itos[x])

print(f"Unique tokens in source (de) vocabulary: {len(de_vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(en_vocab)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""## Modeling"""


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        outputs, (hidden_state, cell_state) = self.lstm(x)

        return hidden_state, cell_state


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state, cell_state):
        x = x.unsqueeze(0)
        x = self.embedding(x)
        x = self.dropout(x)
        outputs, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        preds = self.fc(outputs.squeeze(0))
        return preds, hidden_state, cell_state


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Attention layers
        self.attn = nn.Linear(emb_dim + hidden_dim, max_length)
        self.attn_combine = nn.Linear(emb_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden_state, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_state[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden_state = self.lstm(output, hidden_state)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden_state, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert self.encoder.hidden_dim == decoder.hidden_dim
        assert self.encoder.n_layers == decoder.n_layers

    def forward(self, x, y, teacher_forcing_ratio=0.75):
        target_len = y.shape[0]
        batch_size = y.shape[1]
        target_vocab_size = self.decoder.output_dim  # Output dim

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Encode the source text using encoder
        hidden_state, cell_state = self.encoder(x)

        # First input is <sos>
        input = y[0, :]

        # Decode the encoded vector using decoder
        for t in range(1, target_len):
            output, hidden_state, cell_state = self.decoder(
                input, hidden_state, cell_state
            )
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            pred = output.argmax(1)
            input = y[t] if teacher_force else pred

        return outputs


class EncoderDecoderAttention(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AttnDecoderRNN):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert self.encoder.hidden_dim == decoder.hidden_dim
        assert self.encoder.n_layers == decoder.n_layers

    def forward(self, x, y, teacher_forcing_ratio=0.75):
        target_len = y.shape[0]
        batch_size = y.shape[1]
        target_vocab_size = self.decoder.output_dim  # Output dim

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Encode the source text using encoder
        hidden_state, cell_state = self.encoder(x)

        # First input is <sos>
        input = y[0, :]

        # Decode the encoded vector using decoder
        for t in range(1, target_len):
            output, hidden_state, cell_state = self.decoder(
                input, hidden_state, cell_state
            )
            outputs[t] = output
            teacher_for3 - Neural Machine Translation by Jointly Learning to Align and Translatece = random.random() < teacher_forcing_ratio
            pred = output.argmax(1)
            input = y[t] if teacher_force else pred

        return outputs


# Initialize all models
input_dim = len(de_vocab)
output_dim = len(en_vocab)
emb_dim = 256
hidden_dim = 512
n_layers = 4
dropout = 0.4

encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
decoder = AttnDecoderRNN(output_dim, emb_dim, hidden_dim, n_layers, dropout)
model = EncoderDecoderAttention(encoder, decoder).to(device)


# Initialized weights as defined in paper
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.stoi["<pad>"])
optimizer = optim.Adam(model.parameters())


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in tqdm(
        enumerate(iterator), total=len(iterator), position=0, leave=True
    ):
        src = batch[0].to(device)
        trg = batch[1].to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(iterator), total=len(iterator), position=0, leave=True
        ):
            src = batch[0].to(device)
            trg = batch[1].to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def inference(model, sentence):
    model.eval()
    result = []3 - Neural Machine Translation by Jointly Learning to Align and Translate

    with torch.no_grad():
        sentence = sentence.to(device)

        hidden_state, cell_state = model.encoder(sentence)

        # First input to decoder is "<sos>"
        inp = torch.tensor([en_vocab.stoi["<sos>"]]).to(device)

        # Decode the encoded vector using decoder until max length is reached or <eos> is generated.
        for t in range(1, seq_len_en):
            output, hidden_state, cell_state = model.decoder(
                inp, hidden_state, cell_state
            )
            pred = output.argmax(1)
            if pred == en_vocab.stoi["<eos>"]:
                break
            result.append(en_vocab.itos[pred.item()])
            inp = pred

    return " ".join(res3 - Neural Machine Translation by Jointly Learning to Align and Translateult)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


for sample_batch in valid_loader:
    break

N_EPOCHS = 12
CLIP = 1

best_valid_loss = float("inf")

sample_source = " ".join(
    [
        word
        for word in fun_de(sample_batch[0][:, 101])
        if word not in ["<pad>", "<sos>", "<eos>"]
    ]
)
sample_target = " ".join(
    [3 - Neural Machine Translation by Jointly Learning to Align and Translate
        word
        for word in fun_en(sample_batch[1][:, 101])
        if word not in ["<pad>", "<sos>", "<eos>"]
    ]
)

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best_model.pt")

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")
    print(f"\t Sample Source (German): {sample_source}")
    print(f"\t Sample Target (English): {sample_target}")
    print(f"\t Generated: {inference(model, sample_batch[0][:, 101].reshape(-1, 1))}\n")

# Load the best model.
model_path = "./best_model.pt"
model.load_state_dict(torch.load(model_path))

"""## Results"""

for idx in range(20):
    print(
        f'ACTUAL GERMAN: {" ".join([word for word in fun_de(sample_batch[0][:, idx]) if word not in ["<pad>", "<sos>", "<eos>"]])}'
    )
    print(
        f'ACTUAL: ENGLISH: {" ".join([word for word in fun_en(sample_batch[1][:, idx]) if word not in ["<pad>", "<sos>", "<eos>"]])}'
    )
    print(
        f"GENERATED BY MODEL: {inference(model, sample_batch[0][:, idx].reshape(-1, 1))}"
    )
    print("=" * 92)

"""As you can see, this works well on short sentences."""
