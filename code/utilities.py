from glob import glob
from os.path import join
from pickle import dump, load

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification
from transformers.data.processors.utils import InputFeatures

from arabert.preprocess import ArabertPreprocessor

def get_SMADC_folder_data(path_to_code_folder=""):
    """Returns a dataframe with Text and Region columns."""
    files = glob(join(path_to_code_folder, "data/SMADC/*.txt"))
    dataframes = []

    for file in files:
        region = file[-7:-4]
        temp_df = pd.read_csv(file, encoding="utf8", delimiter="\n", names=["Text"])
        temp_df["Region"] = region
        dataframes.append(temp_df)
    
    return pd.concat(dataframes)

def get_music_df(path_to_code_folder=""):
    files = ["GLF","LEV","NOR","IRQ"]
    dataframes = []
    
    for file in files:
        temp_df = pd.read_csv(join(path_to_code_folder, f"data/extra_data/d7_data/{file}.txt"), encoding="utf8", delimiter="\n", names=["Text"])
        temp_df["Region"] = file
        dataframes.append(temp_df)
    
    return pd.concat(dataframes)

def tokenize(tokenizer, batch, sequence_length):
    """Tokenizes a list of strings"""
    return tokenizer.batch_encode_plus(
        batch,
        add_special_tokens=True,
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )

def batch_tokenize_iter(data, batch_size):
    len_data = len(data)
    batch_num = len_data // batch_size
    batch_rest = len_data / batch_size - batch_num
    
    for i in range(batch_size):
        yield tokenize(data[i * batch_num:(i+1) * batch_num].to_list())
        
    if batch_rest:
        yield tokenize(data[batch_num:].to_list())

def batch_tokenize(data, batch_size):
    bt = batch_tokenize_iter(data, batch_size)
    for i, tokenization in enumerate(bt):
        if not i:
            encoding = tokenization
            continue
        encoding["input_ids"] = torch.cat([encoding["input_ids"], tokenization["input_ids"]])
        encoding["attention_mask"] = torch.cat([encoding["attention_mask"], tokenization["attention_mask"]])
    return encoding

def preprocess_sample(sample):
    """Sample list of strings"""
    return tokenize(list(arabert_prep.preprocess(text) for text in sample))

def save_preprocessed_data(dataset, dataset_name):
    with open(f"preprocessed_data/{dataset_name}.pkl", "wb") as file:
        dump(dataset, file)
        
def load_preprocessed_data(dataset_name):
    with open(f"preprocessed_data/{dataset_name}.pkl", "rb") as file:
        temp = load(file)
    return temp

def model_init(model_name, num_labels, label2id, id2label):
    return AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=num_labels, label2id=label2id, id2label=id2label)

def compute_metrics(p): 
    preds = np.argmax(p.predictions, axis=1)
    assert len(preds) == len(p.label_ids)

    macro_f1 = f1_score(p.label_ids,preds,average='macro')
    macro_precision = precision_score(p.label_ids,preds,average='macro')
    macro_recall = recall_score(p.label_ids,preds,average='macro')
    acc = accuracy_score(p.label_ids,preds)
    return {
      'macro_f1' : macro_f1,
      'macro_precision': macro_precision,
      'macro_recall': macro_recall,
      'accuracy': acc
    }

# Dataset class
class Dialect_dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super(Dialect_dataset).__init__()
        self.X = X
        self.Y = Y
        
    def __getitem__(self, key):
        return InputFeatures(self.X["input_ids"][key], self.X["attention_mask"][key], label=self.Y[key])
        
    def __len__(self):
        return len(self.X["input_ids"])