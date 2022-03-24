from glob import glob
from os.path import join
from os import path
import os
from pickle import dump, load

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification
from transformers.data.processors.utils import InputFeatures
from transformers import Trainer, TrainingArguments

from arabert.preprocess import ArabertPreprocessor

def get_SMADC_folder_data(code_folder_path=""):
    """Returns a dataframe with Text and Region columns."""
    files = glob(join(code_folder_path, "data/SMADC/*.txt"))
    dataframes = []

    for file in files:
        region = file[-7:-4]
        temp_df = pd.read_csv(file, encoding="utf8", delimiter="\n", names=["Text"])
        temp_df["Region"] = region
        dataframes.append(temp_df)
    
    return pd.concat(dataframes)

def get_annotated_data_folder_data(code_folder_path=""):
    """Returns a dataframe with Text and Region columns."""
    regions = ["egyptian", "gulf", "iraqi", "levantine", "maghrebi"]
    labels = ["EGY", "GLF", "IRQ", "LEV", "NOR"]
    region_to_label = {region:label for region, label in zip(regions, labels)}
    files = [join(code_folder_path, "data", "annotated_data", region) for region in regions]
    
    dfs = [pd.read_csv(file, encoding="utf8", sep="\t", names=["Region", "Text"])[2:] 
        for file in files]
    
    dfs = pd.concat(dfs)
    dfs["Region"] = dfs["Region"].apply(region_to_label.get)
    return dfs

def get_arabic_dialects_dataset_folder_data(code_folder_path=""):
    """Returns a dataframe with Text and Region columns."""
    # CHECK LAV MEANS LEV? # MENTION NO IRQ
    # Filter low quality test?
    labels = ["EGY", "GLF", "LEV", "NOR"]
    regions = [f"all{label if label != 'LEV' else 'LAV'}.txt" for label in labels]
    region_to_label = {region:label for region, label in zip(regions, labels)}
    files = [join(code_folder_path, "data", "ArabicDialectsDataset", "Dialects Full Text", region)
                for region in regions]

    dfs = []
    for file in files:
        norm_file_path = path.normpath(file)
        file_name = norm_file_path.split(os.sep)[-1]

        df = pd.read_csv(file, encoding="utf8", names=["Text"])
        df["Region"] = region_to_label[file_name]
        dfs.append(df)
        
    return pd.concat(dfs)

def get_dart_folder_data(code_folder_path=""):
    """Returns a dataframe with Text and Region columns."""
    # What about dart gold? what is it
    # Filter low quality test?
    labels = ["EGY", "GLF", "IRQ", "LEV", "NOR"]
    regions = [region + ".txt" for region in ["EGY", "GLF", "IRQ", "LEV", "MGH"]]
    region_to_label = {region:label for region, label in zip(regions, labels)}
    files = [join(code_folder_path, "data", "DART", "cf-data", region)
                for region in regions]

    dfs = []
    for file in files:
        norm_file_path = os.path.normpath(file)
        file_name = norm_file_path.split(os.sep)[-1]

        df = pd.read_csv(file, delimiter="\t",encoding="utf8", names=["_", "__", "Text"])
        df["Region"] = region_to_label[file_name]
        df = df[["Text", "Region"]].iloc[1:]
        dfs.append(df)
        
    return pd.concat(dfs)

def get_music_df(code_folder_path=""):
    files = ["GLF","LEV","NOR","IRQ"]
    dataframes = []
    
    for file in files:
        temp_df = pd.read_csv(join(code_folder_path, f"data/extra_data/d7_data/{file}.txt"), encoding="utf8", delimiter="\n", names=["Text"])
        temp_df["Region"] = file
        dataframes.append(temp_df)
    
    return pd.concat(dataframes)

def get_arabic_lexicon_data(code_folder_path=""):
    """Returns a dictionary of emotions and a list of words that represent them. e.g. {emotion1: [word1, word2, word3, ..], ..}"""
    path_to_files = join(code_folder_path, "data", "emotion-lexicon-master", "arb", "*")
    paths = glob(path_to_files)
    emotions = [path[path.rfind("\\")+1:path.rfind(".")] for path in paths]
    emotion_to_words = {}
    for emotion, path in zip(emotions, paths):
        with open(path, encoding="utf8") as file:
            emotion_to_words[emotion] = file.read().split()
    return emotion_to_words

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

def batch_tokenize_iter(tokenizer, batch, batch_size, sequence_length):
    len_batch = len(batch)
    batch_num = len_batch // batch_size
    batch_rest = len_batch / batch_size - batch_num
    
    for i in range(batch_size):
        yield tokenize(tokenizer, batch[i * batch_num:(i+1) * batch_num].to_list(), sequence_length)
        
    if batch_rest:
        yield tokenize(tokenizer, batch[batch_num:].to_list(), sequence_length)

def batch_tokenize(tokenizer, batch, batch_size, sequence_length):
    bt = batch_tokenize_iter(tokenizer, batch, batch_size, sequence_length)
    for i, tokenization in enumerate(bt):
        if not i:
            encoding = tokenization
            continue
        encoding["input_ids"] = torch.cat([encoding["input_ids"], tokenization["input_ids"]])
        encoding["attention_mask"] = torch.cat([encoding["attention_mask"], tokenization["attention_mask"]])
    return encoding

def preprocess_sample(tokenizer, sample, sequence_length, arabert_prep):
    """Sample list of strings"""
    return tokenize(tokenizer, list(arabert_prep.preprocess(text) for text in sample), sequence_length)

def save_preprocessed_data(dataset, dataset_name):
    with open(f"preprocessed_data/{dataset_name}.pkl", "wb") as file:
        dump(dataset, file)
        
def load_preprocessed_data(dataset_name):
    with open(f"preprocessed_data/{dataset_name}.pkl", "rb") as file:
        temp = load(file)
    return temp

def model_init(model_name, num_labels, label2id, id2label):
    return AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=num_labels, label2id=label2id, id2label=id2label)

def generate_training_args(output_dir, epochs=5, do_warmup=True, warmup_ratio=0.05, save_model=True, 
        eval_while_training=True, learning_rate=1e-5, batch_size=32, train_dataset_length=0, seed=42):
    training_args = TrainingArguments(output_dir)

    training_args.adam_epsilon = 1e-8
    training_args.learning_rate = learning_rate

    training_args.fp16 = True

    training_args.per_device_train_batch_size = batch_size
    training_args.per_device_eval_batch_size = batch_size

    training_args.gradient_accumulation_steps = 1
    
    if epochs:
        training_args.num_train_epochs = epochs

    if do_warmup:
        if not train_dataset_length:
            print("WARNING do_warmup is TRUE but train_dataset_length == 0. Set train_dataset_length")
        steps_per_epoch = train_dataset_length // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * training_args.num_train_epochs
        training_args.warmup_steps = total_steps * warmup_ratio 
    
    training_args.logging_steps = 10 ** 4
    
    if eval_while_training:
        training_args.evaluation_strategy = "steps"
        training_args.evaluate_during_training = True
        training_args.load_best_model_at_end = True
        training_args.eval_steps = 10 ** 4 # defaults to logging_steps
        training_args.metric_for_best_model = "macro_f1"
    
    if save_model:
        training_args.save_steps = 10 ** 4
        training_args.save_total_limit = 120
        training_args.save_strategy = "steps"

    training_args.seed = seed

    return training_args

def compute_metrics(p): 
    preds = np.argmax(p.predictions, axis=1)
    assert len(preds) == len(p.label_ids)

    return {
      'macro_f1' : f1_score(p.label_ids, preds, average= "macro"),
      "macro_precision": precision_score(p.label_ids, preds, average= "macro"),
      "macro_recall": recall_score(p.label_ids, preds, average= "macro"),
      "accuracy": accuracy_score(p.label_ids, preds),
      "report": classification_report(p.label_ids, preds, labels=['NOR', 'IRQ', 'LEV', 'EGY', 'GLF'], output_dict=True)
    }

def predict_dialect(model_path, dialect_text, tokenizer, preprocessing_function, sequence_length):
    id2label = {0 : "EGY", 1 : "GLF", 2 : "IRQ", 3 : "LEV", 4 : "NOR"}
     
    df = pd.DataFrame({"Text" : [dialect_text]})
    df["Text"] = df["Text"].apply(preprocessing_function)
    df_encoding = tokenize(tokenizer, df["Text"].to_list(), sequence_length)
    
    prediction_input = Dialect_dataset(df_encoding, [1])

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model)
    
    prediction = trainer.predict(prediction_input)
    label_id = np.argmax(prediction[0])
    
    return id2label[label_id]

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

