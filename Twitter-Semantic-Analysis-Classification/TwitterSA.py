import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List

import pandas as pd
import numpy as np

import re

import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
import transformers

from transformers import (ElectraForSequenceClassification,
                          ElectraTokenizerFast,
                          EvalPrediction,
                          InputFeatures,
                          Trainer,
                          TrainingArguments,
                          get_linear_schedule_with_warmup)

import tweetnlp
import tweetnlp.text_classification
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

class TrainerDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer

        # Tokenize the input
        self.tokenized_inputs = tokenizer(inputs, padding=True)   

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return InputFeatures(
            input_ids=self.tokenized_inputs['input_ids'][idx],
            token_type_ids=self.tokenized_inputs['token_type_ids'][idx],
            attention_mask=self.tokenized_inputs['attention_mask'][idx],
            label=self.targets[idx])

# function to perform basic preprocessing
def preprocess(inputs: list) -> list:
    outputs = []

    for input in inputs:
        # remove '@user' tags as these are present in almost every tweet
        input = re.sub(r"@user", "", input)
        # reformat emojis from [token]\uxxxx to [token][space]\uxxx
        input = re.sub(r"\\u", r" \\u", input)
        # remove select punctuation
        input = re.sub(r"[\.|'|\"|,]", "", input)
        # convert &amp; into &
        input = re.sub(r"&amp;", "&", input)
        
        outputs.append(input)

    return outputs


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'precision': precision_recall_fscore_support(p.label_ids, preds, average='weighted')[0],
        'recall': precision_recall_fscore_support(p.label_ids, preds, average='weighted')[1],
        'f1': precision_recall_fscore_support(p.label_ids, preds, average='weighted')[2],
        'mathews' : matthews_corrcoef(p.label_ids, preds),
    }

def data_collator(features: List[InputFeatures]) -> Dict[str, torch.Tensor]:
    batch = {}
    batch["input_ids"] = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    batch["attention_mask"] = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    batch["token_type_ids"] = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    batch["labels"] = torch.tensor([f.label for f in features], dtype=torch.long)
    return batch

class TestBench():

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels = 3)
        model.to(device)

        tokeniser = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator", do_lower_case=True)

        dataset, labels = tweetnlp.load_dataset('sentiment')
        train_dataset = TrainerDataset(preprocess(dataset["train"]["text"]), dataset["train"]["label"], tokeniser)
        test_dataset = TrainerDataset(preprocess(dataset["test"]["text"]), dataset["test"]["label"], tokeniser)
        val_dataset = TrainerDataset(preprocess(dataset["validation"]["text"]), dataset["validation"]["label"], tokeniser)
        
        training_args = TrainingArguments(
                            output_dir="./models/model_electra",
                            evaluation_strategy = "epoch", # evaluate every epoch
                            save_strategy = "epoch", # save every epoch
                            num_train_epochs=5,
                            learning_rate=1e-5,
                            per_device_train_batch_size=64,
                            per_device_eval_batch_size=64,
                            warmup_steps=500,
                            weight_decay=0.001,
                            logging_dir='./logs',
                            logging_steps=100,
                            load_best_model_at_end=True,
                            metric_for_best_model="accuracy",
                            greater_is_better=True,
                            dataloader_drop_last=True,  # Make sure all batches are of equal size
                            save_total_limit=5,
                        )



        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-7)
        num_training_steps = len(train_dataset) / training_args.per_device_train_batch_size * training_args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler)
        )

    def report(self):
        # Train the model
        self.trainer.train()

        # Get the training logs
        logs = self.trainer.state.log_history

        # Extract the loss values for each epoch
        train_history, test_history = {},{}
        for log in logs:
            if log.get('loss'):
                train_history[log.get('epoch')] = log.get('loss')
            if log.get('eval_loss'):
                test_history[log.get('epoch')] = log.get('eval_loss')

        # Plot the loss curves
        plt.plot(train_history.keys(), train_history.values(), label='Training loss')
        plt.plot(test_history.keys(),test_history.values(), label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()