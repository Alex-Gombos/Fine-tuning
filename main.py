import datasets 
from datasets import Dataset
import torch
import numpy as np 
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification
import json
import torch

print(torch.cuda.memory_summary(device=None, abbreviated=False))

tokenizer = BertTokenizerFast.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1", do_lower_case=True,model_max_length=512)

with open('output.txt', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def tokenize_and_align_labels(examples, label_all_tokens=True): 
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_ids"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token. 
        previous_word_idx = None 
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None 
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids: 
            if word_idx is None: 
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token                 
                label_ids.append(label[word_idx]) 
            else: 
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                # mask the subword representations after the first subword
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1", num_labels=81)

from transformers import TrainingArguments, Trainer 
args = TrainingArguments( 
"test-ner",
evaluation_strategy = "epoch", 
learning_rate=2e-5, 
per_device_train_batch_size=2, 
per_device_eval_batch_size=2, 
num_train_epochs=1, 
weight_decay=0.01, 
)
data_collator = DataCollatorForTokenClassification(tokenizer)

metric = datasets.load_metric("seqeval")

example = dataset[0]

label_list = dataset.features["ner_tags"].feature.names 

labels = [label_list[i] for i in example["ner_tags"]] 


metric.compute(predictions=[labels], references=[labels])

def compute_metrics(eval_preds): 
    pred_logits, labels = eval_preds 
    
    pred_logits = np.argmax(pred_logits, axis=2) 
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax
    
    # We remove all the values where the label is -100
    predictions = [ 
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ] 
    
    true_labels = [ 
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(pred_logits, labels) 
   ] 
    results = metric.compute(predictions=predictions, references=true_labels) 
    return { 
   "precision": results["overall_precision"], 
   "recall": results["overall_recall"], 
   "f1": results["overall_f1"], 
  "accuracy": results["overall_accuracy"], 
  }

trainer = Trainer( 
    model, 
    args, 
   train_dataset=tokenized_datasets, 
   eval_dataset=tokenized_datasets, 
   data_collator=data_collator, 
   tokenizer=tokenizer, 
   compute_metrics=compute_metrics,
)
print(torch.cuda.memory_summary(device=None, abbreviated=False))
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
trainer.train()



