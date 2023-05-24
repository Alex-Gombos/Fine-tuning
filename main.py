import datasets 
from datasets import Dataset, ClassLabel, Sequence
import torch
import numpy as np 
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification
import json
import torch
from sklearn.model_selection import train_test_split


tokenizer = BertTokenizerFast.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1", do_lower_case=True,model_max_length=512)


with open('outputSentence.txt', 'r', encoding="utf8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)


label_names = sorted(set(label for labels in dataset["ner_tags"] for label in labels))
dataset = dataset.cast_column("ner_tags", Sequence(ClassLabel(names=label_names)))


# print(dataset.features["ner_ids"])

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

train_datasetDataset = Dataset.from_dict(train_dataset)
val_datasetDataset = Dataset.from_dict(val_dataset)



def tokenize_and_align_labels(examples, label_all_tokens=True): 
    tokenized_inputs = tokenizer(examples["tokens"], truncation= True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        previous_word_idx = None 
        label_ids = []
        for word_idx in word_ids: 
            if word_idx is None: 
                label_ids.append(-100)
            elif word_idx != previous_word_idx:               
                label_ids.append(label[word_idx]) 
            else: 
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs


train_datasetDataset = train_datasetDataset.cast_column("ner_tags", Sequence(ClassLabel(names=label_names)))
val_datasetDataset = val_datasetDataset.cast_column("ner_tags", Sequence(ClassLabel(names=label_names)))

tokenized_train = train_datasetDataset.map(tokenize_and_align_labels, batched=True, remove_columns=['space_after', 'ner_tags', 'id', 'tokens', 'ner_ids'])
tokenized_val = val_datasetDataset.map(tokenize_and_align_labels, batched=True, remove_columns=['space_after', 'ner_tags', 'id','tokens', 'ner_ids'])

# print(tokenized_train.features)


model = AutoModelForTokenClassification.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1", num_labels=10)

# print(model)


from transformers import TrainingArguments, Trainer 
args = TrainingArguments( 
"test-ner",
evaluation_strategy = "epoch", 
learning_rate=2e-5, 
per_device_train_batch_size=2, 
per_device_eval_batch_size=2, 
num_train_epochs=1, 
weight_decay=0.01,
remove_unused_columns=False
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
   train_dataset=tokenized_train, 
   eval_dataset=tokenized_val, 
   data_collator=data_collator, 
   tokenizer=tokenizer, 
   compute_metrics=compute_metrics,
)
torch.cuda.empty_cache()
trainer.train()
model.save_pretrained(r"D:\Alex\Licenta\model")