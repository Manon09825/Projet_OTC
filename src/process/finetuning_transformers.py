import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

""" Ce script sert à finetuner un transformer de Huggingface préentraîné. Il n'utilise que le dataset d'entraînement. Il convertit le dataframe pandas (dans l'ensemble d'entraînement) en dataset lisible par HuggingFace. Il charge ensuite le modèle (DistilBert), puis l'entraîne. """

df = pd.read_csv("../../data/clean/train.csv")


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize_data(data):
    return tokenizer(data["text"], truncation=True, padding="max_length")

train_dataset = Dataset.from_pandas(df[["text", "label"]])

label_list = df["label"].unique().tolist()
label2id = {label: idx for idx, label in enumerate(label_list)}

train_dataset = train_dataset.map(lambda x: {"label": label2id[x["label"]]})

train_dataset = train_dataset.map(tokenize_data, batched=True)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label2id))

# Définition des paramètres d'entraînement du modèle, puis phase d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Sauvegarde du modèle finetuné
model.save_pretrained("../../bin/my_distilbert_model")
tokenizer.save_pretrained("../../bin/my_distilbert_model")
