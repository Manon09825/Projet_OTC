import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


df = pd.read_csv("../../data/clean/minimalist_baker_recipes_balanced.csv")

df["text"] = df["ingredients_clean"] + " " + df["instructions_clean"]

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_data(data):
    return tokenizer(data["text"], truncation=True, padding="max_length")

hf_dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
hf_dataset = hf_dataset.train_test_split(test_size=0.2)
tokenized_dataset = hf_dataset.map(tokenize_data, batched=True)


model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)


trainer.train()

model.save_pretrained("../../bin/my_distilbert_model")
tokenizer.save_pretrained("../../bin/my_distilbert_model")

