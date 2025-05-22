import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

""" Ce script teste le modèle sur le dataset d'entraînement puis évalue ses performances. Il utilise des mesures intrinsèques (précision, rappel, f-mesure) et sauvegarde les résultats de l'évaluation dans des fichiers png, enregistrés dans le dossier figures. """


# Chargement du dataset de test et du modèle finetuné
test_df = pd.read_csv("../../data/clean/test.csv")

label_list = sorted(test_df["label"].unique().tolist())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}


test_df["label_id"] = test_df["label"].map(label2id)


model_path = "../../bin/my_distilbert_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)


# Préparation des données de test pour l'évaluation du modèle
def tokenize_data(data):
    return tokenizer(data["text"], truncation=True, padding="max_length")

test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]])
test_dataset = test_dataset.map(tokenize_data, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

trainer = Trainer(model=model, tokenizer=tokenizer)

# Définition des prédictions et des vraies étiquettes
predictions_output = trainer.predict(test_dataset)
y_pred = predictions_output.predictions.argmax(axis=1)
y_true = test_df["label_id"].values


# Calcul de la précision, du rappel et de la f-mesure (mesures intrinsèques)
def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=label_list, output_dict=True)

    metrics_names = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']
    metrics_values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(metrics_names, metrics_values, color='skyblue')
    plt.ylim(0, 1.05)
    plt.title("Métriques d'évaluation du modèle")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("../../figures/metrics_summary.png")
    plt.close()


# Calcul de la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, label_names, title="Matrice de confusion"):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("Vraies étiquettes")
    plt.xlabel("Étiquettes prédites")
    plt.tight_layout()
    plt.savefig("../../figures/evaluation.png")
    plt.close()

# Évaluation du modèle
print_metrics(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, label_names=label_list)
