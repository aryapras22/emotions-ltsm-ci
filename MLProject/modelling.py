import os
import json
import pickle
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("artifacts", exist_ok=True)

MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Emotion_LSTM_Baseline")

train_df = pd.read_csv("emotions_preprocessing/train_preprocessed.csv")
val_df = pd.read_csv("emotions_preprocessing/val_preprocessed.csv")
test_df = pd.read_csv("emotions_preprocessing/test_preprocessed.csv")

X_train_text = train_df["text_clean"].astype(str).tolist()
X_val_text = val_df["text_clean"].astype(str).tolist()
X_test_text = test_df["text_clean"].astype(str).tolist()

y_train_text = train_df["emotions"].astype(str).tolist()
y_val_text = val_df["emotions"].astype(str).tolist()
y_test_text = test_df["emotions"].astype(str).tolist()

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_text)
y_val = label_encoder.transform(y_val_text)
y_test = label_encoder.transform(y_test_text)

with open("artifacts/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

max_words = 20000
max_len = 50
embedding_dim = 128

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(X_train_text),
    maxlen=max_len,
    padding="post",
    truncating="post",
)
X_val = pad_sequences(
    tokenizer.texts_to_sequences(X_val_text),
    maxlen=max_len,
    padding="post",
    truncating="post",
)
X_test = pad_sequences(
    tokenizer.texts_to_sequences(X_test_text),
    maxlen=max_len,
    padding="post",
    truncating="post",
)

with open("artifacts/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

config = {
    "max_words": max_words,
    "max_len": max_len,
    "embedding_dim": embedding_dim,
    "num_classes": len(label_encoder.classes_),
}
with open("artifacts/preprocessing_config.json", "w") as f:
    json.dump(config, f, indent=2)

model = Sequential(
    [
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

mlflow.tensorflow.autolog()

with mlflow.start_run():
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_acc)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(report)

    with open("artifacts/classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("artifacts/tokenizer.pkl")
    mlflow.log_artifact("artifacts/label_encoder.pkl")
    mlflow.log_artifact("artifacts/preprocessing_config.json")
    mlflow.log_artifact("artifacts/classification_report.txt")
