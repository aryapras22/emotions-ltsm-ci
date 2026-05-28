import os
import json
import pickle
import argparse
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="emotions_preprocessing")
    parser.add_argument("--max_words", type=int, default=20000)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train_preprocessed.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_preprocessed.csv"))

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

    tokenizer = Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(X_train_text),
        maxlen=args.max_len,
        padding="post",
        truncating="post",
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(X_val_text),
        maxlen=args.max_len,
        padding="post",
        truncating="post",
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(X_test_text),
        maxlen=args.max_len,
        padding="post",
        truncating="post",
    )

    with open("artifacts/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_words": args.max_words,
        "max_len": args.max_len,
        "embedding_dim": args.embedding_dim,
        "num_classes": len(label_encoder.classes_),
    }
    with open("artifacts/preprocessing_config.json", "w") as f:
        json.dump(config, f, indent=2)

    model = Sequential(
        [
            tf.keras.layers.Input(shape=(args.max_len,)),
            Embedding(input_dim=args.max_words, output_dim=args.embedding_dim),
            LSTM(
                args.lstm_units,
                dropout=args.dropout_rate,
                recurrent_dropout=args.dropout_rate,
            ),
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

    mlflow.set_tag("project_type", "MLflow Project CI")
    mlflow.set_tag("model_type", "LSTM Text Classification")
    mlflow.set_tag("dataset", "dair-ai/emotion")

    mlflow.log_params(
        {
            "max_words": args.max_words,
            "max_len": args.max_len,
            "embedding_dim": args.embedding_dim,
            "lstm_units": args.lstm_units,
            "dropout_rate": args.dropout_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    for i in range(len(history.history["loss"])):
        mlflow.log_metric("train_loss", float(history.history["loss"][i]), step=i)
        mlflow.log_metric("val_loss", float(history.history["val_loss"][i]), step=i)
        mlflow.log_metric(
            "train_accuracy", float(history.history["accuracy"][i]), step=i
        )
        mlflow.log_metric(
            "val_accuracy", float(history.history["val_accuracy"][i]), step=i
        )

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    print("Test Accuracy:", test_acc)
    print(report)

    mlflow.log_metric("test_accuracy", float(test_acc))

    with open("artifacts/classification_report.txt", "w") as f:
        f.write(report)

    model.save("artifacts/model.keras")
    mlflow.log_artifact("artifacts/model.keras")
    mlflow.log_artifact("artifacts/tokenizer.pkl")
    mlflow.log_artifact("artifacts/label_encoder.pkl")
    mlflow.log_artifact("artifacts/preprocessing_config.json")
    mlflow.log_artifact("artifacts/classification_report.txt")

    mlflow.tensorflow.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
