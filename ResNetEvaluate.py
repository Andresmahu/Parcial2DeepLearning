import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.applications.resnet import preprocess_input

# =========================
# Paths y parámetros
# =========================
BEST_H5   = "mejor_modelo.h5"                 # usa el mejor por val_accuracy
ALT_H5    = "resnet_transfer_learning.h5"
CSV_TEST  = "data/test/etiquetas_test.csv"    # columnas: filename, label
TEST_DIR  = "data/test/test_cropped"          # carpeta base para filename relativo
IMG_SIZE  = (224, 224)
BATCH     = 64

# =========================
# Cargar modelo (inferencia)
# =========================
h5_path = BEST_H5 if os.path.exists(BEST_H5) else ALT_H5
model = load_model(h5_path, compile=False)
print(f"Evaluando modelo: {h5_path}")

# =========================
# Cargar CSV test y validar
# =========================
df = pd.read_csv(CSV_TEST)
required = {"filename", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"El CSV debe tener columnas {required}")

# Normaliza rutas
def to_path(p):
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(TEST_DIR, p)

df["filepath"] = df["filename"].apply(to_path)
exists = df["filepath"].apply(os.path.exists)
if (~exists).any():
    print(f"[AVISO] {(~exists).sum()} imágenes no existen. "
          f"Ejemplos: {df.loc[~exists, 'filename'].head(5).tolist()}")
    df = df.loc[exists].reset_index(drop=True)

if df.empty:
    raise RuntimeError("No hay imágenes válidas en test.")

# Etiquetas deben ser enteras 0..9 (sparse)
if df["label"].dtype.kind in {"U","S","O"}:
    raise ValueError("Tus etiquetas están en texto. Convierte a índices enteros 0..9 como en entrenamiento.")

df["label"] = df["label"].astype(int)
y_true = df["label"].to_numpy()

# Chequeos rápidos
n_classes_expected = 10
uniq = np.unique(y_true)
if y_true.min() < 0 or y_true.max() >= n_classes_expected:
    raise ValueError(f"Etiqueta fuera de rango (min={y_true.min()}, max={y_true.max()}). "
                     f"Se esperaban índices en 0..{n_classes_expected-1}.")

print(f"Total imágenes test: {len(df)} | Clases presentes: {uniq.tolist()}")

# =========================
# Generador de test (x/255)
# =========================
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=df,
    directory=None,                 # ya pasamos rutas completas en x_col
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="raw",            # etiquetas enteras (sparse)
    shuffle=False
)

# =========================
# Predicción y métricas
# =========================
probs = model.predict(test_gen, verbose=1)     # (N, 10) esperado
if probs.ndim != 2 or probs.shape[1] != n_classes_expected:
    raise RuntimeError(f"Forma inesperada de predicciones: {probs.shape}. "
                       f"Se esperaba (N, {n_classes_expected}). ¿Guardaste el modelo correcto?")

y_pred = probs.argmax(axis=1)

acc  = accuracy_score(y_true, y_pred)
f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"\n== Métricas globales ==\nAccuracy: {acc:.4f} | F1-macro: {f1m:.4f}\n")

print("== Classification report ==")
print(classification_report(y_true, y_pred, zero_division=0))

print("== Matriz de confusión ==")
print(confusion_matrix(y_true, y_pred, labels=list(range(n_classes_expected))))

# =========================
# Exportar resultados
# =========================
out = df[["filename"]].copy()
out["y_true"] = y_true
out["y_pred"] = y_pred
for i in range(probs.shape[1]):
    out[f"prob_{i}"] = probs[:, i]
out.to_csv("eval_resultados_test_ResNet.csv", index=False)
print("\nResultados guardados en: eval_resultados_test_resnet.csv")
