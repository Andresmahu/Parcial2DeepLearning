import pandas as pd, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar modelo y generador test (idéntico al entrenamiento)
model = load_model('Models/mejor_modelo.h5', compile=False)
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_dataframe(
    dataframe=pd.read_csv('data/test/etiquetas_test.csv'),
    directory='data/test/test_cropped',
    x_col='filename', y_col='label',
    target_size=(224,224), class_mode='raw', shuffle=False)

# Predicciones y discrepancias
probs = model.predict(test_gen, verbose=0)
y_pred = probs.argmax(1)
y_true = test_gen.labels
filenames = test_gen.filenames

# Extraer los errores
errors = np.where(y_pred != y_true)[0]
error_df = pd.DataFrame({
    "filename": [filenames[i] for i in errors],
    "y_true": y_true[errors],
    "y_pred": y_pred[errors],
    "prob_pred": probs[errors].max(1)
})
error_df.to_csv("errores_modelo.csv", index=False)
print(f"{len(errors)} errores guardados en errores_modelo.csv")
