# import pandas as pd
# import numpy as np
# import seaborn as sns
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from xgboost import XGBClassifier
# from sklearn.preprocessing import LabelEncoder

# # Cargar datos (descarga el CSV de Kaggle y ponlo en tu directorio)
# df = pd.read_csv('machine failure.csv')  # Asume archivo train.csv del dataset

# # ==============================
# # 3. VALIDACIÓN DE VALORES NULOS
# # ==============================
# print("\n" + "="*50)
# print("VALIDACIÓN DE VALORES NULOS")
# print("="*50)

# # Conteo de nulos por columna
# nulos = df.isnull().sum()
# porcentaje_nulos = (nulos / len(df)) * 100

# # DataFrame resumen
# resumen_nulos = pd.DataFrame({
#     'Columna': nulos.index,
#     'Nulos': nulos.values,
#     '% Nulos': porcentaje_nulos.values
# })

# print(resumen_nulos)

# # Verificación final
# if nulos.sum() == 0:
#     print("\n✅ ¡NO HAY VALORES NULOS! El dataset está 100% completo.")
# else:
#     print("\n⚠️  Se encontraron valores nulos. Se requiere imputación.")
# # ==============================
# # 4. TIPOS DE DATOS Y EJEMPLOS
# # ==============================
# print("\n" + "="*50)
# print("TIPOS DE DATOS")
# print("="*50)
# print(df.dtypes)

# print("\nPrimeras 3 filas:")
# print(df.head(3))

# # ==============================
# # ==============================
# # 5. DUPLICADOS
# # ==============================
# duplicados = df.duplicated().sum()
# print(f"\nFilas duplicadas: {duplicados}")
# if duplicados == 0:
#     print("✅ No hay filas duplicadas.")
# else:
#     print("⚠️  Hay duplicados. Considera eliminarlos con df.drop_duplicates()")

# # ==============================
# # 6. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# # ==============================
# print("\n" + "="*50)
# print("DISTRIBUCIÓN DE 'Machine failure'")
# print("="*50)
# print(df['Machine failure'].value_counts())

# # Porcentaje
# porcentaje_fallos = df['Machine failure'].mean() * 100
# print(f"\nPorcentaje de fallos: {porcentaje_fallos:.2f}% → Dataset desbalanceado")

# # Gráfico
# plt.figure(figsize=(6,4))
# sns.countplot(x='Machine failure', data=df, palette='viridis')
# plt.title('Distribución de Machine Failure')
# plt.show()

# # ==============================
# # 7. ESTADÍSTICAS DESCRIPTIVAS (solo numéricas)
# # ==============================
# print("\n" + "="*50)
# print("ESTADÍSTICAS DESCRIPTIVAS")
# print("="*50)
# print(df.describe())


# # Hacer una copia limpia
# df_ml = df.copy()

# # Eliminar columnas no predictivas
# df_ml = df_ml.drop(['UDI', 'Product ID'], axis=1)

# # Codificar variable categórica 'Type'
# le = LabelEncoder()
# df_ml['Type'] = le.fit_transform(df_ml['Type'])

# # Separar X e y
# X = df_ml.drop('Machine failure', axis=1)
# y = df_ml['Machine failure']

# print("\nPreprocesamiento completado.")
# print(f"X shape: {X.shape}, y shape: {y.shape}")
# print("Columnas finales:", list(X.columns))

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from collections import Counter

# # Ver distribución actual
# print("Distribución original:", Counter(y))

# # Estrategia: SMOTE para oversamplear fallos + under para reducir no-fallos
# smote = SMOTE(sampling_strategy=0.3, random_state=42)  # 30% fallos
# undersample = RandomUnderSampler(sampling_strategy=0.6, random_state=42)  # 60% no-fallos

# # Pipeline
# pipeline = Pipeline([
#     ('smote', smote),
#     ('undersample', undersample)
# ])

# X_bal, y_bal = pipeline.fit_resample(X, y)

# print("Distribución balanceada:", Counter(y_bal))


# # Ver distribución final
# print("Distribución final:", Counter(y_bal))
# print(f"Porcentaje de fallos: {y_bal.mean()*100:.2f}%")

# # Gráfico comparativo
# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# sns.countplot(x=y, ax=ax[0], palette='Reds')
# ax[0].set_title('Antes del balanceo')
# ax[0].set_xlabel('Machine failure')

# sns.countplot(x=y_bal, ax=ax[1], palette='Greens')
# ax[1].set_title('Después del balanceo (SMOTE + Undersample)')
# ax[1].set_xlabel('Machine failure')

# plt.tight_layout()
# plt.show()

# # Split con datos balanceados
# X_train, X_test, y_train, y_test = train_test_split(
#     X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
# )

# def limpiar_nombres(df):
#     df = df.copy()
#     df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True)
#     df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
#     return df

# X_train = limpiar_nombres(X_train)
# X_test = limpiar_nombres(X_test)

# print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# # Modelos (sin class_weight porque ya está balanceado)
# rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
# xgb = XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', n_jobs=-1, verbosity=0)
# # 3. LightGBM
# lgbm = lgb.LGBMClassifier(
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1,
#     verbose=-1,  # Silenciar logs
#     scale_pos_weight=(y_train==0).sum() / (y_train==1).sum()  # Opcional, ya balanceado
# )

# # Entrenar
# rf.fit(X_train, y_train)
# xgb.fit(X_train, y_train)
# lgbm.fit(X_train, y_train)

# # Predecir
# y_pred_rf = rf.predict(X_test)
# y_pred_xgb = xgb.predict(X_test)
# y_pred_lgbm = lgbm.predict(X_test)

# from tabulate import tabulate


# # --- Evaluación ---
# modelos = [
#     ('Random Forest', rf),
#     ('XGBoost', xgb),
#     ('LightGBM', lgbm)
# ]

# resultados = []

# for nombre, modelo in modelos:
#     y_pred = modelo.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     resultados.append({
#         'Modelo': nombre,
#         'Accuracy': report['accuracy'],
#         'Precision (1)': report['1']['precision'],
#         'Recall (1)': report['1']['recall'],
#         'F1-Score (1)': report['1']['f1-score']
#     })

# # --- Tabla con tabulate (MUY BONITA) ---
# df_resultados = pd.DataFrame(resultados)
# df_resultados = df_resultados.sort_values('F1-Score (1)', ascending=False).round(4)

# # Multiplicar por 100 y agregar %
# df_print = df_resultados.copy()
# df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']] *= 100
# df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']] = df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']].astype(str) + '%'

# # Imprimir con estilo
# print("\n" + "═" * 70)
# print(" " * 20 + "TORNEO DE ALGORITMOS - RESULTADOS FINALES")
# print("═" * 70)

# print(tabulate(
#     df_print,
#     headers='keys',
#     tablefmt='pretty',
#     showindex=False,
#     stralign='center',
#     numalign='center'
# ))
# print("═" * 70)

# # ==============================
# # CREAR test_real.csv CON ETIQUETAS (PARA TUS PRUEBAS)
# # ==============================

# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 1. Cargar train original
# df = pd.read_csv('machine failure.csv')

# # 2. Dividir: 80% train, 20% test (con etiquetas)
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Machine failure'])

# # 3. Guardar
# train_df.to_csv('train_real.csv', index=False)
# test_df.to_csv('test_real.csv', index=False)

# print(f"train_real.csv: {train_df.shape}")
# print(f"test_real.csv: {test_df.shape} → ¡CON ETIQUETAS!")
# print("\n¡Archivos creados en tu carpeta!")

# # ==============================
# # CARGAR TU test_real.csv (CON ETIQUETAS)
# # ==============================

# test_df = pd.read_csv('test_real.csv')
# print(f"Test real: {test_df.shape}")
# print(test_df['Machine failure'].value_counts())  # Ver fallos

# # Preprocesar
# test_df = test_df.drop(['UDI', 'Product ID'], axis=1)
# test_df['Type'] = test_df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# def limpiar(df):
#     df = df.copy()
#     df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True).str.strip()
#     return df

# X_test = limpiar(test_df.drop('Machine failure', axis=1))
# y_test = test_df['Machine failure']

# # Predicciones
# pred_rf = rf.predict(X_test)
# pred_xgb = xgb.predict(X_test)
# pred_lgbm = lgbm.predict(X_test)

# # Resultados
# from sklearn.metrics import classification_report
# print("LIGHTGBM:")
# print(classification_report(y_test, pred_lgbm))
# print("RANDOM FOREST:")
# print(classification_report(y_test, pred_rf))
# print("XGBOOST:")
# print(classification_report(y_test, pred_xgb))

# # ==============================
# # FUNCIÓN: predecir_maquina()
# # Devuelve: predicción de RF, XGB, LGBM
# # ==============================

# def predecir_maquina(fila_dict):
#     """
#     Recibe un diccionario con los valores de una máquina.
#     Devuelve las predicciones de los 3 modelos.

#     Ejemplo de uso:
#     predecir_maquina({
#         'Type': 'M',
#         'Air temperature [K]': 298.1,
#         'Process temperature [K]': 308.6,
#         'Rotational speed [rpm]': 1550,
#         'Torque [Nm]': 42.8,
#         'Tool wear [min]': 82,
#         'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
#     })
#     """
#     # 1. Convertir a DataFrame
#     df = pd.DataFrame([fila_dict])

#     # 2. Codificar Type
#     df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

#     # 3. Limpiar nombres de columnas
#     df = limpiar(df)

#     # 4. Hacer predicciones
#     pred_rf = rf.predict(df)[0]
#     pred_xgb = xgb.predict(df)[0]
#     pred_lgbm = lgbm.predict(df)[0]

#     # 5. Devolver resultado bonito
#     print("PREDICCIONES DE LA MÁQUINA")
#     print("="*50)
#     print(f"Random Forest : {'FALLA' if pred_rf == 1 else 'NO FALLA'}")
#     print(f"XGBoost       : {'FALLA' if pred_xgb == 1 else 'NO FALLA'}")
#     print(f"LightGBM      : {'FALLA' if pred_lgbm == 1 else 'NO FALLA'}")
#     print("="*50)

#     # Votación mayoritaria
#     votos = [pred_rf, pred_xgb, pred_lgbm]
#     final = 1 if sum(votos) >= 2 else 0
#     print(f"DECISIÓN FINAL (2/3): {'FALLA' if final == 1 else 'NO FALLA'}")

#     return {
#         'Random Forest': pred_rf,
#         'XGBoost': pred_xgb,
#         'LightGBM': pred_lgbm,
#         'Final': final
#     }

# predecir_maquina({
#     'Type': 'M',
#     'Air temperature [K]': 298.1,
#     'Process temperature [K]': 308.6,
#     'Rotational speed [rpm]': 1550,
#     'Torque [Nm]': 42.8,
#     'Tool wear [min]': 82,
#     'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# predecir_maquina({
#     'Type': 'L',
#     'Air temperature [K]': 305.1,
#     'Process temperature [K]': 315.8,
#     'Rotational speed [rpm]': 1400,
#     'Torque [Nm]': 50.0,
#     'Tool wear [min]': 210,
#     'TWF': 1, 'HDF': 1, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# predecir_maquina({
#     'Type': 'H',
#     'Air temperature [K]': 301.0,
#     'Process temperature [K]': 316.5,   # ¡Muy alta!
#     'Rotational speed [rpm]': 1380,    # Lento
#     'Torque [Nm]': 62.3,               # ¡Muy alto!
#     'Tool wear [min]': 215,            # Herramienta gastada
#     'TWF': 1, 'HDF': 1, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# predecir_maquina({
#     'Type': 'M',
#     'Air temperature [K]': 298.2,
#     'Process temperature [K]': 308.7,
#     'Rotational speed [rpm]': 1521,
#     'Torque [Nm]': 41.2,
#     'Tool wear [min]': 73,
#     'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# predecir_maquina({
#     'Type': 'M',
#     'Air temperature [K]': 298.2,
#     'Process temperature [K]': 308.7,
#     'Rotational speed [rpm]': 1521,
#     'Torque [Nm]': 41.2,
#     'Tool wear [min]': 73,
#     'TWF': 0, 'HDF': 1, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# # ==============================
# # GUARDAR MODELOS EN .pkl
# # ==============================
# import joblib

# # Guardar cada modelo
# joblib.dump(rf, 'rf.pkl')
# joblib.dump(xgb, 'xgb.pkl')
# joblib.dump(lgbm, 'lgbm.pkl')

# print("Modelos guardados: rf.pkl, xgb.pkl, lgbm.pkl")

# # ==============================
# # PREPROCESADOR (para usar en producción)
# # ==============================

# preprocesador = {
#     'type_mapping': {'L': 0, 'M': 1, 'H': 2},
#     'columnas_esperadas': [
#         'Type', 'Air temperature K', 'Process temperature K',
#         'Rotational speed rpm', 'Torque Nm', 'Tool wear min',
#         'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
#     ]
# }

# joblib.dump(preprocesador, 'preprocesador.pkl')
# print("Preprocesador guardado: preprocesador.pkl")


import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from tabulate import tabulate
import joblib

# ==============================
# 1. CARGAR Y EXPLORAR DATOS
# ==============================
df = pd.read_csv('machine failure.csv')

print("="*50)
print("VALIDACIÓN DE VALORES NULOS")
print("="*50)
nulos = df.isnull().sum()
print(pd.DataFrame({
    'Columna': nulos.index,
    'Nulos': nulos.values,
    '% Nulos': (nulos / len(df)) * 100
}))

if nulos.sum() == 0:
    print("\n✅ NO HAY VALORES NULOS")
else:
    print("\n⚠️ Se encontraron valores nulos")

print("\n" + "="*50)
print("DISTRIBUCIÓN DE 'Machine failure'")
print("="*50)
print(df['Machine failure'].value_counts())
print(f"Porcentaje de fallos: {df['Machine failure'].mean()*100:.2f}%")

# ==============================
# 2. PREPROCESAMIENTO
# ==============================
df_ml = df.copy()
df_ml = df_ml.drop(['UDI', 'Product ID'], axis=1)

# Codificar 'Type' con LabelEncoder
le = LabelEncoder()
df_ml['Type'] = le.fit_transform(df_ml['Type'])

# Limpiar nombres de columnas
df_ml.columns = df_ml.columns.str.replace(r'[\[\]<>]', '', regex=True)
df_ml.columns = df_ml.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

X = df_ml.drop('Machine failure', axis=1)
y = df_ml['Machine failure']

print("\nColumnas finales:", list(X.columns))

# ==============================
# 3. DIVISIÓN CORRECTA (ANTES DE BALANCEAR)
# ==============================
# ✅ CORRECCIÓN: Dividir ANTES de balancear
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain original: {X_train.shape}")
print(f"Distribución train original:", Counter(y_train))

# ==============================
# 4. BALANCEO (SOLO EN TRAIN)
# ==============================
# ✅ CORRECCIÓN: Balancear SOLO train
smote = SMOTE(sampling_strategy=0.455, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

pipeline = Pipeline([
    ('smote', smote),
    ('undersample', undersample)
])

X_train_bal, y_train_bal = pipeline.fit_resample(X_train, y_train)

print(f"\nTrain balanceado: {X_train_bal.shape}")
print(f"Distribución train balanceado:", Counter(y_train_bal))
print(f"Test (sin tocar): {X_test.shape}")
print(f"Distribución test:", Counter(y_test))

# ==============================
# 5. ENTRENAMIENTO DE MODELOS
# ==============================
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
xgb = XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', n_jobs=-1, verbosity=0)
lgbm = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)

print("\n🔄 Entrenando modelos...")
rf.fit(X_train_bal, y_train_bal)
xgb.fit(X_train_bal, y_train_bal)
lgbm.fit(X_train_bal, y_train_bal)
print("✅ Modelos entrenados")

# ==============================
# 6. EVALUACIÓN (CON DATOS REALES DESBALANCEADOS)
# ==============================
modelos = [
    ('Random Forest', rf),
    ('XGBoost', xgb),
    ('LightGBM', lgbm)
]

resultados = []

for nombre, modelo in modelos:
    y_pred = modelo.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    resultados.append({
        'Modelo': nombre,
        'Accuracy': report['accuracy'],
        'Precision (1)': report['1']['precision'],
        'Recall (1)': report['1']['recall'],
        'F1-Score (1)': report['1']['f1-score']
    })

df_resultados = pd.DataFrame(resultados).sort_values('F1-Score (1)', ascending=False).round(4)

# Mostrar con porcentajes
df_print = df_resultados.copy()
df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']] *= 100
df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']] = \
    df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']].astype(str) + '%'

print("\n" + "═"*70)
print(" "*20 + "RESULTADOS EN TEST REAL (DESBALANCEADO)")
print("═"*70)
print(tabulate(df_print, headers='keys', tablefmt='pretty', showindex=False))
print("═"*70)

# ==============================
# 7. FUNCIÓN DE PREDICCIÓN
# ==============================
def predecir_maquina(fila_dict):
    """
    Predice si una máquina fallará usando los 3 modelos.
    
    Ejemplo:
    predecir_maquina({
        'Type': 'M',
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1550,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 82,
        'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
    })
    """
    df = pd.DataFrame([fila_dict])
    
    # Codificar Type usando el LabelEncoder entrenado
    df['Type'] = le.transform(df['Type'])
    
    # Limpiar nombres
    df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True).str.strip()
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Predicciones
    pred_rf = rf.predict(df)[0]
    pred_xgb = xgb.predict(df)[0]
    pred_lgbm = lgbm.predict(df)[0]
    
    print("\nPREDICCIONES DE LA MÁQUINA")
    print("="*50)
    print(f"Random Forest : {'FALLA ❌' if pred_rf == 1 else 'NO FALLA ✅'}")
    print(f"XGBoost       : {'FALLA ❌' if pred_xgb == 1 else 'NO FALLA ✅'}")
    print(f"LightGBM      : {'FALLA ❌' if pred_lgbm == 1 else 'NO FALLA ✅'}")
    print("="*50)
    
    # Votación mayoritaria
    votos = [pred_rf, pred_xgb, pred_lgbm]
    final = 1 if sum(votos) >= 2 else 0
    print(f"DECISIÓN FINAL (2/3): {'FALLA ❌' if final == 1 else 'NO FALLA ✅'}\n")
    
    return {
        'Random Forest': pred_rf,
        'XGBoost': pred_xgb,
        'LightGBM': pred_lgbm,
        'Final': final
    }

# ==============================
# 8. PRUEBAS
# ==============================
print("\n🧪 PRUEBA 1: Máquina normal")
predecir_maquina({
    'Type': 'M',
    'Air temperature [K]': 298.1,
    'Process temperature [K]': 308.6,
    'Rotational speed [rpm]': 1550,
    'Torque [Nm]': 42.8,
    'Tool wear [min]': 82,
    'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
})

print("\n🧪 PRUEBA 2: Máquina con problemas")
predecir_maquina({
    'Type': 'H',
    'Air temperature [K]': 301.0,
    'Process temperature [K]': 316.5,
    'Rotational speed [rpm]': 1380,
    'Torque [Nm]': 62.3,
    'Tool wear [min]': 215,
    'TWF': 1, 'HDF': 1, 'PWF': 0, 'OSF': 0, 'RNF': 0
})

print("\n🧪 PRUEBA 3: Máquina casi nueva")
predecir_maquina({
    'Type': 'M',
    'Air temperature [K]': 298.2,
    'Process temperature [K]': 308.7,
    'Rotational speed [rpm]': 1521,
    'Torque [Nm]': 41.2,
    'Tool wear [min]': 73,
    'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
})

print("\n🧪 PRUEBA 4: Máquina con historial de fallos")
predecir_maquina({
    'Type': 'H',
    'Air temperature [K]': 300.5,
    'Process temperature [K]': 315.0,
    'Rotational speed [rpm]': 1400,
    'Torque [Nm]': 60.0,
    'Tool wear [min]': 200,
    'TWF': 1, 'HDF': 1, 'PWF': 1, 'OSF': 0, 'RNF': 0
})

# ==============================
# 9. GUARDAR MODELOS Y PREPROCESADOR
# ==============================
# joblib.dump(rf, 'rf.pkl')
# joblib.dump(xgb, 'xgb.pkl')
# joblib.dump(lgbm, 'lgbm.pkl')
# joblib.dump(le, 'label_encoder.pkl')  # ✅ Guardar el encoder

# preprocesador = {
#     'type_mapping': dict(zip(le.classes_, le.transform(le.classes_))),
#     'columnas_esperadas': list(X.columns)
# }
# joblib.dump(preprocesador, 'preprocesador.pkl')

# print("\n✅ Modelos guardados:")
# print("   - rf.pkl")
# print("   - xgb.pkl")
# print("   - lgbm.pkl")
# print("   - label_encoder.pkl")
# print("   - preprocesador.pkl")


# import pandas as pd
# import numpy as np
# import seaborn as sns
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from xgboost import XGBClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression   # 🔵 LOGISTIC REGRESSION
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from collections import Counter
# from tabulate import tabulate
# import joblib

# # ==============================
# # 1. CARGAR Y EXPLORAR DATOS
# # ==============================
# df = pd.read_csv('machine failure.csv')

# print("="*50)
# print("VALIDACIÓN DE VALORES NULOS")
# print("="*50)
# nulos = df.isnull().sum()
# print(pd.DataFrame({
#     'Columna': nulos.index,
#     'Nulos': nulos.values,
#     '% Nulos': (nulos / len(df)) * 100
# }))

# print("\n" + "="*50)
# print("DISTRIBUCIÓN DE 'Machine failure'")
# print("="*50)
# print(df['Machine failure'].value_counts())
# print(f"Porcentaje de fallos: {df['Machine failure'].mean()*100:.2f}%")

# # ==============================
# # 2. PREPROCESAMIENTO
# # ==============================
# df_ml = df.copy()
# df_ml = df_ml.drop(['UDI', 'Product ID'], axis=1)

# le = LabelEncoder()
# df_ml['Type'] = le.fit_transform(df_ml['Type'])

# df_ml.columns = df_ml.columns.str.replace(r'[\[\]<>]', '', regex=True)
# df_ml.columns = df_ml.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

# X = df_ml.drop('Machine failure', axis=1)
# y = df_ml['Machine failure']

# # ==============================
# # 3. DIVISIÓN
# # ==============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # ==============================
# # 4. BALANCEO (SOLO TRAIN)
# # ==============================
# smote = SMOTE(sampling_strategy=0.4767, random_state=42)
# undersample = RandomUnderSampler(sampling_strategy=0.490105, random_state=42)

# pipeline = Pipeline([
#     ('smote', smote),
#     ('undersample', undersample)
# ])

# X_train_bal, y_train_bal = pipeline.fit_resample(X_train, y_train)

# # ==============================
# # 5. ENTRENAMIENTO DE MODELOS
# # ==============================
# rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
# xgb = XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0)
# lgbm = lgb.LGBMClassifier(n_estimators=200, random_state=42)

# # 🔵 LOGISTIC REGRESSION (NUEVO MODELO)
# logreg = LogisticRegression(
#     max_iter=2000,
#     class_weight='balanced',
#     n_jobs=-1
# )

# print("\n🔄 Entrenando modelos...")
# rf.fit(X_train_bal, y_train_bal)
# xgb.fit(X_train_bal, y_train_bal)
# lgbm.fit(X_train_bal, y_train_bal)
# logreg.fit(X_train_bal, y_train_bal)  # 🔵 Entrenamiento LR
# print("✅ Modelos entrenados")

# # ==============================
# # 6. EVALUACIÓN
# # ==============================
# modelos = [
#     ('Random Forest', rf),
#     ('XGBoost', xgb),
#     ('LightGBM', lgbm),
#     ('Logistic Regression', logreg)  # 🔵 Evaluación LR
# ]

# resultados = []

# for nombre, modelo in modelos:
#     y_pred = modelo.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     resultados.append({
#         'Modelo': nombre,
#         'Accuracy': report['accuracy'],
#         'Precision (1)': report['1']['precision'],
#         'Recall (1)': report['1']['recall'],
#         'F1-Score (1)': report['1']['f1-score']
#     })

# df_resultados = pd.DataFrame(resultados).sort_values('F1-Score (1)', ascending=False).round(4)

# df_print = df_resultados.copy()
# df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']] *= 100
# df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']] = \
# df_print[['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']].astype(str) + '%'

# print("\n" + "═"*70)
# print(" "*15 + "RESULTADOS EN TEST REAL (DESBALANCEADO)")
# print("═"*70)
# print(tabulate(df_print, headers='keys', tablefmt='pretty', showindex=False))
# print("═"*70)

# # ==============================
# # 7. FUNCIÓN DE PREDICCIÓN
# # ==============================
# def predecir_maquina(fila_dict):

#     df = pd.DataFrame([fila_dict])
#     df['Type'] = le.transform(df['Type'])

#     df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True).str.strip()
#     df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

#     pred_rf = rf.predict(df)[0]
#     pred_xgb = xgb.predict(df)[0]
#     pred_lgbm = lgbm.predict(df)[0]
#     pred_lr = logreg.predict(df)[0]  # 🔵 Predicción LR

#     print("\nPREDICCIONES DE LA MÁQUINA")
#     print("="*50)
#     print(f"Random Forest       : {'FALLA ❌' if pred_rf else 'NO FALLA ✅'}")
#     print(f"XGBoost             : {'FALLA ❌' if pred_xgb else 'NO FALLA ✅'}")
#     print(f"LightGBM            : {'FALLA ❌' if pred_lgbm else 'NO FALLA ✅'}")
#     print(f"Logistic Regression : {'FALLA ❌' if pred_lr else 'NO FALLA ✅'}")
#     print("="*50)

#     votos = [pred_rf, pred_xgb, pred_lgbm, pred_lr]
#     final = 1 if sum(votos) >= 2 else 0

#     print(f"DECISIÓN FINAL (2/4): {'FALLA ❌' if final == 1 else 'NO FALLA ✅'}\n")

#     return {
#         'Random Forest': pred_rf,
#         'XGBoost': pred_xgb,
#         'LightGBM': pred_lgbm,
#         'Logistic Regression': pred_lr,
#         'Final': final
#     }

# # ==============================
# # 8. PRUEBAS
# # ==============================

# print("\n🧪 PRUEBA 1: Máquina normal")
# predecir_maquina({
#     'Type': 'M',
#     'Air temperature [K]': 298.1,
#     'Process temperature [K]': 308.6,
#     'Rotational speed [rpm]': 1550,
#     'Torque [Nm]': 42.8,
#     'Tool wear [min]': 82,
#     'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# print("\n🧪 PRUEBA 2: Máquina con problemas")
# predecir_maquina({
#     'Type': 'H',
#     'Air temperature [K]': 301.0,
#     'Process temperature [K]': 316.5,
#     'Rotational speed [rpm]': 1380,
#     'Torque [Nm]': 62.3,
#     'Tool wear [min]': 215,
#     'TWF': 1, 'HDF': 1, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# print ("\n🧪 PRUEBA 3: Máquina borderline")
# predecir_maquina({
#     'Type': 'M',
#     'Air temperature [K]': 300.0,
#     'Process temperature [K]': 312.0,
#     'Rotational speed [rpm]': 1450,
#     'Torque [Nm]':  55.0,
#     'Tool wear [min]': 150,
#     'TWF': 1, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# print ("\n🧪 PRUEBA 4: Máquina óptima")
# predecir_maquina({
#     'Type': 'L',
#     'Air temperature [K]': 295.0,
#     'Process temperature [K]': 305.0,
#     'Rotational speed [rpm]': 1600,
#     'Torque [Nm]': 40.0,
#     'Tool wear [min]': 50,
#     'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })

# print ("\n🧪 PRUEBA 5: Máquina con desgaste leve")
# predecir_maquina({
#     'Type': 'M',
#     'Air temperature [K]': 299.0,
#     'Process temperature [K]': 319.0,
#     'Rotational speed [rpm]': 1500,
#     'Torque [Nm]': 77.0,
#     'Tool wear [min]': 90,
#     'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0
# })





# # # ==============================
# # # 9. GUARDAR MODELOS Y PREPROCESADOR
# # # ==============================

# # joblib.dump(rf, 'rf.pkl')
# # joblib.dump(xgb, 'xgb.pkl')
# # joblib.dump(lgbm, 'lgbm.pkl')
# # joblib.dump(logreg, 'logreg.pkl')  # 🔵 Guardar LR

# # joblib.dump(le, 'label_encoder.pkl')
# # preprocesador = {
# #     'type_mapping': dict(zip(le.classes_, le.transform(le.classes_))),
# #     'columnas_esperadas': list(X.columns)
# # }
# # joblib.dump(preprocesador, 'preprocesador.pkl')
# # print("\n✅ Modelos guardados:")
# # print("   - rf.pkl")
# # print("   - xgb.pkl")
# # print("   - lgbm.pkl")
# # print("   - logreg.pkl")
# # print("   - label_encoder.pkl")
# # print("   - preprocesador.pkl")