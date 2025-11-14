# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.responses import JSONResponse, FileResponse
# import joblib
# import pandas as pd
# import os
# from schemas import MaquinaInput
# from database import SessionLocal
# from models import Maquina, Prediccion, ModeloEnum, DecisionEnum
# from sqlalchemy.orm import Session
# import matplotlib.pyplot as plt
# import seaborn as sns
# #from datetime import datetime

# app = FastAPI(title="Predicción de Fallo en Máquinas", version="1.0.0")

# # CARGAR MODELOS
# MODELOS_PATH = "saved_models"
# rf = joblib.load(os.path.join(MODELOS_PATH, "rf.pkl"))
# xgb = joblib.load(os.path.join(MODELOS_PATH, "xgb.pkl"))
# lgbm = joblib.load(os.path.join(MODELOS_PATH, "lgbm.pkl"))
# prep = joblib.load(os.path.join(MODELOS_PATH, "preprocesador.pkl"))

# # DEPENDENCIA BD
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # PREPROCESAMIENTO
# def preparar_datos(data: MaquinaInput) -> pd.DataFrame:
#     fila = data.dict()
#     df = pd.DataFrame([fila])
#     rename_map = {
#         'air_temperature': 'Air temperature K',
#         'process_temperature': 'Process temperature K',
#         'rotational_speed': 'Rotational speed rpm',
#         'torque': 'Torque Nm',
#         'tool_wear': 'Tool wear min',
#         'twf': 'TWF', 'hdf': 'HDF', 'pwf': 'PWF', 'osf': 'OSF', 'rnf': 'RNF'
#     }
#     df = df.rename(columns=rename_map)
#     df['Type'] = df['Type'].map(prep['type_mapping'])
#     df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True).str.strip()
#     df = df[prep['columnas_esperadas']]
#     return df

# # main.py (solo cambios en /predecir)
# @app.post("/predecir")
# async def predecir_fallo(maquina: MaquinaInput, db: Session = Depends(get_db)):
#     try:
#         X = preparar_datos(maquina)
        
#         # Predicciones SIN probabilidad
#         pred_rf = int(rf.predict(X)[0])
#         pred_xgb = int(xgb.predict(X)[0])
#         pred_lgbm = int(lgbm.predict(X)[0])
        
#         final = 1 if sum([pred_rf, pred_xgb, pred_lgbm]) >= 2 else 0
#         decision = "FALLA" if final else "NO FALLA"
        
#         # GUARDAR EN BD
#         nueva_maquina = Maquina(
#             type=maquina.Type,
#             air_temperature=maquina.air_temperature,
#             process_temperature=maquina.process_temperature,
#             rotational_speed=maquina.rotational_speed,
#             torque=maquina.torque,
#             tool_wear=maquina.tool_wear,
#             twf=maquina.twf,
#             hdf=maquina.hdf,
#             pwf=maquina.pwf,
#             osf=maquina.osf,
#             rnf=maquina.rnf,
#             machine_failure=None,
#         )
#         db.add(nueva_maquina)
#         db.commit()
#         db.refresh(nueva_maquina)

#         # Guardar 3 predicciones (solo 0/1)
#         for modelo_name, pred in [
#             ("Random Forest", pred_rf),
#             ("XGBoost", pred_xgb),
#             ("LightGBM", pred_lgbm)
#         ]:
#             db.add(Prediccion(
#                 maquina_id=nueva_maquina.id,
#                 modelo=ModeloEnum.rf if modelo_name == "Random Forest" else 
#                         ModeloEnum.xgb if modelo_name == "XGBoost" else ModeloEnum.lgbm,
#                 prediccion=pred,
#                 decision_final=DecisionEnum.falla if pred else DecisionEnum.no_falla
#             ))
#         db.commit()

#         return {
#             "maquina_id": nueva_maquina.id,
#             "predicciones": {
#                 "Random Forest": pred_rf,
#                 "XGBoost": pred_xgb,
#                 "LightGBM": pred_lgbm
#             },
#             "DECISION_FINAL": decision
#         }
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))

# # REPORTE 1: POR TIPO
# @app.get("/reporte/tipo")
# async def reporte_tipo(db: Session = Depends(get_db)):
#     query = """
#     SELECT m.type, COUNT(*) as total, SUM(p.prediccion) as fallos
#     FROM maquinas m
#     JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
#     GROUP BY m.type
#     """
#     df = pd.read_sql(query, db.bind)
#     if df.empty:
#         raise HTTPException(404, "No hay datos")
    
#     plt.figure(figsize=(6,4))
#     sns.barplot(x='type', y='fallos', data=df)
#     plt.title("Fallos por Tipo de Máquina")
#     plt.savefig("reporte_tipo.png")
#     plt.close()
#     return FileResponse("reporte_tipo.png")

# # REPORTE 2: POR MES
# @app.get("/reporte/mes")
# async def reporte_mes(db: Session = Depends(get_db)):
#     query = """
#     SELECT DATE_FORMAT(m.fecha_registro, '%Y-%m') as mes, COUNT(*) as registros, SUM(p.prediccion) as fallos
#     FROM maquinas m
#     JOIN predicciones p ON m.id = p.maquina_id
#     GROUP BY mes ORDER BY mes DESC
#     """
#     df = pd.read_sql(query, db.bind)
#     if df.empty:
#         raise HTTPException(404, "No hay datos")
    
#     plt.figure(figsize=(8,5))
#     sns.lineplot(x='mes', y='fallos', data=df, marker='o')
#     plt.title("Fallos por Mes")
#     plt.xticks(rotation=45)
#     plt.savefig("reporte_mes.png")
#     plt.close()
#     return FileResponse("reporte_mes.png")

# # REPORTE 3: ÚLTIMOS 5 FALLOS
# @app.get("/reporte/riesgo")
# async def reporte_riesgo(db: Session = Depends(get_db)):
#     query = """
#     SELECT m.id, m.type, m.fecha_registro
#     FROM maquinas m
#     JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
#     WHERE p.prediccion = 1
#     ORDER BY m.fecha_registro DESC LIMIT 5
#     """
#     df = pd.read_sql(query, db.bind)
#     if df.empty:
#         raise HTTPException(404, "No hay fallos predichos")

#     df.to_dict(orient="records")
#     return JSONResponse(content={"ultimos_fallos": df.to_dict(orient="records")})


# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
import joblib
import pandas as pd
import os
from schemas import MaquinaInput
from database import SessionLocal
from models import Maquina, Prediccion, ModeloEnum, DecisionEnum
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI(title="Predicción de Fallo en Máquinas", version="1.0.0")

# ==============================
# CARGAR MODELOS Y PREPROCESADOR
# ==============================
MODELOS_PATH = "saved_models"
rf = joblib.load(os.path.join(MODELOS_PATH, "rf.pkl"))
xgb = joblib.load(os.path.join(MODELOS_PATH, "xgb.pkl"))
lgbm = joblib.load(os.path.join(MODELOS_PATH, "lgbm.pkl"))
le = joblib.load(os.path.join(MODELOS_PATH, "label_encoder.pkl"))  # ✅ NUEVO
prep = joblib.load(os.path.join(MODELOS_PATH, "preprocesador.pkl"))

# DEPENDENCIA BD
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================
# PREPROCESAMIENTO CORREGIDO
# ==============================
def preparar_datos(data: MaquinaInput) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada usando el LabelEncoder guardado.
    Ahora maneja valores desconocidos de 'Type' de forma segura.
    """
    fila = data.dict()
    df = pd.DataFrame([fila])
    
    # Renombrar columnas
    rename_map = {
        'air_temperature': 'Air temperature K',
        'process_temperature': 'Process temperature K',
        'rotational_speed': 'Rotational speed rpm',
        'torque': 'Torque Nm',
        'tool_wear': 'Tool wear min',
        'twf': 'TWF', 'hdf': 'HDF', 'pwf': 'PWF', 'osf': 'OSF', 'rnf': 'RNF'
    }
    df = df.rename(columns=rename_map)
    
    # ✅ CAMBIO CRÍTICO: Usar LabelEncoder con manejo de errores
    try:
        df['Type'] = le.transform(df['Type'])
    except ValueError as e:
        # Si llega un valor desconocido (ej: 'X' cuando solo conoce L, M, H)
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de máquina inválido: '{data.Type}'. Valores permitidos: {list(le.classes_)}"
        )
    
    # Limpiar nombres de columnas
    df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True).str.strip()
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Asegurar orden correcto de columnas
    df = df[prep['columnas_esperadas']]
    
    return df

# ==============================
# ENDPOINT DE PREDICCIÓN
# ==============================
@app.post("/predecir")
async def predecir_fallo(maquina: MaquinaInput, db: Session = Depends(get_db)):
    """
    Predice si una máquina fallará usando 3 modelos de ML.
    Devuelve la decisión final por votación mayoritaria (2/3).
    """
    try:
        X = preparar_datos(maquina)
        
        # Predicciones binarias (0 o 1)
        pred_rf = int(rf.predict(X)[0])
        pred_xgb = int(xgb.predict(X)[0])
        pred_lgbm = int(lgbm.predict(X)[0])
        
        # Votación mayoritaria
        votos = [pred_rf, pred_xgb, pred_lgbm]
        final = 1 if sum(votos) >= 2 else 0
        decision = "FALLA" if final else "NO FALLA"
        
        # GUARDAR EN BD - Máquina
        nueva_maquina = Maquina(
            type=maquina.Type,
            air_temperature=maquina.air_temperature,
            process_temperature=maquina.process_temperature,
            rotational_speed=maquina.rotational_speed,
            torque=maquina.torque,
            tool_wear=maquina.tool_wear,
            twf=maquina.twf,
            hdf=maquina.hdf,
            pwf=maquina.pwf,
            osf=maquina.osf,
            rnf=maquina.rnf,
        )
        db.add(nueva_maquina)
        db.commit()
        db.refresh(nueva_maquina)

        # GUARDAR EN BD - Predicciones
        predicciones_modelo = [
            ("Random Forest", pred_rf, ModeloEnum.rf),
            ("XGBoost", pred_xgb, ModeloEnum.xgb),
            ("LightGBM", pred_lgbm, ModeloEnum.lgbm)
        ]
        
        for nombre, pred, enum_modelo in predicciones_modelo:
            db.add(Prediccion(
                maquina_id=nueva_maquina.id,
                modelo=enum_modelo,
                prediccion=pred,
                decision_final=DecisionEnum.falla if pred else DecisionEnum.no_falla
            ))
        db.commit()

        return {
            "maquina_id": nueva_maquina.id,
            "predicciones": {
                "Random Forest": pred_rf,
                "XGBoost": pred_xgb,
                "LightGBM": pred_lgbm
            },
            "votos_falla": sum(votos),
            "DECISION_FINAL": decision
        }
        
    except HTTPException:
        raise  # Re-lanzar errores de validación
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ==============================
# REPORTE 1: FALLOS POR TIPO
# ==============================
@app.get("/reporte/tipo")
async def reporte_tipo(db: Session = Depends(get_db)):
    """Genera gráfico de fallos por tipo de máquina (L, M, H)"""
    query = """
    SELECT m.type, COUNT(*) as total, SUM(p.prediccion) as fallos
    FROM maquinas m
    JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
    GROUP BY m.type
    """
    df = pd.read_sql(query, db.bind)
    
    if df.empty:
        raise HTTPException(404, "No hay datos para generar el reporte")
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='type', y='fallos', data=df, palette='Reds')
    plt.title("Fallos Predichos por Tipo de Máquina", fontsize=14, fontweight='bold')
    plt.xlabel("Tipo de Máquina")
    plt.ylabel("Cantidad de Fallos")
    plt.tight_layout()
    plt.savefig("reporte_tipo.png", dpi=150)
    plt.close()
    
    return FileResponse("reporte_tipo.png", media_type="image/png")


@app.post("/login")
async def login(username: str, password: str, db: Session = Depends(get_db)):
    """
    Login simple SIN JWT:
    - Valida usuario y contraseña en BD
    - Retorna el ID del usuario si es correcto
    """

    usuario = db.query(usuario).filter(
        usuario.username == username,
        usuario.password == password
    ).first()

    if not usuario:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    return {
        "message": "Login exitoso",
        "usuario_id": usuario.id,
        "username": usuario.username
    }

# ==============================
# REPORTE 2: TENDENCIA MENSUAL
# ==============================
@app.get("/reporte/mes")
async def reporte_mes(db: Session = Depends(get_db)):
    """Genera gráfico de evolución mensual de fallos"""
    query = """
    SELECT DATE_FORMAT(m.fecha_registro, '%Y-%m') as mes, 
           COUNT(*) as registros, 
           SUM(p.prediccion) as fallos
    FROM maquinas m
    JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
    GROUP BY mes 
    ORDER BY mes DESC
    """
    df = pd.read_sql(query, db.bind)
    
    if df.empty:
        raise HTTPException(404, "No hay datos para generar el reporte")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='mes', y='fallos', data=df, marker='o', linewidth=2, color='crimson')
    plt.title("Evolución Mensual de Fallos Predichos", fontsize=14, fontweight='bold')
    plt.xlabel("Mes")
    plt.ylabel("Cantidad de Fallos")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reporte_mes.png", dpi=150)
    plt.close()
    
    return FileResponse("reporte_mes.png", media_type="image/png")

# ==============================
# REPORTE 3: ÚLTIMOS FALLOS
# ==============================
@app.get("/reporte/riesgo")
async def reporte_riesgo(db: Session = Depends(get_db)):
    """Devuelve las últimas 5 máquinas con predicción de fallo"""
    query = """
    SELECT m.id, m.type, m.torque, m.tool_wear, m.fecha_registro
    FROM maquinas m
    JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
    WHERE p.prediccion = 1
    ORDER BY m.fecha_registro DESC 
    LIMIT 5
    """
    df = pd.read_sql(query, db.bind)
    
    if df.empty:
        raise HTTPException(404, "No hay fallos predichos en el sistema")
    
    # Convertir timestamp a string para JSON
    df['fecha_registro'] = df['fecha_registro'].astype(str)
    
    return JSONResponse(content={
        "ultimos_fallos": df.to_dict(orient="records"),
        "total": len(df)
    })

# ==============================
# ENDPOINT DE SALUD
# ==============================
@app.get("/")
async def root():
    """Verifica que la API esté funcionando"""
    return {
        "status": "online",
        "modelos_cargados": ["Random Forest", "XGBoost", "LightGBM"],
        "tipos_permitidos": list(le.classes_)  # ✅ Muestra L, M, H
    }