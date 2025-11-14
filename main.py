from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
import os

from database import SessionLocal, engine
from models import Base, Usuario, Maquina, Prediccion
from schemas import MaquinaInput, LoginRequest, LoginResponse, PrediccionResponse, RegistroUsuarioRequest
from services.ml_service import MLService
from services.auth_service import AuthService
from services.report_service import ReportService

# Crear tablas en BD (solo primera vez)
# Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sistema de Predicción de Fallas - API",
    version="2.0.0",
    description="API con autenticación y predicción de fallas en máquinas"
)

# ==============================
# DEPENDENCIAS
# ==============================
def get_db():
    """Generador de sesiones de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================
# SERVICIOS (se inicializan una vez)
# ==============================
ml_service = MLService(model_path="saved_models")
auth_service = AuthService()
report_service = ReportService()

# ==============================
# ENDPOINTS DE AUTENTICACIÓN
# ==============================
@app.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """
    Endpoint de login.
    Retorna información del usuario si las credenciales son correctas.
    """
    usuario = auth_service.authenticate(
        db=db,
        username=credentials.usuario,
        password=credentials.contraseña
    )
    
    if not usuario:
        raise HTTPException(
            status_code=401,
            detail="Usuario o contraseña incorrectos"
        )
    
    return LoginResponse(
        message="Login exitoso",
        usuario_id=usuario.id,
        nombre_completo=f"{usuario.nombre} {usuario.apellido}",
        usuario=usuario.usuario
    )

@app.post("/registro")
async def registrar_usuario(
    datos: RegistroUsuarioRequest,  # ✅ Ahora acepta JSON en el body
    db: Session = Depends(get_db)
):
    """
    Registra un nuevo usuario en el sistema.
    Envía los datos en formato JSON en el body.
    """
    try:
        nuevo_usuario = auth_service.crear_usuario(
            db=db,
            telefono=datos.telefono,
            identificacion=datos.identificacion,
            nombre=datos.nombre,
            apellido=datos.apellido,
            usuario=datos.usuario,
            contraseña=datos.contraseña
        )
        
        return {
            "message": "Usuario creado exitosamente",
            "usuario_id": nuevo_usuario.id,
            "usuario": nuevo_usuario.usuario
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==============================
# ENDPOINTS DE PREDICCIÓN
# ==============================
@app.post("/predecir", response_model=PrediccionResponse)
async def predecir_fallo(
    maquina_input: MaquinaInput,
    db: Session = Depends(get_db)
):
    """
    Predice si una máquina fallará.
    Requiere el ID del usuario que realiza la predicción.
    """
    # Verificar que el usuario existe
    usuario = db.query(Usuario).filter(Usuario.id == maquina_input.usuario_id).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    try:
        # Realizar predicción
        resultado = ml_service.predecir(maquina_input)
        
        # Guardar en base de datos
        maquina_id = ml_service.guardar_prediccion(
            db=db,
            maquina_input=maquina_input,
            predicciones=resultado['predicciones'],
            decision_final=resultado['decision_final'],
            usuario_id=maquina_input.usuario_id  # ✅ Vincular con usuario
        )
        
        return PrediccionResponse(
            maquina_id=maquina_id,
            usuario_id=maquina_input.usuario_id,
            usuario_nombre=f"{usuario.nombre} {usuario.apellido}",
            predicciones=resultado['predicciones'],
            votos_falla=resultado['votos_falla'],
            decision_final=resultado['decision_final']
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ==============================
# ENDPOINTS DE REPORTES
# ==============================
@app.get("/reporte/tipo")
async def reporte_tipo(db: Session = Depends(get_db)):
    """Genera gráfico de fallos por tipo de máquina"""
    return report_service.generar_reporte_tipo(db)

@app.get("/reporte/mes")
async def reporte_mes(db: Session = Depends(get_db)):
    """Genera gráfico de evolución mensual de fallos"""
    return report_service.generar_reporte_mensual(db)

@app.get("/reporte/riesgo")
async def reporte_riesgo(db: Session = Depends(get_db)):
    """Retorna las últimas 5 máquinas con predicción de fallo"""
    return report_service.obtener_ultimos_fallos(db)

@app.get("/reporte/usuario/{usuario_id}")
async def reporte_usuario(usuario_id: int, db: Session = Depends(get_db)):
    """
    Retorna estadísticas de las predicciones realizadas por un usuario.
    """
    return report_service.obtener_reporte_usuario(db, usuario_id)

# ==============================
# ENDPOINT DE SALUD
# ==============================
@app.get("/")
async def root():
    """Verifica que la API esté funcionando"""
    return {
        "status": "online",
        "version": "2.0.0",
        "modelos_cargados": ml_service.modelos_disponibles,
        "tipos_permitidos": ml_service.tipos_permitidos
    }

@app.get("/health")
async def health_check():
    """Health check para monitoreo"""
    return {"status": "healthy", "service": "ML Prediction API"}