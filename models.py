# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import TINYINT
import enum
from datetime import datetime

Base = declarative_base()

class TipoMaquinaEnum(enum.Enum):
    L = "L"
    M = "M"
    H = "H"

# ✅ CORRECCIÓN: Nombres coinciden con MySQL
class ModeloEnum(enum.Enum):
    Random_Forest = "Random_Forest"
    XGBoost = "XGBoost"
    LightGBM = "LightGBM"

# ✅ CORRECCIÓN: Nombres coinciden con MySQL
class DecisionEnum(enum.Enum):
    FALLA = "FALLA"
    NO_FALLA = "NO_FALLA"  # ⚠️ Con espacio

# ==============================
# TABLA: usuarios
# ==============================
class Usuario(Base):
    __tablename__ = "usuarios"
    
    id = Column(Integer, primary_key=True, index=True)
    telefono = Column(String(20), unique=True, nullable=False)
    identificacion = Column(String(10), unique=True, nullable=False)
    nombre = Column(String(100))
    apellido = Column(String(100))
    usuario = Column(String(100), unique=True, nullable=False)
    contraseña = Column(String(100))
    fecha_registro = Column(DateTime, default=datetime.utcnow)

# ==============================
# TABLA: maquinas
# ==============================
class Maquina(Base):
    __tablename__ = "maquinas"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True)
    type = Column(Enum(TipoMaquinaEnum), nullable=False)
    air_temperature = Column(Float, nullable=False)
    process_temperature = Column(Float, nullable=False)
    rotational_speed = Column(Integer, nullable=False)
    torque = Column(Float, nullable=False)
    tool_wear = Column(Integer, nullable=False)
    twf = Column(TINYINT(1), default=0)
    hdf = Column(TINYINT(1), default=0)
    pwf = Column(TINYINT(1), default=0)
    osf = Column(TINYINT(1), default=0)
    rnf = Column(TINYINT(1), default=0)
    #machine_failure = Column(TINYINT(1), nullable=True)
    fecha_registro = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_maquina_fecha', 'fecha_registro'),
        Index('idx_maquina_type', 'type'),
    )

# models.py (solo cambios en Prediccion)
class Prediccion(Base):
    __tablename__ = "predicciones"
    
    id = Column(Integer, primary_key=True, index=True)
    maquina_id = Column(Integer, ForeignKey("maquinas.id", ondelete="CASCADE"), nullable=False)
    modelo = Column(Enum(ModeloEnum), nullable=False)
    prediccion = Column(TINYINT(1), nullable=False)
    decision_final = Column(Enum(DecisionEnum), nullable=False)
    fecha_prediccion = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        #UniqueConstraint('maquina_id', 'modelo', name='unique_prediccion'),
        Index('idx_prediccion_maquina', 'maquina_id'),
    )