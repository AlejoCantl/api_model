# schemas.py
# from pydantic import BaseModel
# from typing import Literal, Optional

# class MaquinaInput(BaseModel):
#     Type: Literal['L', 'M', 'H']
#     air_temperature: float
#     process_temperature: float
#     rotational_speed: int
#     torque: float
#     tool_wear: int
    
#     # OPCIONALES (default = 0)
#     twf: Optional[int] = 0
#     hdf: Optional[int] = 0
#     pwf: Optional[int] = 0
#     osf: Optional[int] = 0
#     rnf: Optional[int] = 0

#     class Config:
#         schema_extra = {
#             "example": {
#                 "Type": "M",
#                 "air_temperature": 298.1,
#                 "process_temperature": 308.6,
#                 "rotational_speed": 1550,
#                 "torque": 42.8,
#                 "tool_wear": 82
#             }
#         }

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict

class MaquinaInput(BaseModel):
    usuario_id: Optional[int] = None  # ✅ Nuevo campo para el ID del usuario
    Type: Literal['L', 'M', 'H']
    air_temperature: float = Field(..., ge=290, le=310)
    process_temperature: float = Field(..., ge=300, le=320)
    rotational_speed: int = Field(..., ge=1000, le=3000)
    torque: float = Field(..., ge=0, le=100)
    tool_wear: int = Field(..., ge=0, le=300)
    
    twf: Optional[int] = Field(0, ge=0, le=1)
    hdf: Optional[int] = Field(0, ge=0, le=1)
    pwf: Optional[int] = Field(0, ge=0, le=1)
    osf: Optional[int] = Field(0, ge=0, le=1)
    rnf: Optional[int] = Field(0, ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "Type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1550,
                "torque": 42.8,
                "tool_wear": 82
            }
        }

class LoginRequest(BaseModel):
    usuario: str
    contraseña: str

class LoginResponse(BaseModel):
    message: str
    usuario_id: int
    nombre_completo: str
    usuario: str

class PrediccionResponse(BaseModel):
    maquina_id: int
    usuario_id: int
    usuario_nombre: str
    predicciones: Dict[str, int]
    votos_falla: int
    decision_final: str

class RegistroUsuarioRequest(BaseModel):
    telefono: str = Field(..., min_length=10, max_length=20)
    identificacion: str = Field(..., min_length=6, max_length=20)
    nombre: str = Field(..., min_length=2, max_length=100)
    apellido: str = Field(..., min_length=2, max_length=100)
    usuario: str = Field(..., min_length=4, max_length=50)
    contraseña: str = Field(..., min_length=4, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "telefono": "3001234567",
                "identificacion": "1001234567",
                "nombre": "Carlos",
                "apellido": "Rodríguez",
                "usuario": "carlosr",
                "contraseña": "carlos123"
            }
        }