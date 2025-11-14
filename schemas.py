# schemas.py
from pydantic import BaseModel
from typing import Literal, Optional

class MaquinaInput(BaseModel):
    Type: Literal['L', 'M', 'H']
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    
    # OPCIONALES (default = 0)
    twf: Optional[int] = 0
    hdf: Optional[int] = 0
    pwf: Optional[int] = 0
    osf: Optional[int] = 0
    rnf: Optional[int] = 0

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