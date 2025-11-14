import joblib
import pandas as pd
import os
from fastapi import HTTPException
from models import Maquina, Prediccion, ModeloEnum, DecisionEnum
from schemas import MaquinaInput

class MLService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._cargar_modelos()
    
    def _cargar_modelos(self):
        try:
            self.rf = joblib.load(os.path.join(self.model_path, "rf.pkl"))
            self.xgb = joblib.load(os.path.join(self.model_path, "xgb.pkl"))
            self.lgbm = joblib.load(os.path.join(self.model_path, "lgbm.pkl"))
            self.le = joblib.load(os.path.join(self.model_path, "label_encoder.pkl"))
            self.prep = joblib.load(os.path.join(self.model_path, "preprocesador.pkl"))
            
            self.modelos = {
                "Random Forest": self.rf,
                "XGBoost": self.xgb,
                "LightGBM": self.lgbm
            }
            
            print("✅ Modelos cargados correctamente")
        except Exception as e:
            print(f"❌ Error al cargar modelos: {e}")
            raise
    
    @property
    def modelos_disponibles(self):
        return list(self.modelos.keys())
    
    @property
    def tipos_permitidos(self):
        return list(self.le.classes_)
    
    def _preparar_datos(self, data: MaquinaInput) -> pd.DataFrame:
        fila = data.dict()
        df = pd.DataFrame([fila])
        
        rename_map = {
            'air_temperature': 'Air temperature K',
            'process_temperature': 'Process temperature K',
            'rotational_speed': 'Rotational speed rpm',
            'torque': 'Torque Nm',
            'tool_wear': 'Tool wear min',
            'twf': 'TWF', 'hdf': 'HDF', 'pwf': 'PWF', 'osf': 'OSF', 'rnf': 'RNF'
        }
        df = df.rename(columns=rename_map)
        
        try:
            df['Type'] = self.le.transform(df['Type'])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo inválido. Valores permitidos: {self.tipos_permitidos}"
            )
        
        df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True).str.strip()
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        df = df[self.prep['columnas_esperadas']]
        
        return df
    
    def predecir(self, maquina_input: MaquinaInput) -> dict:
        X = self._preparar_datos(maquina_input)
        
        predicciones = {
            "Random Forest": int(self.rf.predict(X)[0]),
            "XGBoost": int(self.xgb.predict(X)[0]),
            "LightGBM": int(self.lgbm.predict(X)[0])
        }
        
        votos = list(predicciones.values())
        final = 1 if sum(votos) >= 2 else 0
        decision = "FALLA" if final else "NO FALLA"
        
        return {
            "predicciones": predicciones,
            "votos_falla": sum(votos),
            "decision_final": decision
        }
    
    def guardar_prediccion(self, db, maquina_input: MaquinaInput, 
                          predicciones: dict, decision_final: str, usuario_id: int) -> int:
        nueva_maquina = Maquina(
            usuario_id=usuario_id,  # ✅ Vincular con usuario
            type=maquina_input.Type,
            air_temperature=maquina_input.air_temperature,
            process_temperature=maquina_input.process_temperature,
            rotational_speed=maquina_input.rotational_speed,
            torque=maquina_input.torque,
            tool_wear=maquina_input.tool_wear,
            twf=maquina_input.twf,
            hdf=maquina_input.hdf,
            pwf=maquina_input.pwf,
            osf=maquina_input.osf,
            rnf=maquina_input.rnf
        )
        db.add(nueva_maquina)
        db.commit()
        db.refresh(nueva_maquina)
        
        # ✅ CORRECCIÓN: Mapeo con nombres correctos del enum
        modelo_map = {
            "Random Forest": ModeloEnum.Random_Forest,
            "XGBoost": ModeloEnum.XGBoost,
            "LightGBM": ModeloEnum.LightGBM
        }
        
        # ✅ CORRECCIÓN: Convertir string a enum correcto
        decision_enum = DecisionEnum.FALLA if decision_final == "FALLA" else DecisionEnum.NO_FALLA
        
        # 🔍 DEBUG
        print("=" * 60)
        print(f"decision_final recibido: '{decision_final}'")
        print(f"decision_enum calculado: {decision_enum}")
        print(f"decision_enum.value: '{decision_enum.value}'")
        print(f"¿Es FALLA?: {decision_final == 'FALLA'}")
        print(f"DecisionEnum.NO_FALLA: {DecisionEnum.NO_FALLA}")
        print(f"DecisionEnum.NO_FALLA.value: '{DecisionEnum.NO_FALLA.value}'")
        print("=" * 60)

        for nombre, pred in predicciones.items():
            print(f"Guardando: {nombre} | pred={pred} | decision={decision_enum.value}")
            db.add(Prediccion(
                maquina_id=nueva_maquina.id,
                modelo=modelo_map[nombre],
                prediccion=pred,
                decision_final=decision_enum
            ))
        db.commit()
        
        return nueva_maquina.id