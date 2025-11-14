import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from models import Usuario, Maquina, Prediccion

class ReportService:
    def generar_reporte_tipo(self, db: Session):
        query ="""
        SELECT m.type, COUNT(*) as total, SUM(p.prediccion) as fallos
        FROM maquinas m
        JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
        GROUP BY m.type
        """
        df = pd.read_sql(query, db.bind)
        
        if df.empty:
            raise HTTPException(404, "No hay datos")
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x='type', y='fallos', data=df, palette='Reds')
        plt.title("Fallos por Tipo de Máquina", fontsize=14, fontweight='bold')
        plt.xlabel("Tipo")
        plt.ylabel("Fallos")
        plt.tight_layout()
        plt.savefig("reporte_tipo.png", dpi=150)
        plt.close()
        
        return FileResponse("reporte_tipo.png", media_type="image/png")
    
    def generar_reporte_mensual(self, db: Session):
        query = """
        SELECT DATE_FORMAT(m.fecha_registro, '%Y-%m') as mes, 
               SUM(p.prediccion) as fallos
        FROM maquinas m
        JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
        GROUP BY mes ORDER BY mes DESC
        """
        df = pd.read_sql(query, db.bind)
        
        if df.empty:
            raise HTTPException(404, "No hay datos")
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='mes', y='fallos', data=df, marker='o', color='crimson')
        plt.title("Evolución Mensual", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("reporte_mes.png", dpi=150)
        plt.close()
        
        return FileResponse("reporte_mes.png", media_type="image/png")
    
    def obtener_ultimos_fallos(self, db: Session):
        query = """
        SELECT m.id, m.type, m.torque, m.tool_wear, m.fecha_registro
        FROM maquinas m
        JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
        WHERE p.prediccion = 1
        ORDER BY m.fecha_registro DESC LIMIT 5
        """
        df = pd.read_sql(query, db.bind)
        
        if df.empty:
            raise HTTPException(404, "No hay fallos")
        
        df['fecha_registro'] = df['fecha_registro'].astype(str)
        return JSONResponse({"ultimos_fallos": df.to_dict(orient="records")})
    
    def obtener_reporte_usuario(self, db: Session, usuario_id: int):
        # Verificar usuario
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(404, "Usuario no encontrado")
        
        # Contar predicciones
        total_predicciones = db.query(Maquina).filter(
            Maquina.usuario_id == usuario_id
        ).count()
        
        # Contar fallos predichos
        query = """
        SELECT COUNT(*) as total_fallos
        FROM maquinas m
        JOIN predicciones p ON m.id = p.maquina_id AND p.modelo = 'LightGBM'
        WHERE m.usuario_id = :usuario_id AND p.prediccion = 1
        """
        result = db.execute(query, {"usuario_id": usuario_id}).fetchone()
        
        return {
            "usuario_id": usuario_id,
            "nombre": f"{usuario.nombre} {usuario.apellido}",
            "total_predicciones": total_predicciones,
            "total_fallos_detectados": result[0] if result else 0,
            "porcentaje_fallos": round((result[0] / total_predicciones * 100), 2) if total_predicciones > 0 else 0
        }