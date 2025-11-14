from models import Usuario
from sqlalchemy.orm import Session

class AuthService:
    def authenticate(self, db: Session, username: str, password: str) -> Usuario:
        '''
        Autentica un usuario con usuario y contraseña.
        Retorna el objeto Usuario si las credenciales son correctas, None si no.
        '''
        # Buscar usuario por nombre de usuario
        usuario = db.query(Usuario).filter(Usuario.usuario == username).first()
        
        if not usuario:
            return None
        
        # Verificar contraseña (texto plano)
        if usuario.contraseña == password:
            return usuario
        
        return None
    
    def crear_usuario(self, db: Session, telefono: str, identificacion: str,
                     nombre: str, apellido: str, usuario: str, contraseña: str) -> Usuario:
        '''
        Crea un nuevo usuario en la base de datos.
        Lanza ValueError si el usuario, identificación o teléfono ya existe.
        '''
        # Verificar si ya existe un usuario con estos datos
        existente = db.query(Usuario).filter(
            (Usuario.usuario == usuario) | 
            (Usuario.identificacion == identificacion) |
            (Usuario.telefono == telefono)
        ).first()
        
        if existente:
            if existente.usuario == usuario:
                raise ValueError("El nombre de usuario ya está registrado")
            elif existente.identificacion == identificacion:
                raise ValueError("La identificación ya está registrada")
            else:
                raise ValueError("El teléfono ya está registrado")
        
        # Crear nuevo usuario (contraseña en texto plano)
        nuevo_usuario = Usuario(
            telefono=telefono,
            identificacion=identificacion,
            nombre=nombre,
            apellido=apellido,
            usuario=usuario,
            contraseña=contraseña
        )
        
        db.add(nuevo_usuario)
        db.commit()
        db.refresh(nuevo_usuario)
        
        return nuevo_usuario
    
    def obtener_usuario_por_id(self, db: Session, usuario_id: int) -> Usuario:
        '''
        Obtiene un usuario por su ID.
        Retorna None si no existe.
        '''
        return db.query(Usuario).filter(Usuario.id == usuario_id).first()
    
    def obtener_usuario_por_username(self, db: Session, username: str) -> Usuario:
        '''
        Obtiene un usuario por su nombre de usuario.
        Retorna None si no existe.
        '''
        return db.query(Usuario).filter(Usuario.usuario == username).first()