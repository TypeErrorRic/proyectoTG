# Patrón DAO - Documentación de Refactorización

## Resumen

El proyecto ha sido refactorizado para implementar correctamente el patrón **DAO (Data Access Object)**, separando la lógica de negocio de la capa de acceso a datos.

## Estructura Nueva

```
src/
├── models/           # Entidades de dominio
│   └── __init__.py   # User, Configuration, Capture (dataclasses)
│
├── dao/              # Capa de acceso a datos
│   ├── __init__.py
│   ├── user_dao.py          # UserDAO
│   ├── configuration_dao.py # ConfigurationDAO
│   └── capture_dao.py       # CaptureDAO
│
├── api/
│   └── dbConection.py  # Funciones de compatibilidad (ahora wrappea DAOs)
│
├── GUI.py            # Interfaz gráfica (usa DAOs directamente)
└── LoginWindow.py    # Login (usa UserDAO)
```

## Cambios Principales

### 1. Modelos de Entidad (`src/models/__init__.py`)

Se crearon clases `@dataclass` para representar las entidades del dominio:

- **User**: Representa un usuario del sistema
- **Configuration**: Representa una configuración RANSAC
- **Capture**: Representa una captura de imagen

Cada clase tiene:
- `from_dict(data)`: Constructor desde diccionario de BD
- `to_dict()`: Conversión a diccionario (compatibilidad con código antiguo)

```python
from src.models import User, Configuration, Capture

# Crear desde BD
user = User.from_dict(row_from_database)

# Usar como objeto
print(user.username, user.email)

# Convertir a dict si es necesario
user_dict = user.to_dict()
```

### 2. Clases DAO (`src/dao/`)

Se implementaron 3 clases DAO:

#### **UserDAO** (`user_dao.py`)
```python
from src.dao.user_dao import UserDAO

# Autenticación
user = UserDAO.authenticate(username, password)

# Obtener por ID
user = UserDAO.get_by_id(user_id)

# Crear usuario
user_id = UserDAO.create(username, email, password, full_name, role)

# Obtener estadísticas
stats = UserDAO.get_stats(user_id)
```

#### **ConfigurationDAO** (`configuration_dao.py`)
```python
from src.dao.configuration_dao import ConfigurationDAO

# Obtener configuraciones de un usuario
configs = ConfigurationDAO.get_user_configurations(user_id)

# Crear configuración
config_id = ConfigurationDAO.create(user_id, name, params, description)

# Actualizar configuración
ConfigurationDAO.update(config_id, params, name, description)

# Eliminar configuración
ConfigurationDAO.delete(config_id)

# Establecer como predeterminada
ConfigurationDAO.set_as_default(config_id, user_id)
```

#### **CaptureDAO** (`capture_dao.py`)
```python
from src.dao.capture_dao import CaptureDAO

# Obtener capturas de usuario
captures = CaptureDAO.get_user_captures(user_id, limit=100)

# Crear captura
capture_id = CaptureDAO.create(user_id, filename, mode, config_id, metadata, image_bytes)

# Eliminar captura
CaptureDAO.delete(capture_id)

# Toggle favorito
is_fav = CaptureDAO.toggle_favorite(capture_id)

# Actualizar notas
CaptureDAO.update_notes(capture_id, notes, tags)
```

### 3. Compatibilidad hacia atrás (`src/api/dbConection.py`)

Las funciones originales **siguen existiendo** pero ahora delegan a las clases DAO:

```python
# Estas funciones todavía funcionan (pero están marcadas como DEPRECATED)
from src.api.dbConection import authenticate_user, get_user_captures

user = authenticate_user(username, password)  # → UserDAO.authenticate()
captures = get_user_captures(user_id)         # → CaptureDAO.get_user_captures()
```

⚠️ **Se recomienda usar las clases DAO directamente en código nuevo.**

### 4. Actualización de GUI y LoginWindow

- **LoginWindow.py**: Ahora usa `UserDAO.authenticate()` directamente
- **GUI.py**: Todas las llamadas a BD usan las clases DAO

Ejemplo de cambio en `GUI.py`:

```python
# ANTES (no recomendado)
from src.api.dbConection import get_user_captures
captures = get_user_captures(user_id)

# AHORA (patrón DAO)
from src.dao.capture_dao import CaptureDAO
captures = CaptureDAO.get_user_captures(user_id)
```

## Beneficios del Patrón DAO

1. **Separación de responsabilidades**: La GUI no conoce los detalles de SQL
2. **Encapsulación**: Los datos se manejan como objetos, no como diccionarios
3. **Mantenibilidad**: Cambios en BD solo afectan la capa DAO
4. **Testabilidad**: Se pueden crear mocks de las clases DAO fácilmente
5. **Type safety**: Las clases tienen tipos definidos (User, Configuration, Capture)

## Migración Gradual

El proyecto mantiene **compatibilidad 100% hacia atrás**:

- Las funciones antiguas en `dbConection.py` siguen funcionando
- El código existente no se rompe
- Se puede migrar gradualmente a usar DAOs

## Ejemplo de Uso Completo

```python
# Login
from src.dao.user_dao import UserDAO
user = UserDAO.authenticate("admin", "password")

if user:
    print(f"Bienvenido {user.username}")
    
    # Obtener configuraciones
    from src.dao.configuration_dao import ConfigurationDAO
    configs = ConfigurationDAO.get_user_configurations(user.id)
    
    for config in configs:
        print(f"Config: {config.config_name}")
        print(f"  - Max iters: {config.max_iters}")
        print(f"  - Dist thresh: {config.dist_thresh}")
    
    # Obtener capturas
    from src.dao.capture_dao import CaptureDAO
    captures = CaptureDAO.get_user_captures(user.id, limit=10)
    
    for capture in captures:
        print(f"Captura: {capture.filename}")
        print(f"  - Tamaño: {capture.file_size_bytes} bytes")
        print(f"  - FPS: {capture.fps}")
```

## Notas Técnicas

- Las clases DAO usan métodos **estáticos** (`@staticmethod`)
- Los métodos DAO retornan **objetos** (User, Configuration, Capture) o listas de objetos
- Los wrappers de compatibilidad en `dbConection.py` convierten objetos a dicts con `.to_dict()`
- La conexión a BD sigue usando `get_connection()` de `dbConection.py`

## Próximos Pasos (Opcional)

Para mejorar aún más el patrón DAO:

1. **Agregar validación** en los modelos (validators en dataclass)
2. **Connection pooling** para optimizar conexiones BD
3. **Transacciones** para operaciones que requieran atomicidad
4. **Cache** para consultas frecuentes (configuraciones, usuarios)
5. **Unit tests** para cada clase DAO

---

**Fecha de refactorización**: 9 de diciembre de 2025
**Archivos creados**:
- `src/models/__init__.py`
- `src/dao/__init__.py`
- `src/dao/user_dao.py`
- `src/dao/configuration_dao.py`
- `src/dao/capture_dao.py`

**Archivos modificados**:
- `src/api/dbConection.py` (ahora wrapper de compatibilidad)
- `src/GUI.py` (usa DAOs directamente)
- `src/LoginWindow.py` (usa UserDAO)
