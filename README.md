# Hair Try-On | Probador Virtual de Peinados

Un aplicativo web que utiliza inteligencia artificial para que los usuarios puedan ver cómo se verían con diferentes estilos de cabello. La aplicación usa modelos de segmentación y generación de imágenes basados en inteligencia artificial.

---

## Características

✨ **Análisis de cabello**: Detecta y segmenta automáticamente el área del cabello utilizando modelos de visión por computadora.

✨ **Generación de estilos**: Genera nuevos estilos de cabello basados en descripciones textuales usando inteligencia artificial.

✨ **Interfaz web intuitiva**: UI responsiva y fácil de usar para cargar imágenes y visualizar resultados.

✨ **API moderna**: Backend basado en FastAPI para procesamiento rápido de imágenes.

---

## Requisitos previos

- Python 3.8+
- pip (gestor de paquetes de Python)
- Navegador moderno (Chrome, Firefox, Safari, Edge)

---

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd hair_try_on
```

### 2. Crear un entorno virtual

```bash
python -m venv venv
```

### 3. Activar el entorno virtual

**En Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**En Windows (CMD):**
```bash
.\venv\Scripts\activate.bat
```

**En macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Descargar modelos (obligatorio)

Ejecuta el script para descargar los modelos de segmentación y detección facial:

```bash
python scripts/download_model.py
```

Esto descargará:
- `hair_segmenter.tflite` – segmentación de cabello
- `face_landmarker.task` – landmarks faciales (protección precisa del rostro, óvalo facial)

Sin estos modelos, la aplicación iniciará pero las funciones de cabello devolverán error 503.

---

## Uso

### Iniciar la aplicación

Ejecuta desde la raíz del proyecto:

```bash
uvicorn main:app --reload
```

O simplemente:

```bash
python -m uvicorn main:app --reload
```

La aplicación estará disponible en: **http://localhost:8000**

---

## Estructura del Proyecto

```
hair_try_on/
├── main.py                          # Aplicación principal (FastAPI)
├── requirements.txt                 # Dependencias de Python
├── Models/                          # Modelos de TensorFlow Lite (ejecutar scripts/download_model.py)
│   ├── hair_segmenter.tflite       # Segmentador de cabello
│   └── face_landmarker.task  # Landmarks faciales (protección precisa del rostro)
├── scripts/
│   └── download_model.py           # Script para descargar modelos
├── routes/                          # Rutas de la API
│   ├── segmentation.py              # Endpoint de segmentación
│   └── inpainting.py                # Endpoint de generación de estilos
├── services/                        # Lógica de negocio
│   ├── segmentation_service.py      # Servicio de segmentación
│   └── inpainting_service.py        # Servicio de inpainting/generación
├── ui/                              # Interfaz web frontend
│   ├── index.html                   # Página HTML principal
│   └── assets/
│       ├── app.js                   # Lógica de la interfaz (JavaScript)
│       └── styles.css               # Estilos de la interfaz
└── venv/                            # Entorno virtual (no incluirlo en git)
```

---

## API Endpoints

### 1. **GET `/`**

Retorna la interfaz web principal.

**Respuesta:** Archivo HTML de la interfaz

---

### 2. **POST `/segmentation`**

Realiza la segmentación del cabello en una imagen.

**Parámetros:**
- `image` (FormData): Archivo de imagen (JPG, PNG, etc.)

**Respuesta:**
```json
{
  "category_mask": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**Ejemplo de uso con cURL:**
```bash
curl -X POST "http://localhost:8000/segmentation" \
  -F "image=@foto.jpg"
```

---

### 3. **POST `/hair/change`**

Genera un nuevo estilo de cabello basado en una descripción textual.

**Parámetros:**
- `image` (FormData): Archivo de imagen
- `prompt` (FormData): Descripción del estilo deseado (ej: "cabello rojo ondulado")

**Respuesta:**
```json
{
  "status_code": 200,
  "message": "Imagen generada correctamente",
  "image_base64": "iVBORw0KGgoAAAANS...",
  "image_mime_type": "image/png",
  "saved_to": "Inputs/hair_result_20260312_153045.png"
}
```

**Ejemplo de uso con cURL:**
```bash
curl -X POST "http://localhost:8000/hair/change" \
  -F "image=@foto.jpg" \
  -F "prompt=cabello rubio liso"
```

---

## Dependencias principales

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| FastAPI | ≥ 0.134.0 | Framework web moderno para la API |
| MediaPipe | ≥ 0.10.32 | Visión por computadora y modelos de IA |
| Boto3 | ≥ 1.42.55 | Integración con servicios AWS |
| Uvicorn | (incluido en FastAPI[standard]) | Servidor ASGI |
| OpenCV | (indirecto) | Procesamiento de imágenes |
| NumPy | (indirecto) | Operaciones numéricas |

---

## Archivos de configuración

### `.env_example`

Archivo de ejemplo de variables de entorno. Cópialo a `.env` si necesitas configurar credenciales AWS u otras variables.

```bash
cp .env_example .env
```

Luego edita `.env` con tus valores.

---

## Solución de problemas

### El servidor no inicia

- Verifica que el entorno virtual esté activado
- Asegúrate de que has instalado todas las dependencias: `pip install -r requirements.txt`
- Prueba con un puerto diferente: `uvicorn main:app --port 8001`

### Errores de modelos (`*.tflite`)

- Ejecuta `python scripts/download_model.py` para descargar los modelos automáticamente
- Si falla, descarga manualmente desde:
  - [hair_segmenter.tflite](https://huggingface.co/yolain/selfie_multiclass_256x256/blob/main/hair_segmenter.tflite)
  - [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

### Problemas de memoria

- Si obtienes errores de memoria al procesar imágenes grandes, la función `resize_for_model()` en `inpainting.py` redimensiona automáticamente hasta 1024x1024

### La interfaz web no se carga

- Verifica que estás accediendo a `http://localhost:8000` (no `http://127.0.0.1:8000`)
- Limpia el caché del navegador (Ctrl+Shift+Del)

---

## Desarrollo

### Agregar nuevos endpoints

1. Crea un nuevo archivo en la carpeta `routes/`
2. Define las funciones con decoradores `@router.post()` o `@router.get()`
3. Importa y registra el router en `main.py`:

```python
from routes.nuevo_endpoint import nuevo_route
app.include_router(nuevo_route)
```

### Estructura de respuestas

Todas las respuestas de la API retornan imágenes en formato **base64** para compatibilidad con la interfaz web.

---

## Notas de seguridad

⚠️ **IMPORTANTE:**

- Nunca expongas variables sensibles en el código
- Usa un archivo `.env` para credenciales AWS
- Para producción, configura CORS adecuadamente y usa variables de entorno
- Las imágenes procesadas se guardan en `Inputs/` (considera limpiar periódicamente)

---

## Rendimiento

- **Modelos TensorFlow Lite**: Modelos optimizados para ejecución rápida en CPU
- **Resizing automático**: Las imágenes se redimensionan a máximo 1024x1024 para optimizar memoria
- **Base64 encoding**: Optimizado para transferencia web

---

## Licencia

Este proyecto está disponible bajo una licencia libre (especificar si aplica).

---

## Última actualización

23 de marzo de 2026
