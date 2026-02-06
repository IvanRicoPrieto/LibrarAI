# Referencia API

## ClaudeMaxClient

```python
from src.claude_max_client import ClaudeMaxClient
```

### Constructor

```python
ClaudeMaxClient(
    model: str = "claude-opus-4-5-20251101",
    default_max_tokens: int = 4096,
    default_temperature: float = 0.3,
)
```

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `model` | `str` | `"claude-opus-4-5-20251101"` | Modelo Claude a usar por defecto |
| `default_max_tokens` | `int` | `4096` | Tokens maximos de salida por defecto |
| `default_temperature` | `float` | `0.3` | Temperatura de sampling (0.0 - 1.0) |

Si el modelo no esta en `AVAILABLE_MODELS`, se emite un warning pero se permite su uso.

---

### complete()

```python
def complete(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    images: list[str] | None = None,
    files: list[str] | None = None,
    json_mode: bool = False,
    stream: bool = False,
    stream_callback: Callable[[str], None] | None = None,
) -> CompletionResponse
```

Genera una completacion de texto de forma sincrona.

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `prompt` | `str` | *requerido* | Texto de entrada / pregunta |
| `system` | `str \| None` | `None` | System prompt para guiar el comportamiento |
| `model` | `str \| None` | `None` | Override del modelo (usa el del constructor si None) |
| `max_tokens` | `int \| None` | `None` | Override de tokens maximos de salida |
| `temperature` | `float \| None` | `None` | Override de temperatura de sampling |
| `images` | `list[str] \| None` | `None` | Rutas a imagenes locales para analisis visual |
| `files` | `list[str] \| None` | `None` | Rutas a archivos de texto/codigo como contexto |
| `json_mode` | `bool` | `False` | Si True, fuerza respuesta JSON |
| `stream` | `bool` | `False` | Si True, entrega respuesta incrementalmente |
| `stream_callback` | `Callable[[str], None] \| None` | `None` | Funcion que recibe cada fragmento de texto |

**Returns**: `CompletionResponse`

**Raises**:
- `AuthenticationError` - Suscripcion no activa
- `RateLimitError` - Limite de rate alcanzado
- `CompletionError` - Error durante la generacion
- `ImageProcessingError` - Imagen no encontrada o formato no soportado
- `FileProcessingError` - Archivo no encontrado, binario, o demasiado grande

---

### acomplete()

```python
async def acomplete(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    images: list[str] | None = None,
    files: list[str] | None = None,
    json_mode: bool = False,
    stream: bool = False,
    stream_callback: Callable[[str], None] | None = None,
) -> CompletionResponse
```

Version asincrona de `complete()`. Mismos parametros, comportamiento y excepciones.

---

### batch_complete()

```python
def batch_complete(
    prompts: list[str | dict],
    system: str | None = None,
    model: str | None = None,
    max_concurrency: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
    retry_on_error: bool = True,
    max_retries: int = 3,
) -> list[CompletionResponse]
```

Procesa multiples prompts en paralelo de forma sincrona.

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `prompts` | `list[str \| dict]` | *requerido* | Lista de prompts (string o dict con campos) |
| `system` | `str \| None` | `None` | System prompt compartido |
| `model` | `str \| None` | `None` | Override del modelo para todas las llamadas |
| `max_concurrency` | `int` | `5` | Llamadas concurrentes maximas |
| `progress_callback` | `Callable[[int, int], None] \| None` | `None` | Callback `(completados, total)` |
| `retry_on_error` | `bool` | `True` | Reintentar llamadas fallidas |
| `max_retries` | `int` | `3` | Reintentos maximos por prompt |

**Formato de prompts como dict**:

```python
{
    "prompt": str,           # Requerido
    "system": str,           # Opcional - sobreescribe el system compartido
    "model": str,            # Opcional
    "images": list[str],     # Opcional
    "files": list[str],      # Opcional
    "json_mode": bool,       # Opcional
    "max_tokens": int,       # Opcional
    "temperature": float,    # Opcional
}
```

**Returns**: `list[CompletionResponse]` - En el mismo orden que los prompts. Las fallidas tienen `error != None`.

---

### abatch_complete()

```python
async def abatch_complete(
    prompts: list[str | dict],
    system: str | None = None,
    model: str | None = None,
    max_concurrency: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
    retry_on_error: bool = True,
    max_retries: int = 3,
) -> list[CompletionResponse]
```

Version asincrona de `batch_complete()`. Mismos parametros y comportamiento.

---

### is_authenticated()

```python
def is_authenticated(self) -> bool
```

Verifica que la suscripcion de Claude Code Max esta activa realizando una llamada minima.

**Returns**: `True` si la suscripcion funciona correctamente.

---

## CompletionResponse

```python
from src.claude_max_client import CompletionResponse
```

Dataclass con la respuesta de una completacion.

### Campos

| Campo | Tipo | Default | Descripcion |
|-------|------|---------|-------------|
| `content` | `str` | *requerido* | Texto de la respuesta generada |
| `model` | `str` | *requerido* | ID del modelo usado |
| `tokens_input` | `int` | `0` | Tokens de entrada (estimados o reportados) |
| `tokens_output` | `int` | `0` | Tokens de salida (estimados o reportados) |
| `cost_usd` | `float \| None` | `None` | Coste reportado (tipicamente 0 con Max) |
| `latency_ms` | `float` | `0.0` | Latencia total en milisegundos |
| `session_id` | `str \| None` | `None` | ID de sesion del Agent SDK |
| `raw_usage` | `dict \| None` | `None` | Datos crudos de uso del SDK |
| `error` | `str \| None` | `None` | Mensaje de error (solo en batch) |

### Propiedades

#### ok

```python
@property
def ok(self) -> bool
```

`True` si la respuesta no tiene errores y el contenido no esta vacio.

### Metodos

#### to_dict()

```python
def to_dict(self) -> dict[str, Any]
```

Serializa la respuesta a diccionario con todos los campos.

---

## Excepciones

Todas heredan de `ClaudeMaxError`:

```
ClaudeMaxError
├── AuthenticationError    - Suscripcion no valida o no autenticada
├── RateLimitError         - Limite de rate alcanzado
├── ModelNotAvailableError - Modelo no disponible
├── ImageProcessingError   - Error procesando imagen
├── FileProcessingError    - Error procesando archivo de texto/codigo
└── CompletionError        - Error durante la generacion
```

### RateLimitError

Tiene un campo adicional `retry_after: float | None` que indica los segundos de espera sugeridos.

---

## Constantes

### AVAILABLE_MODELS

```python
from src.claude_max_client import AVAILABLE_MODELS
```

Diccionario con los modelos disponibles. Cada entrada tiene:

```python
{
    "name": str,            # Nombre legible
    "context_window": int,  # Ventana de contexto en tokens
    "description": str,     # Descripcion del modelo
}
```

### DEFAULT_MODEL

```python
from src.claude_max_client import DEFAULT_MODEL
# "claude-opus-4-5-20251101"
```

---

## Utilidades (utils)

Funciones auxiliares exportadas desde `src.claude_max_client.utils`:

### validate_image_path(image_path: str) -> Path

Valida existencia y formato de una imagen. Formatos: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`.

**Raises**: `ImageProcessingError`

### validate_file_path(file_path: str) -> Path

Valida existencia, tipo y tamano de un archivo de texto. Rechaza binarios conocidos y archivos mayores a 1MB.

**Raises**: `FileProcessingError`

### read_text_file(file_path: str, encoding: str = "utf-8") -> str

Lee un archivo de texto. Intenta UTF-8, fallback a Latin-1.

**Raises**: `FileProcessingError`

### is_image_file(file_path: str) -> bool

Comprueba si un archivo es una imagen soportada por Claude Vision.

### is_text_file(file_path: str) -> bool

Comprueba si un archivo es reconocido como texto legible. Archivos sin extension devuelven `True`.

### encode_image_to_base64(image_path: str) -> tuple[str, str]

Lee una imagen y la codifica en base64. Devuelve `(base64_data, media_type)`.

### estimate_tokens(text: str) -> int

Estimacion rapida de tokens (~4 caracteres por token).

### estimate_image_tokens(image_path: str) -> int

Estimacion de tokens que consumira una imagen basandose en el tamano del archivo.

### normalize_prompt_input(prompt_input: str | dict, shared_system: str | None = None) -> dict

Normaliza un input de prompt a formato dict para batch processing. Aplica el system prompt compartido si el input no tiene uno propio.
