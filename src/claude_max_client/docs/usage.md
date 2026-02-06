# Guia de uso

## Completacion basica

```python
from src.claude_max_client import ClaudeMaxClient

client = ClaudeMaxClient()
response = client.complete("Explica la superposicion cuantica en 3 frases.")
print(response.content)
```

## Configuracion del cliente

El constructor acepta valores por defecto que se aplican a todas las llamadas:

```python
client = ClaudeMaxClient(
    model="claude-opus-4-5-20251101",     # Modelo por defecto
    default_max_tokens=8192,               # Tokens maximos de salida
    default_temperature=0.7,               # Temperatura de sampling (0.0-1.0)
)
```

Cada parametro puede ser sobreescrito en llamadas individuales.

## System prompt

```python
response = client.complete(
    prompt="Que es la decoherencia?",
    system="Eres un profesor de fisica cuantica. Responde de forma concisa y precisa.",
)
```

## Seleccion de modelo por llamada

```python
# Usar Opus para una tarea compleja
response = client.complete(
    prompt="Analiza las implicaciones del teorema de Bell...",
    model="claude-opus-4-5-20251101",
)

# Usar Haiku para algo rapido
response = client.complete(
    prompt="Resume en una frase: ...",
    model="claude-haiku-4-5-20251001",
)
```

## Control de temperatura y tokens

```python
# Respuesta determinista (temperatura baja)
response = client.complete(
    prompt="Cual es la constante de Planck?",
    temperature=0.0,
    max_tokens=100,
)

# Respuesta creativa (temperatura alta)
response = client.complete(
    prompt="Escribe una analogia para el entrelazamiento cuantico.",
    temperature=0.9,
    max_tokens=2000,
)
```

## Streaming

Para recibir la respuesta incrementalmente a medida que se genera:

```python
def on_chunk(text):
    print(text, end="", flush=True)

response = client.complete(
    prompt="Explica el algoritmo de Shor paso a paso.",
    stream=True,
    stream_callback=on_chunk,
)
print()  # Salto de linea final

# response.content contiene la respuesta completa al terminar
```

## Modo JSON

Fuerza a Claude a responder exclusivamente con JSON valido:

```python
import json

response = client.complete(
    prompt='Lista 3 puertas cuanticas con su nombre y descripcion.',
    json_mode=True,
)

data = json.loads(response.content)
for gate in data["gates"]:
    print(f"- {gate['name']}: {gate['description']}")
```

El modulo limpia automaticamente los bloques markdown `` ```json `` que Claude pueda envolver alrededor del JSON.

## Vision (imagenes)

Envia imagenes locales para que Claude las analice:

```python
response = client.complete(
    prompt="Describe esta figura del circuito cuantico.",
    images=["data/figures/circuit_bell.png"],
)
```

Multiples imagenes:

```python
response = client.complete(
    prompt="Compara estas dos graficas de resultados.",
    images=[
        "results/histogram_sim.png",
        "results/histogram_real.png",
    ],
)
```

**Formatos soportados**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

Internamente, el modulo configura `allowed_tools=["Read"]` para que Claude pueda acceder a las imagenes del filesystem.

## Archivos de texto/codigo como contexto

Adjunta archivos de texto o codigo fuente para que Claude los analice:

```python
response = client.complete(
    prompt="Revisa este codigo y sugiere mejoras.",
    files=["src/retrieval/engine.py"],
)
```

Multiples archivos:

```python
response = client.complete(
    prompt="Compara estas dos implementaciones y di cual es mas eficiente.",
    files=[
        "src/v1/processor.py",
        "src/v2/processor.py",
    ],
)
```

Combinado con imagenes:

```python
response = client.complete(
    prompt="Analiza el codigo y la grafica de rendimiento.",
    files=["benchmark.py"],
    images=["results/perf_chart.png"],
)
```

**Extensiones soportadas**: Codigo fuente (`.py`, `.js`, `.ts`, `.java`, `.cpp`, etc.), datos (`.json`, `.yaml`, `.csv`, `.sql`), documentacion (`.md`, `.txt`, `.tex`), config (`.toml`, `.ini`, `.env`), y mas.

**Limitaciones**:
- Tamano maximo por archivo: 1MB
- Archivos binarios (`.exe`, `.zip`, `.pdf`, `.mp4`, etc.) son rechazados
- Archivos sin extension se aceptan (pueden ser scripts)

## Batch processing

Procesa multiples prompts en paralelo con control de concurrencia:

```python
prompts = [
    "Que es un qubit?",
    "Que es superposicion?",
    "Que es entrelazamiento?",
    "Que es decoherencia?",
    "Que es una puerta cuantica?",
]

responses = client.batch_complete(
    prompts=prompts,
    max_concurrency=3,     # Maximo 3 llamadas simultaneas
    progress_callback=lambda done, total: print(f"Progreso: {done}/{total}"),
)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response.content[:100]}...")
    print()
```

### Batch con prompts heterogeneos

Cada prompt puede tener su propia configuracion:

```python
prompts = [
    {"prompt": "Responde en ingles: What is a qubit?", "system": "Answer in English."},
    {"prompt": "Explica gates cuanticos", "model": "claude-opus-4-5-20251101"},
    "Pregunta simple sobre superposicion",  # Usa defaults
]

responses = client.batch_complete(
    prompts=prompts,
    system="Eres experto en computacion cuantica.",  # System compartido (si no tiene uno propio)
    max_concurrency=5,
    retry_on_error=True,
    max_retries=3,
)
```

### Manejo de errores en batch

Las respuestas fallidas no interrumpen el lote. Puedes verificar cada una:

```python
for i, response in enumerate(responses):
    if response.ok:
        print(f"[{i}] OK: {response.content[:50]}...")
    else:
        print(f"[{i}] ERROR: {response.error}")
```

## Interfaz asincrona

Todos los metodos tienen equivalentes `async`:

```python
import asyncio

async def main():
    client = ClaudeMaxClient()

    # Completacion asincrona
    response = await client.acomplete(
        prompt="Que es un qubit?",
        system="Responde brevemente.",
    )
    print(response.content)

    # Batch asincrono
    responses = await client.abatch_complete(
        prompts=["Pregunta 1", "Pregunta 2"],
        max_concurrency=5,
    )

asyncio.run(main())
```

La interfaz asincrona es especialmente util en aplicaciones web (FastAPI, etc.) o cuando necesitas integrar con otros sistemas async.

**Nota**: Los metodos sincronos (`complete`, `batch_complete`) internamente llaman a los metodos async. Si ya estas en un contexto async (e.g. Jupyter notebooks), el modulo detecta el event loop existente y usa un thread separado para evitar deadlocks.

## Verificar autenticacion

```python
client = ClaudeMaxClient()
if client.is_authenticated():
    print("Suscripcion Claude Code Max activa")
else:
    print("No autenticado - ejecuta: claude setup-token")
```

## Inspeccion de respuestas

`CompletionResponse` proporciona metadatos utiles:

```python
response = client.complete("Hola")

print(f"Contenido: {response.content}")
print(f"Modelo: {response.model}")
print(f"Tokens entrada: {response.tokens_input}")
print(f"Tokens salida: {response.tokens_output}")
print(f"Latencia: {response.latency_ms:.0f}ms")
print(f"Coste: {response.cost_usd}")        # Tipicamente 0 o None con Max
print(f"Session ID: {response.session_id}")
print(f"OK: {response.ok}")

# Serializar a dict (util para logs, bases de datos, etc.)
data = response.to_dict()
```

## Manejo de errores

```python
from src.claude_max_client import (
    ClaudeMaxClient,
    ClaudeMaxError,
    AuthenticationError,
    RateLimitError,
    ModelNotAvailableError,
    CompletionError,
    ImageProcessingError,
    FileProcessingError,
)

client = ClaudeMaxClient()

try:
    response = client.complete("Hola")
except AuthenticationError:
    print("Suscripcion no activa. Ejecuta: claude setup-token")
except RateLimitError as e:
    print(f"Rate limit alcanzado. Retry after: {e.retry_after}s")
except ModelNotAvailableError:
    print("Modelo no disponible en tu suscripcion")
except ImageProcessingError as e:
    print(f"Error con imagen: {e}")
except FileProcessingError as e:
    print(f"Error con archivo: {e}")
except CompletionError as e:
    print(f"Error en la generacion: {e}")
except ClaudeMaxError as e:
    # Captura cualquier error del modulo
    print(f"Error general: {e}")
```
