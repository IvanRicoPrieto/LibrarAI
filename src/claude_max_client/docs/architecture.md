# Arquitectura

## Vision general

`claude_max_client` actua como un wrapper simplificado sobre el Claude Agent SDK oficial. Su objetivo es convertir el SDK orientado a agentes (con herramientas, multiples turnos, etc.) en una interfaz simple de completacion de texto.

```
Tu codigo
    │
    ▼
ClaudeMaxClient.complete()      ← Interfaz sincrona simple
    │
    ▼
ClaudeMaxClient.acomplete()     ← Core asincrono
    │
    ▼
claude_agent_sdk.query()        ← SDK oficial (async generator)
    │
    ▼
Claude Code CLI (subproceso)    ← Usa la suscripcion Max
    │
    ▼
Anthropic API                   ← Sin coste por token
```

## Como funciona la autenticacion

El mecanismo clave es forzar que el Claude Agent SDK NO use una API key, sino la suscripcion:

```python
options = ClaudeAgentOptions(
    model="claude-opus-4-5-20251101",
    max_turns=1,
    env={"ANTHROPIC_API_KEY": ""},  # <-- Fuerza uso de suscripcion
)
```

Cuando `ANTHROPIC_API_KEY` esta vacia, el SDK cae al mecanismo de autenticacion de Claude Code, que usa el token OAuth de la suscripcion Max. Esto permite hacer llamadas a los modelos Claude sin coste adicional por token.

**Requisito**: La CLI de Claude Code (`claude`) debe estar instalada y autenticada en el sistema. Si nunca se ha autenticado, ejecutar `claude setup-token`.

## Flujo de una llamada

### 1. Entrada del usuario

```python
response = client.complete(
    prompt="Que es un qubit?",
    system="Eres un profesor de fisica.",
    files=["context.py"],
    images=["diagram.png"],
)
```

### 2. Construccion del prompt (`_build_prompt`)

El metodo `_build_prompt` ensambla un prompt compuesto:

```
[prompt original]

--- ARCHIVOS ADJUNTOS ---
### context.py
```python
[contenido del archivo]
```
--- FIN ARCHIVOS ---

Analiza las siguientes imagenes:
- Imagen: /ruta/absoluta/diagram.png
Para cada imagen, lee el archivo y describe su contenido...
```

Los archivos de texto se inyectan inline. Las imagenes se pasan como rutas absolutas para que Claude Code las lea via la herramienta `Read`.

### 3. Configuracion del SDK

```python
options = ClaudeAgentOptions(
    model="claude-opus-4-5-20251101",
    max_turns=1,                    # Sin agente, completacion directa
    system_prompt="Eres un profesor de fisica.",
    allowed_tools=["Read"],         # Solo si hay imagenes
    permission_mode="bypassPermissions",  # Solo si hay herramientas
    env={"ANTHROPIC_API_KEY": ""},
)
```

- `max_turns=1`: Desactiva el comportamiento de agente. Una sola iteracion.
- `allowed_tools=[]`: Sin herramientas por defecto. Se habilita `["Read"]` solo cuando hay imagenes.
- `permission_mode`: Se activa `"bypassPermissions"` solo cuando hay herramientas habilitadas.

### 4. Iteracion de mensajes del SDK

El SDK devuelve un async generator con dos tipos de mensajes:

```python
async for message in query(prompt=full_prompt, options=options):
    if isinstance(message, AssistantMessage):
        # Contiene TextBlock con el texto generado
        for block in message.content:
            if isinstance(block, TextBlock):
                content_parts.append(block.text)
                # Streaming callback aqui si esta habilitado

    elif isinstance(message, ResultMessage):
        # Metadata final: session_id, uso, errores
        result_data = message
```

### 5. Post-procesamiento

- **JSON mode**: Si `json_mode=True`, se aplica `_strip_markdown_json()` que limpia bloques `` ```json...``` `` que Claude pueda haber envuelto.
- **Metricas**: Se extraen tokens, coste (tipicamente 0 con Max), y latencia.
- **Errores**: Se clasifican y lanzan excepciones especificas segun el contenido del error.

## Sync/Async wrapping

Los metodos sincronos (`complete`, `batch_complete`) envuelven los metodos async:

```python
def complete(self, **kwargs):
    return self._run_async(self.acomplete(**kwargs))

@staticmethod
def _run_async(coro):
    try:
        asyncio.get_running_loop()
        # Ya hay un loop corriendo (e.g. Jupyter)
        # Usar thread para evitar deadlock
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No hay loop - caso normal
        return asyncio.run(coro)
```

Esto permite usar `client.complete()` tanto en scripts normales como dentro de Jupyter notebooks o aplicaciones async existentes.

## Batch processing

El batch usa `asyncio.Semaphore` para controlar la concurrencia:

```
Prompt 0 ──┐
Prompt 1 ──┤
Prompt 2 ──┼──▶ Semaphore(max_concurrency=3) ──▶ acomplete() ──▶ results[i]
Prompt 3 ──┤
Prompt 4 ──┘
```

- Los prompts se normalizan a formato dict via `normalize_prompt_input()`
- Cada prompt se procesa independientemente con reintentos exponenciales
- Los errores no interrumpen el lote: las respuestas fallidas tienen `error != None`
- El `progress_callback` se invoca despues de cada completacion

## Manejo de archivos

### Imagenes

Las imagenes se pasan como rutas al prompt. Claude Code accede a ellas via la herramienta `Read` del filesystem. Por eso cuando hay imagenes se configura `allowed_tools=["Read"]`.

Formatos soportados (limitados por Claude Vision):
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- WebP (`.webp`)

### Archivos de texto

Los archivos de texto se leen previamente y su contenido se inyecta directamente en el prompt como bloques de codigo con syntax highlighting.

Validaciones:
- El archivo debe existir y ser un archivo regular (no directorio)
- Se rechazan binarios conocidos (`.exe`, `.dll`, `.zip`, `.pdf`, `.mp3`, etc.)
- Tamano maximo: 1MB por archivo
- Encoding: UTF-8 con fallback a Latin-1

## Clasificacion de errores

El metodo `_handle_sdk_error` clasifica errores del SDK en excepciones especificas basandose en el contenido del texto de error:

| Patron en el error | Excepcion |
|---------------------|-----------|
| "authentication", "unauthorized", "login" | `AuthenticationError` |
| "rate" + "limit" | `RateLimitError` |
| "model" + ("not available" / "not found") | `ModelNotAvailableError` |
| Cualquier otro | `CompletionError` |

## Diseno modular

El modulo esta disenado para ser independiente:

- **Sin dependencias de LibrarAI**: No importa nada de `src.*` (excepto a si mismo)
- **Sin configuracion externa**: No lee archivos de config, solo usa parametros del constructor
- **Sin estado global**: Cada instancia de `ClaudeMaxClient` es independiente
- **Reutilizable**: Se puede copiar la carpeta `src/claude_max_client/` a cualquier proyecto Python

La unica dependencia externa es `claude-agent-sdk`, que a su vez solo requiere que la CLI de Claude Code este instalada y autenticada.
