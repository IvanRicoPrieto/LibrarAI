# claude_max_client

Cliente Python para usar modelos Claude (Opus, Sonnet, Haiku) a traves de la suscripcion de Claude Code Max, sin pagar por token via API.

## Que es

`claude_max_client` es un modulo autocontenido que envuelve el [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) oficial para proporcionar una interfaz simple de completacion de texto. En lugar de usar la API de Anthropic con `ANTHROPIC_API_KEY` (que cobra por token), este modulo aprovecha la suscripcion mensual de Claude Code Max para hacer llamadas ilimitadas.

## Requisitos

- Python 3.10+
- `claude-agent-sdk >= 0.1.0`
- `pytest-asyncio` (solo para tests)
- Suscripcion activa de Claude Code Max
- CLI de Claude Code autenticada (`claude` debe estar instalado y con sesion iniciada)

## Instalacion

```bash
pip install claude-agent-sdk
```

No se requiere configuracion adicional. El modulo detecta automaticamente la suscripcion de Claude Code Max si la CLI de Claude esta autenticada en el sistema.

## Inicio rapido

```python
from src.claude_max_client import ClaudeMaxClient

client = ClaudeMaxClient()

# Completacion simple
response = client.complete("Explica que es un qubit.")
print(response.content)

# Verificar estado
print(f"OK: {response.ok}")
print(f"Modelo: {response.model}")
print(f"Latencia: {response.latency_ms:.0f}ms")
```

## Documentacion

- [Guia de uso](usage.md) - Ejemplos detallados de todas las funcionalidades
- [Referencia API](api_reference.md) - Documentacion completa de clases, metodos y tipos
- [Arquitectura](architecture.md) - Como funciona internamente el modulo

## Estructura del modulo

```
src/claude_max_client/
├── __init__.py       # Exports publicos
├── client.py         # ClaudeMaxClient - clase principal
├── types.py          # CompletionResponse, AVAILABLE_MODELS, DEFAULT_MODEL
├── exceptions.py     # Jerarquia de excepciones
├── utils.py          # Validacion de archivos/imagenes, estimacion de tokens
└── docs/
    ├── README.md         # Este archivo
    ├── usage.md          # Guia de uso con ejemplos
    ├── api_reference.md  # Referencia completa de la API
    └── architecture.md   # Arquitectura interna
```

## Modelos disponibles

| Modelo | ID | Ventana de contexto | Uso recomendado |
|--------|----|--------------------:|-----------------|
| Claude Opus 4.5 | `claude-opus-4-5-20251101` | 200K tokens | Maximo razonamiento, tareas complejas (default) |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | 200K tokens | Balance calidad/velocidad |
| Claude Sonnet 4 | `claude-sonnet-4-20250514` | 200K tokens | Rapido, buen rendimiento general |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200K tokens | Ultra-rapido, alto volumen |

## Tests

```bash
# Tests unitarios (sin suscripcion)
pytest tests/test_claude_max_client.py -v -k "not integration"

# Tests de integracion (requieren suscripcion activa)
pytest tests/test_claude_max_client.py -v -m integration
```
