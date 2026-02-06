"""
ClaudeMaxClient - Cliente para usar modelos Claude via suscripcion Max.

Envuelve el Claude Agent SDK oficial para proporcionar una interfaz simple
de completacion de texto, sin necesidad de API key (usa la suscripcion de
Claude Code Max).

Uso basico:
    from claude_max_client import ClaudeMaxClient

    client = ClaudeMaxClient()
    response = client.complete("Explica que es un qubit.")
    print(response.content)
"""

import asyncio
import logging
import time
from typing import Any, Callable

from claude_agent_sdk import (
    query as _sdk_query,
    ClaudeAgentOptions as _ClaudeAgentOptions,
    AssistantMessage as _AssistantMessage,
    TextBlock as _TextBlock,
    ResultMessage as _ResultMessage,
)

from .types import CompletionResponse, DEFAULT_MODEL, AVAILABLE_MODELS
from .utils import (
    validate_image_path,
    validate_file_path,
    read_text_file,
    estimate_tokens,
    normalize_prompt_input,
)
from .exceptions import (
    ClaudeMaxError,
    AuthenticationError,
    RateLimitError,
    CompletionError,
    ImageProcessingError,
    FileProcessingError,
)

logger = logging.getLogger(__name__)


class ClaudeMaxClient:
    """
    Cliente para llamadas a Claude usando la suscripcion de Claude Code Max.

    Proporciona interfaces sincrona y asincrona para completaciones de texto,
    con soporte para streaming, vision (imagenes), modo JSON, y procesamiento
    por lotes.

    La autenticacion se realiza automaticamente a traves de la suscripcion
    de Claude Code Max. No requiere ANTHROPIC_API_KEY.

    Attributes:
        model: Modelo por defecto para las completaciones.
        default_max_tokens: Tokens maximos de salida por defecto.
        default_temperature: Temperatura de sampling por defecto.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        default_max_tokens: int = 4096,
        default_temperature: float = 0.3,
    ):
        """
        Args:
            model: Modelo Claude a usar por defecto.
            default_max_tokens: Maximo de tokens de salida por defecto.
            default_temperature: Temperatura de sampling por defecto (0.0-1.0).
        """
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

        if model not in AVAILABLE_MODELS:
            logger.warning(
                f"Modelo '{model}' no esta en la lista de modelos conocidos. "
                f"Modelos disponibles: {list(AVAILABLE_MODELS.keys())}"
            )

    # =========================================================================
    # Interfaz sincrona
    # =========================================================================

    def complete(
        self,
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
    ) -> CompletionResponse:
        """
        Genera una completacion de texto usando Claude via suscripcion Max.

        Args:
            prompt: Texto de entrada / pregunta para Claude.
            system: System prompt opcional.
            model: Override del modelo para esta llamada.
            max_tokens: Override de max tokens para esta llamada.
            temperature: Override de temperatura para esta llamada.
            images: Lista de rutas a imagenes locales para analisis visual.
            files: Lista de rutas a archivos de texto/codigo para incluir como contexto.
            json_mode: Si True, instruye a Claude a responder unicamente con JSON.
            stream: Si True, entrega la respuesta incrementalmente via callback.
            stream_callback: Funcion que recibe cada fragmento de texto en streaming.

        Returns:
            CompletionResponse con el texto generado y metadatos.

        Raises:
            AuthenticationError: Si la suscripcion no esta activa.
            RateLimitError: Si se alcanzo el limite de rate.
            CompletionError: Si ocurrio un error durante la generacion.
            ImageProcessingError: Si una imagen no se pudo procesar.
            FileProcessingError: Si un archivo de texto no se pudo procesar.
        """
        return self._run_async(
            self.acomplete(
                prompt=prompt,
                system=system,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                images=images,
                files=files,
                json_mode=json_mode,
                stream=stream,
                stream_callback=stream_callback,
            )
        )

    def batch_complete(
        self,
        prompts: list[str | dict],
        system: str | None = None,
        model: str | None = None,
        max_concurrency: int = 5,
        progress_callback: Callable[[int, int], None] | None = None,
        retry_on_error: bool = True,
        max_retries: int = 3,
    ) -> list[CompletionResponse]:
        """
        Procesa multiples prompts en paralelo con control de concurrencia.

        Args:
            prompts: Lista de prompts. Cada elemento puede ser un string o un dict
                con claves {prompt, system, model, images, json_mode}.
            system: System prompt compartido (se aplica a prompts que no tengan uno).
            model: Override del modelo para todas las llamadas.
            max_concurrency: Numero maximo de llamadas concurrentes.
            progress_callback: Funcion(completados, total) llamada tras cada completacion.
            retry_on_error: Si True, reintenta llamadas fallidas.
            max_retries: Numero maximo de reintentos por prompt.

        Returns:
            Lista de CompletionResponse en el mismo orden que los prompts de entrada.
            Las respuestas fallidas tendran error != None.
        """
        return self._run_async(
            self.abatch_complete(
                prompts=prompts,
                system=system,
                model=model,
                max_concurrency=max_concurrency,
                progress_callback=progress_callback,
                retry_on_error=retry_on_error,
                max_retries=max_retries,
            )
        )

    def is_authenticated(self) -> bool:
        """
        Verifica que la suscripcion de Claude Code Max esta activa.

        Realiza una llamada minima para comprobar la autenticacion.

        Returns:
            True si la suscripcion esta activa y funcional.
        """
        try:
            response = self.complete(
                prompt="Responde unicamente: ok",
                max_tokens=10,
            )
            return response.ok
        except Exception:
            return False

    # =========================================================================
    # Interfaz asincrona
    # =========================================================================

    async def acomplete(
        self,
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
    ) -> CompletionResponse:
        """Version asincrona de complete(). Mismos parametros y comportamiento."""
        effective_model = model or self.model
        effective_max_tokens = max_tokens or self.default_max_tokens
        effective_temp = temperature if temperature is not None else self.default_temperature

        # Validar imagenes si se proporcionan
        if images:
            for img_path in images:
                validate_image_path(img_path)

        # Validar archivos si se proporcionan
        if files:
            for file_path in files:
                validate_file_path(file_path)

        # Construir el prompt completo
        full_prompt = self._build_prompt(prompt, images, files, json_mode)

        # Configurar las herramientas necesarias
        allowed_tools = []
        if images:
            # Claude Code necesita Read para acceder a imagenes del filesystem
            allowed_tools = ["Read"]

        # Configurar opciones del Agent SDK
        options = _ClaudeAgentOptions(
            model=effective_model,
            max_turns=1,
            system_prompt=system,
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions" if allowed_tools else None,
            # No pasar ANTHROPIC_API_KEY para forzar uso de suscripcion
            env={"ANTHROPIC_API_KEY": ""},
        )

        start_time = time.time()
        content_parts: list[str] = []
        result_data: _ResultMessage | None = None

        try:
            if stream and stream_callback:
                options.include_partial_messages = True

            async for message in _sdk_query(prompt=full_prompt, options=options):
                if isinstance(message, _AssistantMessage):
                    for block in message.content:
                        if isinstance(block, _TextBlock):
                            content_parts.append(block.text)
                            if stream and stream_callback:
                                stream_callback(block.text)

                elif isinstance(message, _ResultMessage):
                    result_data = message
                    if message.is_error:
                        error_text = message.result or "Error desconocido del Agent SDK"
                        self._handle_sdk_error(error_text)

        except ClaudeMaxError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise AuthenticationError(
                    "No se pudo autenticar con la suscripcion de Claude Code Max. "
                    "Verifica que claude esta instalado y autenticado: claude setup-token"
                ) from e
            if "rate" in error_msg.lower() and "limit" in error_msg.lower():
                raise RateLimitError(error_msg) from e
            raise CompletionError(f"Error en la completacion: {error_msg}") from e

        latency_ms = (time.time() - start_time) * 1000
        content = "".join(content_parts)

        # Si json_mode, limpiar posibles bloques markdown que envuelvan el JSON
        if json_mode:
            content = self._strip_markdown_json(content)

        # Extraer metricas de uso
        usage = {}
        session_id = None
        cost_usd = None
        if result_data:
            session_id = result_data.session_id
            cost_usd = getattr(result_data, "total_cost_usd", None)
            usage = getattr(result_data, "usage", {}) or {}

        tokens_in = usage.get("input_tokens", estimate_tokens(full_prompt))
        tokens_out = usage.get("output_tokens", estimate_tokens(content))

        return CompletionResponse(
            content=content,
            model=effective_model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            session_id=session_id,
            raw_usage=usage if usage else None,
        )

    async def abatch_complete(
        self,
        prompts: list[str | dict],
        system: str | None = None,
        model: str | None = None,
        max_concurrency: int = 5,
        progress_callback: Callable[[int, int], None] | None = None,
        retry_on_error: bool = True,
        max_retries: int = 3,
    ) -> list[CompletionResponse]:
        """Version asincrona de batch_complete(). Mismos parametros y comportamiento."""
        total = len(prompts)
        results: list[CompletionResponse | None] = [None] * total
        completed_count = 0
        semaphore = asyncio.Semaphore(max_concurrency)

        # Normalizar todos los prompts a dict
        normalized = []
        for p in prompts:
            norm = normalize_prompt_input(p, shared_system=system)
            if model and "model" not in norm:
                norm["model"] = model
            normalized.append(norm)

        async def process_one(index: int, prompt_data: dict) -> None:
            nonlocal completed_count
            async with semaphore:
                last_error: str | None = None
                attempts = max_retries if retry_on_error else 1

                for attempt in range(attempts):
                    try:
                        result = await self.acomplete(**prompt_data)
                        results[index] = result
                        completed_count += 1
                        if progress_callback:
                            progress_callback(completed_count, total)
                        return
                    except RateLimitError as e:
                        wait_time = (2 ** attempt) * 5
                        logger.warning(
                            f"Rate limit en prompt {index}, reintentando en {wait_time}s "
                            f"(intento {attempt + 1}/{attempts})"
                        )
                        last_error = str(e)
                        await asyncio.sleep(wait_time)
                    except ClaudeMaxError as e:
                        last_error = str(e)
                        if not retry_on_error:
                            break
                        wait_time = 2 ** attempt
                        logger.warning(
                            f"Error en prompt {index}: {e}. "
                            f"Reintentando en {wait_time}s (intento {attempt + 1}/{attempts})"
                        )
                        await asyncio.sleep(wait_time)

                # Todos los reintentos fallaron
                results[index] = CompletionResponse(
                    content="",
                    model=prompt_data.get("model", self.model),
                    error=last_error or "Error desconocido tras reintentos",
                )
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total)

        tasks = [process_one(i, p) for i, p in enumerate(normalized)]
        await asyncio.gather(*tasks)

        return results  # type: ignore[return-value]

    # =========================================================================
    # Metodos internos
    # =========================================================================

    def _build_prompt(
        self,
        prompt: str,
        images: list[str] | None,
        files: list[str] | None,
        json_mode: bool,
    ) -> str:
        """Construye el prompt completo con archivos, imagenes y/o instrucciones JSON."""
        parts = [prompt]

        # Inyectar contenido de archivos de texto como contexto
        if files:
            parts.append("\n\n--- ARCHIVOS ADJUNTOS ---")
            for file_path in files:
                path = validate_file_path(file_path)
                content = read_text_file(file_path)
                filename = path.name
                suffix = path.suffix.lower()
                # Detectar lenguaje para syntax highlighting en el prompt
                lang = suffix.lstrip(".") if suffix else ""
                parts.append(f"\n### {filename}\n```{lang}\n{content}\n```")
            parts.append("\n--- FIN ARCHIVOS ---")

        # Inyectar imagenes como referencias para Claude Code
        if images:
            parts.append("\n\nAnaliza las siguientes imagenes:")
            for img_path in images:
                abs_path = str(validate_image_path(img_path))
                parts.append(f"\n- Imagen: {abs_path}")
            parts.append(
                "\nPara cada imagen, lee el archivo y describe su contenido "
                "de forma detallada y precisa."
            )

        if json_mode:
            parts.append(
                "\n\nIMPORTANTE: Responde UNICAMENTE con JSON valido. "
                "No incluyas texto adicional, ni bloques de codigo markdown. "
                "Solo el JSON puro."
            )

        return "\n".join(parts)

    @staticmethod
    def _strip_markdown_json(content: str) -> str:
        """Limpia bloques markdown ```json ... ``` de la respuesta."""
        stripped = content.strip()
        # Patron: ```json\n{...}\n``` o ```\n{...}\n```
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Eliminar primera linea (```json o ```)
            lines = lines[1:]
            # Eliminar ultima linea si es ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    def _handle_sdk_error(self, error_text: str) -> None:
        """Clasifica y lanza la excepcion apropiada basandose en el error del SDK."""
        lower = error_text.lower()

        if "authentication" in lower or "unauthorized" in lower or "login" in lower:
            raise AuthenticationError(error_text)

        if "rate" in lower and "limit" in lower:
            raise RateLimitError(error_text)

        if "model" in lower and ("not available" in lower or "not found" in lower):
            from .exceptions import ModelNotAvailableError
            raise ModelNotAvailableError(error_text)

        raise CompletionError(error_text)

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """
        Ejecuta una coroutine de forma sincrona.

        Maneja el caso de estar dentro de un event loop existente
        (e.g., Jupyter notebooks) usando un thread separado.
        """
        try:
            asyncio.get_running_loop()
            # Ya hay un loop corriendo - usar thread para evitar deadlock
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No hay loop corriendo - caso normal
            return asyncio.run(coro)
