"""
Tests para claude_max_client.

Tests unitarios (con mocks, sin suscripcion):
    pytest tests/test_claude_max_client.py -v -k "not integration"

Tests de integracion (requieren suscripcion Claude Max activa):
    pytest tests/test_claude_max_client.py -v -m integration
"""

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.claude_max_client import (
    ClaudeMaxClient,
    CompletionResponse,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    ClaudeMaxError,
    AuthenticationError,
    RateLimitError,
    ModelNotAvailableError,
    ImageProcessingError,
    FileProcessingError,
    CompletionError,
)
from src.claude_max_client.utils import (
    validate_image_path,
    validate_file_path,
    read_text_file,
    is_text_file,
    is_image_file,
    encode_image_to_base64,
    estimate_tokens,
    normalize_prompt_input,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Cliente con configuracion por defecto."""
    return ClaudeMaxClient()


@pytest.fixture
def sample_image(tmp_path):
    """Crea una imagen JPG de prueba."""
    img_path = tmp_path / "test_image.jpg"
    # Cabecera JFIF minima valida
    img_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )
    return str(img_path)


@pytest.fixture
def sample_png(tmp_path):
    """Crea una imagen PNG de prueba."""
    img_path = tmp_path / "test_image.png"
    # Cabecera PNG minima
    img_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    )
    return str(img_path)


# ============================================================================
# Tests: CompletionResponse
# ============================================================================

class TestCompletionResponse:
    def test_creation_basic(self):
        r = CompletionResponse(content="Hola", model="claude-sonnet-4-5-20250929")
        assert r.content == "Hola"
        assert r.model == "claude-sonnet-4-5-20250929"
        assert r.tokens_input == 0
        assert r.tokens_output == 0
        assert r.error is None

    def test_ok_property_success(self):
        r = CompletionResponse(content="respuesta", model="test")
        assert r.ok is True

    def test_ok_property_empty_content(self):
        r = CompletionResponse(content="", model="test")
        assert r.ok is False

    def test_ok_property_with_error(self):
        r = CompletionResponse(content="parcial", model="test", error="fallo")
        assert r.ok is False

    def test_to_dict(self):
        r = CompletionResponse(
            content="test",
            model="claude-sonnet-4-5-20250929",
            tokens_input=100,
            tokens_output=50,
            cost_usd=0.0,
            latency_ms=150.5,
        )
        d = r.to_dict()
        assert d["content"] == "test"
        assert d["model"] == "claude-sonnet-4-5-20250929"
        assert d["tokens_input"] == 100
        assert d["tokens_output"] == 50
        assert d["cost_usd"] == 0.0
        assert d["latency_ms"] == 150.5

    def test_to_dict_has_all_fields(self):
        r = CompletionResponse(content="x", model="m")
        d = r.to_dict()
        expected_keys = {
            "content", "model", "tokens_input", "tokens_output",
            "cost_usd", "latency_ms", "session_id", "raw_usage", "error",
        }
        assert set(d.keys()) == expected_keys


# ============================================================================
# Tests: ClaudeMaxClient inicializacion
# ============================================================================

class TestClientInit:
    def test_default_values(self, client):
        assert client.model == DEFAULT_MODEL
        assert client.default_max_tokens == 4096
        assert client.default_temperature == 0.3

    def test_custom_values(self):
        c = ClaudeMaxClient(
            model="claude-opus-4-5-20251101",
            default_max_tokens=8192,
            default_temperature=0.7,
        )
        assert c.model == "claude-opus-4-5-20251101"
        assert c.default_max_tokens == 8192
        assert c.default_temperature == 0.7

    def test_unknown_model_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ClaudeMaxClient(model="modelo-inexistente")
        assert "no esta en la lista" in caplog.text


# ============================================================================
# Tests: _build_prompt
# ============================================================================

class TestBuildPrompt:
    def test_basic_prompt(self, client):
        result = client._build_prompt("Hola mundo", None, None, False)
        assert result == "Hola mundo"

    def test_with_json_mode(self, client):
        result = client._build_prompt("Dame datos", None, None, True)
        assert "JSON valido" in result
        assert "Dame datos" in result

    def test_with_images(self, client, sample_image):
        result = client._build_prompt("Describe", [sample_image], None, False)
        assert "Describe" in result
        assert "Imagen:" in result
        assert sample_image.split("/")[-1] in result

    def test_with_images_and_json(self, client, sample_image):
        result = client._build_prompt("Analiza", [sample_image], None, True)
        assert "Analiza" in result
        assert "Imagen:" in result
        assert "JSON valido" in result

    def test_invalid_image_raises(self, client):
        with pytest.raises(ImageProcessingError):
            client._build_prompt("test", ["/no/existe.jpg"], None, False)

    def test_with_files(self, client, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("print('hello')")
        result = client._build_prompt("Analiza este codigo", None, [str(f)], False)
        assert "Analiza este codigo" in result
        assert "ARCHIVOS ADJUNTOS" in result
        assert "code.py" in result
        assert "print('hello')" in result
        assert "FIN ARCHIVOS" in result

    def test_with_files_and_json(self, client, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        result = client._build_prompt("Resume", None, [str(f)], True)
        assert "Resume" in result
        assert "data.json" in result
        assert "JSON valido" in result

    def test_with_images_and_files(self, client, sample_image, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Notas de ejemplo")
        result = client._build_prompt("Analiza todo", [sample_image], [str(f)], False)
        assert "Analiza todo" in result
        assert "ARCHIVOS ADJUNTOS" in result
        assert "notes.txt" in result
        assert "Imagen:" in result

    def test_invalid_file_raises(self, client):
        with pytest.raises(FileProcessingError):
            client._build_prompt("test", None, ["/no/existe.py"], False)


# ============================================================================
# Tests: utils - validate_image_path
# ============================================================================

class TestValidateImagePath:
    def test_valid_jpg(self, sample_image):
        path = validate_image_path(sample_image)
        assert path.exists()
        assert path.suffix == ".jpg"

    def test_valid_png(self, sample_png):
        path = validate_image_path(sample_png)
        assert path.exists()

    def test_nonexistent_file(self):
        with pytest.raises(ImageProcessingError, match="no encontrada"):
            validate_image_path("/tmp/no_existe_12345.jpg")

    def test_unsupported_type(self, tmp_path):
        txt = tmp_path / "file.txt"
        txt.write_text("no es imagen")
        with pytest.raises(ImageProcessingError, match="no soportado"):
            validate_image_path(str(txt))

    def test_directory_not_file(self, tmp_path):
        with pytest.raises(ImageProcessingError, match="no es un archivo"):
            validate_image_path(str(tmp_path))


# ============================================================================
# Tests: utils - validate_file_path
# ============================================================================

class TestValidateFilePath:
    def test_valid_python_file(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("print('hello')")
        path = validate_file_path(str(f))
        assert path.exists()
        assert path.suffix == ".py"

    def test_valid_text_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("notas")
        path = validate_file_path(str(f))
        assert path.exists()

    def test_nonexistent_file(self):
        with pytest.raises(FileProcessingError, match="no encontrado"):
            validate_file_path("/tmp/no_existe_12345.py")

    def test_directory_not_file(self, tmp_path):
        with pytest.raises(FileProcessingError, match="no es un archivo"):
            validate_file_path(str(tmp_path))

    def test_binary_file_rejected(self, tmp_path):
        f = tmp_path / "binary.exe"
        f.write_bytes(b"\x00\x01\x02")
        with pytest.raises(FileProcessingError, match="binario"):
            validate_file_path(str(f))

    def test_pdf_rejected(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        with pytest.raises(FileProcessingError, match="binario"):
            validate_file_path(str(f))

    def test_file_too_large(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (1_048_576 + 1))
        with pytest.raises(FileProcessingError, match="grande"):
            validate_file_path(str(f))

    def test_file_at_size_limit(self, tmp_path):
        f = tmp_path / "exact.txt"
        f.write_bytes(b"x" * 1_048_576)
        path = validate_file_path(str(f))
        assert path.exists()

    def test_no_extension_accepted(self, tmp_path):
        f = tmp_path / "Makefile"
        f.write_text("all: build")
        path = validate_file_path(str(f))
        assert path.exists()


# ============================================================================
# Tests: utils - read_text_file
# ============================================================================

class TestReadTextFile:
    def test_read_utf8(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("contenido UTF-8 con acentos: café", encoding="utf-8")
        content = read_text_file(str(f))
        assert "café" in content

    def test_read_latin1_fallback(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes("contenido latin-1: café".encode("latin-1"))
        content = read_text_file(str(f))
        assert "caf" in content

    def test_read_nonexistent_raises(self):
        with pytest.raises(FileProcessingError):
            read_text_file("/tmp/no_existe_xyz.txt")

    def test_read_code_file(self, tmp_path):
        f = tmp_path / "app.js"
        code = "function hello() { return 'world'; }"
        f.write_text(code)
        content = read_text_file(str(f))
        assert content == code


# ============================================================================
# Tests: utils - is_text_file / is_image_file
# ============================================================================

class TestFileTypeChecks:
    def test_is_text_file_python(self):
        assert is_text_file("script.py") is True

    def test_is_text_file_markdown(self):
        assert is_text_file("README.md") is True

    def test_is_text_file_json(self):
        assert is_text_file("config.json") is True

    def test_is_text_file_no_extension(self):
        assert is_text_file("Makefile") is True

    def test_is_text_file_binary(self):
        assert is_text_file("image.jpg") is False

    def test_is_image_file_jpg(self):
        assert is_image_file("photo.jpg") is True

    def test_is_image_file_png(self):
        assert is_image_file("screenshot.png") is True

    def test_is_image_file_txt(self):
        assert is_image_file("notes.txt") is False

    def test_is_image_file_webp(self):
        assert is_image_file("image.webp") is True


# ============================================================================
# Tests: utils - encode_image_to_base64
# ============================================================================

class TestEncodeImage:
    def test_encode_jpg(self, sample_image):
        b64, media_type = encode_image_to_base64(sample_image)
        assert len(b64) > 0
        assert media_type == "image/jpeg"

    def test_encode_png(self, sample_png):
        b64, media_type = encode_image_to_base64(sample_png)
        assert len(b64) > 0
        assert media_type == "image/png"


# ============================================================================
# Tests: utils - estimate_tokens
# ============================================================================

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("hola") == 1

    def test_longer_string(self):
        text = "a" * 400
        result = estimate_tokens(text)
        assert result == 133  # 400 // 3

    def test_realistic_text(self):
        text = "Un qubit es la unidad basica de informacion cuantica."
        result = estimate_tokens(text)
        assert 10 <= result <= 20


# ============================================================================
# Tests: utils - normalize_prompt_input
# ============================================================================

class TestNormalizePromptInput:
    def test_string_input(self):
        result = normalize_prompt_input("Hola")
        assert result == {"prompt": "Hola"}

    def test_string_with_shared_system(self):
        result = normalize_prompt_input("Hola", shared_system="Eres experto")
        assert result["prompt"] == "Hola"
        assert result["system"] == "Eres experto"

    def test_dict_input(self):
        result = normalize_prompt_input({"prompt": "Hola", "system": "Custom"})
        assert result["prompt"] == "Hola"
        assert result["system"] == "Custom"

    def test_dict_preserves_own_system(self):
        result = normalize_prompt_input(
            {"prompt": "Hola", "system": "Propio"},
            shared_system="Compartido",
        )
        assert result["system"] == "Propio"

    def test_dict_without_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt"):
            normalize_prompt_input({"system": "sin prompt"})

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="no soportado"):
            normalize_prompt_input(42)  # type: ignore


# ============================================================================
# Tests: _handle_sdk_error
# ============================================================================

class TestHandleSDKError:
    def test_authentication_error(self, client):
        with pytest.raises(AuthenticationError):
            client._handle_sdk_error("Authentication failed: unauthorized")

    def test_rate_limit_error(self, client):
        with pytest.raises(RateLimitError):
            client._handle_sdk_error("Rate limit exceeded")

    def test_model_not_available(self, client):
        with pytest.raises(ModelNotAvailableError):
            client._handle_sdk_error("Model not available: claude-opus")

    def test_generic_error(self, client):
        with pytest.raises(CompletionError):
            client._handle_sdk_error("Algo salio mal")


# ============================================================================
# Tests: AVAILABLE_MODELS
# ============================================================================

class TestAvailableModels:
    def test_default_model_in_list(self):
        assert DEFAULT_MODEL in AVAILABLE_MODELS

    def test_all_models_have_required_fields(self):
        for model_id, info in AVAILABLE_MODELS.items():
            assert "name" in info
            assert "context_window" in info
            assert "description" in info

    def test_opus_available(self):
        assert "claude-opus-4-5-20251101" in AVAILABLE_MODELS

    def test_sonnet_available(self):
        assert "claude-sonnet-4-5-20250929" in AVAILABLE_MODELS

    def test_haiku_available(self):
        assert "claude-haiku-4-5-20251001" in AVAILABLE_MODELS


# ============================================================================
# Tests: Mocked acomplete
# ============================================================================

class TestAcompleteWithMock:
    """Tests de acomplete usando mocks del Agent SDK."""

    @pytest.fixture
    def mock_sdk(self):
        """Configura mocks para el Claude Agent SDK."""
        # Crear clases mock que simulan los tipos del SDK
        @dataclass
        class MockTextBlock:
            text: str

        @dataclass
        class MockAssistantMessage:
            content: list

        @dataclass
        class MockResultMessage:
            session_id: str = "test-session"
            total_cost_usd: float = 0.0
            duration_ms: float = 100.0
            is_error: bool = False
            result: str = None
            usage: dict = None

            def __post_init__(self):
                if self.usage is None:
                    self.usage = {"input_tokens": 50, "output_tokens": 30}

        return MockTextBlock, MockAssistantMessage, MockResultMessage

    @pytest.mark.asyncio
    async def test_simple_completion(self, client, mock_sdk):
        MockTextBlock, MockAssistantMessage, MockResultMessage = mock_sdk

        async def mock_query(prompt, options=None):
            yield MockAssistantMessage(content=[MockTextBlock(text="Respuesta de test")])
            yield MockResultMessage()

        with patch("src.claude_max_client.client._sdk_query", mock_query), \
             patch("src.claude_max_client.client._AssistantMessage", MockAssistantMessage), \
             patch("src.claude_max_client.client._TextBlock", MockTextBlock), \
             patch("src.claude_max_client.client._ResultMessage", MockResultMessage):
            response = await client.acomplete(prompt="Pregunta de test")

        assert response.content == "Respuesta de test"
        assert response.ok is True
        assert response.tokens_input == 50
        assert response.tokens_output == 30
        assert response.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_streaming_callback(self, client, mock_sdk):
        MockTextBlock, MockAssistantMessage, MockResultMessage = mock_sdk
        received_chunks = []

        async def mock_query(prompt, options=None):
            yield MockAssistantMessage(content=[MockTextBlock(text="Parte 1 ")])
            yield MockAssistantMessage(content=[MockTextBlock(text="Parte 2")])
            yield MockResultMessage()

        with patch("src.claude_max_client.client._sdk_query", mock_query), \
             patch("src.claude_max_client.client._AssistantMessage", MockAssistantMessage), \
             patch("src.claude_max_client.client._TextBlock", MockTextBlock), \
             patch("src.claude_max_client.client._ResultMessage", MockResultMessage):
            response = await client.acomplete(
                prompt="Test streaming",
                stream=True,
                stream_callback=lambda t: received_chunks.append(t),
            )

        assert response.content == "Parte 1 Parte 2"
        assert received_chunks == ["Parte 1 ", "Parte 2"]

    @pytest.mark.asyncio
    async def test_error_result_raises(self, client, mock_sdk):
        MockTextBlock, MockAssistantMessage, MockResultMessage = mock_sdk

        async def mock_query(prompt, options=None):
            yield MockResultMessage(is_error=True, result="Authentication failed: unauthorized")

        with patch("src.claude_max_client.client._sdk_query", mock_query), \
             patch("src.claude_max_client.client._AssistantMessage", MockAssistantMessage), \
             patch("src.claude_max_client.client._TextBlock", MockTextBlock), \
             patch("src.claude_max_client.client._ResultMessage", MockResultMessage):
            with pytest.raises(AuthenticationError):
                await client.acomplete(prompt="Test error")

    @pytest.mark.asyncio
    async def test_model_override(self, client, mock_sdk):
        MockTextBlock, MockAssistantMessage, MockResultMessage = mock_sdk
        captured_options = {}

        async def mock_query(prompt, options=None):
            captured_options["model"] = options.model if options else None
            yield MockAssistantMessage(content=[MockTextBlock(text="ok")])
            yield MockResultMessage()

        # Mock ClaudeAgentOptions para capturar el modelo
        mock_options_cls = MagicMock()
        mock_options_cls.return_value = MagicMock(model="claude-opus-4-5-20251101")

        with patch("src.claude_max_client.client._sdk_query", mock_query), \
             patch("src.claude_max_client.client._AssistantMessage", MockAssistantMessage), \
             patch("src.claude_max_client.client._TextBlock", MockTextBlock), \
             patch("src.claude_max_client.client._ResultMessage", MockResultMessage), \
             patch("src.claude_max_client.client._ClaudeAgentOptions", mock_options_cls):
            response = await client.acomplete(
                prompt="Test",
                model="claude-opus-4-5-20251101",
            )

        assert response.content == "ok"
        # Verificar que se paso el modelo correcto al constructor de options
        mock_options_cls.assert_called_once()
        call_kwargs = mock_options_cls.call_args
        assert call_kwargs.kwargs.get("model") == "claude-opus-4-5-20251101"


# ============================================================================
# Tests: Batch processing con mocks
# ============================================================================

class TestBatchCompleteWithMock:
    @pytest.mark.asyncio
    async def test_batch_basic(self, client):
        """Test batch con acomplete mockeado."""
        call_count = 0

        async def mock_acomplete(**kwargs):
            nonlocal call_count
            call_count += 1
            return CompletionResponse(
                content=f"Respuesta {call_count}",
                model="test",
                tokens_input=10,
                tokens_output=5,
            )

        with patch.object(client, "acomplete", side_effect=mock_acomplete):
            results = await client.abatch_complete(
                prompts=["p1", "p2", "p3"],
                max_concurrency=2,
            )

        assert len(results) == 3
        assert all(r.ok for r in results)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_batch_progress_callback(self, client):
        """Test que el progress callback se invoca correctamente."""
        progress_calls = []

        async def mock_acomplete(**kwargs):
            return CompletionResponse(content="ok", model="test")

        with patch.object(client, "acomplete", side_effect=mock_acomplete):
            await client.abatch_complete(
                prompts=["a", "b"],
                progress_callback=lambda done, total: progress_calls.append((done, total)),
            )

        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)

    @pytest.mark.asyncio
    async def test_batch_handles_errors(self, client):
        """Test que los errores en batch no interrumpen el procesamiento."""
        call_idx = 0

        async def mock_acomplete(**kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 2:
                raise CompletionError("Error simulado")
            return CompletionResponse(content=f"ok-{call_idx}", model="test")

        with patch.object(client, "acomplete", side_effect=mock_acomplete):
            results = await client.abatch_complete(
                prompts=["a", "b", "c"],
                retry_on_error=False,
            )

        assert len(results) == 3
        assert results[0].ok
        assert not results[1].ok
        assert "Error simulado" in results[1].error
        assert results[2].ok

    @pytest.mark.asyncio
    async def test_batch_dict_prompts(self, client):
        """Test batch con prompts en formato dict."""
        captured_kwargs = []

        async def mock_acomplete(**kwargs):
            captured_kwargs.append(kwargs)
            return CompletionResponse(content="ok", model="test")

        with patch.object(client, "acomplete", side_effect=mock_acomplete):
            await client.abatch_complete(
                prompts=[
                    {"prompt": "p1", "system": "sys1"},
                    "p2",
                ],
                system="shared_sys",
            )

        assert captured_kwargs[0]["system"] == "sys1"  # Propio
        assert captured_kwargs[1]["system"] == "shared_sys"  # Compartido


# ============================================================================
# Tests: Excepciones
# ============================================================================

class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(AuthenticationError, ClaudeMaxError)
        assert issubclass(RateLimitError, ClaudeMaxError)
        assert issubclass(ModelNotAvailableError, ClaudeMaxError)
        assert issubclass(ImageProcessingError, ClaudeMaxError)
        assert issubclass(FileProcessingError, ClaudeMaxError)
        assert issubclass(CompletionError, ClaudeMaxError)

    def test_rate_limit_retry_after(self):
        e = RateLimitError("limit", retry_after=30.0)
        assert e.retry_after == 30.0

    def test_rate_limit_no_retry(self):
        e = RateLimitError("limit")
        assert e.retry_after is None


# ============================================================================
# Tests de integracion (requieren suscripcion activa)
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    """
    Tests que realizan llamadas reales al Agent SDK con la suscripcion Max.

    Ejecutar con: pytest tests/test_claude_max_client.py -v -m integration
    """

    def test_authentication_check(self):
        client = ClaudeMaxClient()
        assert client.is_authenticated(), (
            "La suscripcion de Claude Code Max no esta activa. "
            "Ejecuta 'claude setup-token' para autenticarte."
        )

    def test_simple_completion(self):
        client = ClaudeMaxClient(model="claude-sonnet-4-5-20250929")
        response = client.complete(
            prompt="Responde unicamente con la palabra 'funciona', sin nada mas.",
            system="Eres un asistente de testing. Sigue las instrucciones al pie de la letra.",
        )
        assert response.ok
        assert response.content is not None
        assert len(response.content.strip()) > 0
        assert response.latency_ms > 0
        assert "funciona" in response.content.lower()

    def test_json_mode(self):
        client = ClaudeMaxClient()
        response = client.complete(
            prompt='Devuelve un JSON con la clave "status" y valor "ok".',
            json_mode=True,
        )
        assert response.ok
        import json
        data = json.loads(response.content)
        assert data["status"] == "ok"

    def test_streaming(self):
        chunks = []
        client = ClaudeMaxClient()
        response = client.complete(
            prompt="Cuenta del 1 al 3, separados por comas.",
            stream=True,
            stream_callback=lambda t: chunks.append(t),
        )
        assert response.ok
        assert len(chunks) > 0
        assert response.content is not None

    def test_model_selection_haiku(self):
        client = ClaudeMaxClient(model="claude-haiku-4-5-20251001")
        response = client.complete(prompt="Di 'hola'.")
        assert response.ok
        assert response.model == "claude-haiku-4-5-20251001"
