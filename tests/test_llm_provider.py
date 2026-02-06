"""
Tests para el adaptador LLM centralizado (src/llm_provider).

Tests unitarios (con mocks, sin suscripcion ni API keys):
    pytest tests/test_llm_provider.py -v -k "not integration"

Tests de integracion (requieren LLM_PROVIDER configurado):
    pytest tests/test_llm_provider.py -v -m integration
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.llm_provider._types import LLMResponse, LLMProviderError
from src.llm_provider._config import Provider, get_active_provider, get_defaults, PROVIDER_DEFAULTS


# ============================================================================
# Tests: LLMResponse
# ============================================================================

class TestLLMResponse:
    def test_ok_with_content(self):
        r = LLMResponse(content="Hola", model="test", provider="test")
        assert r.ok is True

    def test_not_ok_empty_content(self):
        r = LLMResponse(content="", model="test", provider="test")
        assert r.ok is False

    def test_not_ok_with_error(self):
        r = LLMResponse(content="parcial", model="test", provider="test", error="fallo")
        assert r.ok is False

    def test_defaults(self):
        r = LLMResponse(content="x", model="m", provider="p")
        assert r.tokens_input == 0
        assert r.tokens_output == 0
        assert r.latency_ms == 0.0
        assert r.error is None


# ============================================================================
# Tests: Provider config
# ============================================================================

class TestProviderConfig:
    def test_default_provider_is_claude_max(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = get_active_provider()
            assert provider == Provider.CLAUDE_MAX

    def test_provider_openai_api(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai_api"}):
            provider = get_active_provider()
            assert provider == Provider.OPENAI_API

    def test_provider_anthropic_api(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic_api"}):
            provider = get_active_provider()
            assert provider == Provider.ANTHROPIC_API

    def test_provider_case_insensitive(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "CLAUDE_MAX"}):
            provider = get_active_provider()
            assert provider == Provider.CLAUDE_MAX

    def test_invalid_provider_raises(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "invalid_provider"}):
            with pytest.raises(LLMProviderError, match="no reconocido"):
                get_active_provider()

    def test_defaults_claude_max(self):
        defaults = get_defaults(Provider.CLAUDE_MAX)
        assert defaults["model"] == "claude-opus-4-5-20251101"
        assert "temperature" in defaults
        assert "max_tokens" in defaults

    def test_defaults_openai(self):
        defaults = get_defaults(Provider.OPENAI_API)
        assert defaults["model"] == "gpt-4.1-mini"

    def test_defaults_anthropic(self):
        defaults = get_defaults(Provider.ANTHROPIC_API)
        assert defaults["model"] == "claude-sonnet-4-5-20250929"

    def test_all_providers_have_defaults(self):
        for provider in Provider:
            defaults = get_defaults(provider)
            assert "model" in defaults
            assert "temperature" in defaults
            assert "max_tokens" in defaults


# ============================================================================
# Tests: Backend dispatch
# ============================================================================

class TestBackendDispatch:
    def test_dispatch_claude_max(self):
        mock_response = LLMResponse(
            content="Respuesta mock",
            model="claude-opus-4-5-20251101",
            provider="claude_max",
        )
        mock_fn = MagicMock(return_value=mock_response)

        from src.llm_provider._backends import dispatch, _DISPATCH
        with patch.dict(_DISPATCH, {Provider.CLAUDE_MAX: mock_fn}):
            result = dispatch(
                provider=Provider.CLAUDE_MAX,
                prompt="test",
                system=None,
                temperature=0.3,
                max_tokens=4096,
                json_mode=False,
                stream=False,
                stream_callback=None,
            )

        assert result.content == "Respuesta mock"
        assert result.provider == "claude_max"
        mock_fn.assert_called_once()

    def test_dispatch_openai_api(self):
        mock_response = LLMResponse(
            content="OpenAI response",
            model="gpt-4.1-mini",
            provider="openai_api",
        )
        mock_fn = MagicMock(return_value=mock_response)

        from src.llm_provider._backends import dispatch, _DISPATCH
        with patch.dict(_DISPATCH, {Provider.OPENAI_API: mock_fn}):
            result = dispatch(
                provider=Provider.OPENAI_API,
                prompt="test",
                system=None,
                temperature=0.1,
                max_tokens=4096,
                json_mode=False,
                stream=False,
                stream_callback=None,
            )

        assert result.provider == "openai_api"
        mock_fn.assert_called_once()

    def test_dispatch_anthropic_api(self):
        mock_response = LLMResponse(
            content="Anthropic response",
            model="claude-sonnet-4-5-20250929",
            provider="anthropic_api",
        )
        mock_fn = MagicMock(return_value=mock_response)

        from src.llm_provider._backends import dispatch, _DISPATCH
        with patch.dict(_DISPATCH, {Provider.ANTHROPIC_API: mock_fn}):
            result = dispatch(
                provider=Provider.ANTHROPIC_API,
                prompt="test",
                system=None,
                temperature=0.3,
                max_tokens=4096,
                json_mode=False,
                stream=False,
                stream_callback=None,
            )

        assert result.provider == "anthropic_api"
        mock_fn.assert_called_once()


# ============================================================================
# Tests: Error handling (no fallback)
# ============================================================================

class TestErrorHandling:
    def test_claude_max_error_propagates(self):
        mock_fn = MagicMock(side_effect=LLMProviderError("claude_max: Authentication failed"))

        from src.llm_provider._backends import dispatch, _DISPATCH
        with patch.dict(_DISPATCH, {Provider.CLAUDE_MAX: mock_fn}):
            with pytest.raises(LLMProviderError, match="claude_max"):
                dispatch(
                    provider=Provider.CLAUDE_MAX,
                    prompt="test",
                    system=None,
                    temperature=0.3,
                    max_tokens=4096,
                    json_mode=False,
                    stream=False,
                    stream_callback=None,
                )

    def test_openai_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            from src.llm_provider._backends import _complete_openai_api
            with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
                _complete_openai_api(
                    prompt="test",
                    system=None,
                    temperature=0.1,
                    max_tokens=100,
                    json_mode=False,
                    stream=False,
                    stream_callback=None,
                )

    def test_anthropic_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            from src.llm_provider._backends import _complete_anthropic_api
            with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
                _complete_anthropic_api(
                    prompt="test",
                    system=None,
                    temperature=0.3,
                    max_tokens=100,
                    json_mode=False,
                    stream=False,
                    stream_callback=None,
                )

    def test_error_is_llm_provider_error(self):
        assert issubclass(LLMProviderError, Exception)


# ============================================================================
# Tests: complete() function with mocked backend
# ============================================================================

class TestCompleteFunction:
    def test_complete_uses_defaults(self):
        mock_response = LLMResponse(
            content="ok", model="test-model", provider="claude_max"
        )

        with patch("src.llm_provider.dispatch", return_value=mock_response) as mock_dispatch:
            # Re-import to get fresh module with mocked dispatch
            from src.llm_provider import complete
            # Patch at module level
            with patch("src.llm_provider.dispatch", return_value=mock_response) as mock_d:
                result = complete(prompt="Hola")

        assert result.content == "ok"

    def test_complete_passes_overrides(self):
        mock_response = LLMResponse(
            content="json", model="test", provider="claude_max"
        )

        with patch("src.llm_provider.dispatch", return_value=mock_response) as mock_d:
            from src.llm_provider import complete
            with patch("src.llm_provider.dispatch", return_value=mock_response) as mock_d2:
                result = complete(
                    prompt="test",
                    system="sys",
                    temperature=0.9,
                    max_tokens=100,
                    json_mode=True,
                )

        assert result.content == "json"


# ============================================================================
# Tests: get_provider_name
# ============================================================================

class TestGetProviderName:
    def test_returns_string(self):
        from src.llm_provider import get_provider_name
        name = get_provider_name()
        assert isinstance(name, str)
        assert name in ["claude_max", "openai_api", "anthropic_api"]


# ============================================================================
# Tests: Migrated modules use llm_complete (not direct OpenAI/Anthropic)
# ============================================================================

class TestRouterMigration:
    def test_route_with_llm_calls_llm_complete(self):
        """Verifica que router usa llm_complete en lugar de OpenAI directo."""
        from src.agents.router import QueryRouter

        mock_response = LLMResponse(
            content='{"strategy": "HYBRID", "vector_weight": 0.5, "bm25_weight": 0.3, "graph_weight": 0.2, "top_k": 10, "sub_queries": [], "reasoning": "test"}',
            model="test", provider="claude_max",
        )
        router = QueryRouter(use_llm_router=True)

        with patch("src.llm_provider.complete", return_value=mock_response) as mock_llm:
            result = router.route("Que es un qubit?")

        # Verify it called llm_complete (not openai directly)
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs.get("json_mode") is True
        assert call_kwargs.kwargs.get("temperature") == 0.1

    def test_route_with_llm_falls_back_on_error(self):
        """Si LLM falla, router cae a heurísticas."""
        from src.agents.router import QueryRouter

        router = QueryRouter(use_llm_router=True)

        with patch("src.llm_provider.complete", side_effect=Exception("LLM down")):
            result = router.route("Que es un qubit?")

        # Should fall back to heuristics, not crash
        assert result is not None
        assert result.strategy is not None

    def test_router_no_openai_import(self):
        """Verifica que router.py no importa openai directamente."""
        import inspect
        from src.agents import router
        source = inspect.getsource(router)
        assert "from openai import" not in source
        assert "import openai" not in source


class TestPlannerMigration:
    def test_plan_with_llm_calls_llm_complete(self):
        from src.agents.planner import QueryPlanner

        mock_response = LLMResponse(
            content='{"steps": [{"step_id": 0, "step_type": "RETRIEVE", "query": "test", "depends_on": [], "params": {}}], "reasoning": "simple"}',
            model="test", provider="claude_max",
        )
        planner = QueryPlanner(use_llm_planner=True)

        with patch("src.llm_provider.complete", return_value=mock_response) as mock_llm:
            result = planner.plan("Que es superposición?")

        mock_llm.assert_called_once()
        assert result is not None
        assert len(result.steps) > 0

    def test_planner_no_openai_import(self):
        import inspect
        from src.agents import planner
        source = inspect.getsource(planner)
        assert "from openai import" not in source
        assert "import openai" not in source


class TestCriticMigration:
    def test_critic_no_openai_import(self):
        import inspect
        from src.agents import critic
        source = inspect.getsource(critic)
        assert "from openai import" not in source
        assert "import openai" not in source

    def test_critic_constructor_no_llm_model(self):
        """Verifica que ResponseCritic ya no acepta llm_model."""
        from src.agents.critic import ResponseCritic
        import inspect
        sig = inspect.signature(ResponseCritic.__init__)
        params = list(sig.parameters.keys())
        assert "llm_model" not in params
        assert "_llm_client" not in params


class TestHyDEMigration:
    def test_hyde_calls_llm_complete(self):
        from src.retrieval.hyde import HyDEExpander

        mock_response = LLMResponse(
            content="Quantum entanglement is a phenomenon...",
            model="test", provider="claude_max",
        )
        hyde = HyDEExpander()

        with patch("src.llm_provider.complete", return_value=mock_response) as mock_llm:
            result = hyde.generate_hypothetical("What is entanglement?")

        mock_llm.assert_called_once()
        assert result == "Quantum entanglement is a phenomenon..."

    def test_hyde_fallback_on_error(self):
        from src.retrieval.hyde import HyDEExpander

        hyde = HyDEExpander()
        with patch("src.llm_provider.complete", side_effect=Exception("fail")):
            result = hyde.generate_hypothetical("test query")

        # Falls back to original query
        assert result == "test query"

    def test_hyde_no_openai_import(self):
        import inspect
        from src.retrieval import hyde
        source = inspect.getsource(hyde)
        assert "from openai import" not in source
        assert "import openai" not in source

    def test_hyde_constructor_no_llm_client(self):
        from src.retrieval.hyde import HyDEExpander
        import inspect
        sig = inspect.signature(HyDEExpander.__init__)
        params = list(sig.parameters.keys())
        assert "llm_client" not in params


class TestGraphRetrieverMigration:
    def test_graph_retriever_no_openai_import(self):
        import inspect
        from src.retrieval import graph_retriever
        source = inspect.getsource(graph_retriever)
        assert "from openai import" not in source
        assert "import openai" not in source
        # But llm_complete should be used
        assert "llm_complete" in source


class TestSandboxMigration:
    def test_sandbox_generate_calls_llm_complete(self):
        from src.execution.sandbox import CodeSandbox

        mock_response = LLMResponse(
            content="print('hello')",
            model="test", provider="claude_max",
        )
        sandbox = CodeSandbox(timeout_seconds=5)

        with patch("src.llm_provider.complete", return_value=mock_response) as mock_llm:
            code, result = sandbox.generate_and_execute("Print hello")

        mock_llm.assert_called_once()
        assert code == "print('hello')"
        assert result.success is True

    def test_sandbox_no_openai_import(self):
        import inspect
        from src.execution import sandbox
        source = inspect.getsource(sandbox)
        assert "from openai import" not in source
        assert "import openai" not in source


class TestSynthesizerMigration:
    def test_synthesizer_no_model_configs(self):
        """Verifica que MODEL_CONFIGS fue eliminado."""
        from src.generation.synthesizer import ResponseSynthesizer
        assert not hasattr(ResponseSynthesizer, "MODEL_CONFIGS")

    def test_synthesizer_no_fallback_model(self):
        """Verifica que fallback_model fue eliminado."""
        import inspect
        from src.generation.synthesizer import ResponseSynthesizer
        sig = inspect.signature(ResponseSynthesizer.__init__)
        params = list(sig.parameters.keys())
        assert "fallback_model" not in params
        assert "model" not in params

    def test_synthesizer_constructor(self):
        from src.generation.synthesizer import ResponseSynthesizer
        syn = ResponseSynthesizer(temperature=0.5, max_output_tokens=1000)
        assert syn.temperature == 0.5
        assert syn.max_output_tokens == 1000

    def test_synthesizer_no_openai_import(self):
        import inspect
        from src.generation import synthesizer
        source = inspect.getsource(synthesizer)
        assert "from openai import" not in source
        assert "import openai" not in source
        assert "from anthropic import" not in source
        assert "import anthropic" not in source


class TestMetricsMigration:
    def test_evaluator_no_llm_client_param(self):
        from src.evaluation.metrics import RAGASEvaluator
        import inspect
        sig = inspect.signature(RAGASEvaluator.__init__)
        params = list(sig.parameters.keys())
        assert "llm_client" not in params

    def test_evaluator_call_llm_uses_adapter(self):
        from src.evaluation.metrics import RAGASEvaluator

        mock_response = LLMResponse(
            content="evaluation result", model="test", provider="claude_max",
        )
        evaluator = RAGASEvaluator()

        with patch("src.llm_provider.complete", return_value=mock_response) as mock_llm:
            result = evaluator._call_llm("test prompt", "system")

        mock_llm.assert_called_once()
        assert result == "evaluation result"

    def test_evaluator_no_openai_import(self):
        import inspect
        from src.evaluation import metrics
        source = inspect.getsource(metrics)
        assert "from openai import" not in source
        assert "import openai" not in source


class TestAskLibraryMigration:
    def test_ask_library_no_openai_import(self):
        import inspect
        from src.cli import ask_library
        source = inspect.getsource(ask_library)
        assert "import openai" not in source
        # from openai should not appear
        assert "from openai" not in source


class TestCostTrackerUpdate:
    def test_claude_opus_has_zero_cost(self):
        from src.utils.cost_tracker import CostTracker
        pricing = CostTracker.PRICING
        assert "claude-opus-4-5-20251101" in pricing
        assert pricing["claude-opus-4-5-20251101"]["input"] == 0.0
        assert pricing["claude-opus-4-5-20251101"]["output"] == 0.0


class TestNoDirectLLMCalls:
    """Verifica que ningún módulo fuera de llm_provider hace llamadas directas."""

    def test_no_chat_completions_create_outside_backends(self):
        """Solo _backends.py debe tener chat.completions.create."""
        import inspect

        modules_to_check = []
        from src.agents import router, planner, critic
        from src.retrieval import hyde, graph_retriever
        from src.execution import sandbox
        from src.generation import synthesizer
        from src.evaluation import metrics

        for mod in [router, planner, critic, hyde, graph_retriever,
                    sandbox, synthesizer, metrics]:
            source = inspect.getsource(mod)
            assert "chat.completions.create" not in source, \
                f"{mod.__name__} still has direct chat.completions.create call"

    def test_no_anthropic_messages_create_outside_backends(self):
        """Solo _backends.py debe tener messages.create."""
        import inspect
        from src.generation import synthesizer

        source = inspect.getsource(synthesizer)
        assert "messages.create" not in source


# ============================================================================
# Tests de integracion (requieren suscripcion/API activa)
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    def test_complete_basic(self):
        from src.llm_provider import complete
        response = complete(
            prompt="Responde unicamente con la palabra 'funciona'.",
            temperature=0.0,
            max_tokens=20,
        )
        assert response.ok
        assert len(response.content) > 0
        assert response.latency_ms > 0

    def test_complete_json_mode(self):
        from src.llm_provider import complete
        response = complete(
            prompt='Devuelve un JSON con la clave "status" y valor "ok".',
            json_mode=True,
            temperature=0.0,
            max_tokens=50,
        )
        assert response.ok
        data = json.loads(response.content)
        assert data["status"] == "ok"

    def test_complete_with_system(self):
        from src.llm_provider import complete
        response = complete(
            prompt="Que eres?",
            system="Eres un asistente de testing. Responde: soy un test.",
            temperature=0.0,
            max_tokens=50,
        )
        assert response.ok
        assert "test" in response.content.lower()
