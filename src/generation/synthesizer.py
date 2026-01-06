"""
Response Synthesizer - Genera respuestas usando LLMs.

Soporta múltiples proveedores:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude 3.5 Sonnet)
- Ollama (modelos locales)
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import os

from ..retrieval.fusion import RetrievalResult
from .prompt_builder import PromptBuilder, QueryType
from ..utils.cost_tracker import get_tracker

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """Respuesta generada con metadatos."""
    content: str
    query: str
    query_type: str
    sources_used: List[str]  # chunk_ids usados
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    abstained: bool = False  # Si el sistema se abstuvo de responder
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "query": self.query,
            "query_type": self.query_type,
            "sources_used": self.sources_used,
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "abstained": self.abstained
        }


class ResponseSynthesizer:
    """
    Sintetizador de respuestas usando LLMs.
    
    Características:
    - Múltiples proveedores (OpenAI, Anthropic, Ollama)
    - Streaming opcional
    - Control de tokens y costes
    - Fallback entre modelos
    """
    
    # Configuración de modelos (actualizado enero 2026)
    MODEL_CONFIGS = {
        "gpt-4.1": {
            "provider": "openai",
            "max_tokens": 128000,
            "cost_per_1k_input": 0.002,
            "cost_per_1k_output": 0.008
        },
        "gpt-4.1-mini": {
            "provider": "openai",
            "max_tokens": 128000,
            "cost_per_1k_input": 0.0004,
            "cost_per_1k_output": 0.0016
        },
        "gpt-4o": {
            "provider": "openai",
            "max_tokens": 128000,
            "cost_per_1k_input": 0.0025,
            "cost_per_1k_output": 0.01
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "max_tokens": 128000,
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006
        },
        "claude-sonnet-4-5-20250929": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015
        },
        "claude-sonnet-4-5": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015
        },
        "claude-sonnet-4-20250514": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015
        },
        "claude-3-5-haiku-20241022": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "cost_per_1k_input": 0.0008,
            "cost_per_1k_output": 0.004
        }
    }
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.3,
        max_output_tokens: int = 2000,
        fallback_model: Optional[str] = None
    ):
        """
        Args:
            model: Modelo a usar
            temperature: Temperatura de sampling
            max_output_tokens: Máximo tokens de salida
            fallback_model: Modelo alternativo si el principal falla
        """
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.fallback_model = fallback_model
        
        self._clients = {}
        self.prompt_builder = PromptBuilder()
        
        # Validar modelo
        if model not in self.MODEL_CONFIGS:
            logger.warning(f"Modelo {model} no reconocido, usando configuración genérica")
    
    def _get_client(self, provider: str):
        """Obtiene cliente para el proveedor."""
        if provider in self._clients:
            return self._clients[provider]
        
        if provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY no configurada")
            self._clients[provider] = OpenAI(api_key=api_key)
            
        elif provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY no configurada")
            self._clients[provider] = Anthropic(api_key=api_key)
            
        elif provider == "ollama":
            # Ollama usa API REST local
            import httpx
            self._clients[provider] = httpx.Client(
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
            )
        
        return self._clients[provider]
    
    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        query_type: Optional[QueryType] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> GeneratedResponse:
        """
        Genera respuesta basada en resultados de retrieval.
        
        Args:
            query: Consulta del usuario
            results: Resultados del retrieval
            query_type: Tipo de query (auto-detectado si None)
            stream: Si usar streaming
            stream_callback: Función callback para streaming
            
        Returns:
            Respuesta generada con metadatos
        """
        import time
        start_time = time.time()
        
        # Construir prompt
        prompt = self.prompt_builder.build_prompt(
            query=query,
            results=results,
            query_type=query_type
        )
        
        # Detectar tipo si no se proporciona
        if query_type is None:
            query_type = self.prompt_builder.detect_query_type(query)
        
        # Obtener configuración del modelo
        config = self.MODEL_CONFIGS.get(self.model, {})
        provider = config.get("provider", "openai")
        
        try:
            if provider == "openai":
                response, tokens_in, tokens_out = self._generate_openai(
                    prompt, stream, stream_callback
                )
            elif provider == "anthropic":
                response, tokens_in, tokens_out = self._generate_anthropic(
                    prompt, stream, stream_callback
                )
            elif provider == "ollama":
                response, tokens_in, tokens_out = self._generate_ollama(
                    prompt, stream, stream_callback
                )
            else:
                raise ValueError(f"Proveedor no soportado: {provider}")
                
        except Exception as e:
            logger.error(f"Error con {self.model}: {e}")
            
            if self.fallback_model:
                logger.info(f"Usando fallback: {self.fallback_model}")
                original_model = self.model
                self.model = self.fallback_model
                try:
                    return self.generate(query, results, query_type, stream, stream_callback)
                finally:
                    self.model = original_model
            raise
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Registrar coste (QUERY)
        tracker = get_tracker()
        tracker.record_generation(
            model=self.model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            query=query
        )
        
        # Construir respuesta
        generated = GeneratedResponse(
            content=response,
            query=query,
            query_type=query_type.value if isinstance(query_type, QueryType) else str(query_type),
            sources_used=[r.chunk_id for r in results],
            model=self.model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            metadata={
                "temperature": self.temperature,
                "provider": provider
            }
        )
        
        logger.info(
            f"Respuesta generada: {tokens_out} tokens, "
            f"{latency_ms:.0f}ms, modelo={self.model}"
        )
        
        return generated
    
    def _generate_openai(
        self,
        prompt: Dict[str, str],
        stream: bool,
        callback: Optional[Callable]
    ) -> tuple:
        """Genera con OpenAI."""
        client = self._get_client("openai")
        
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
        
        if stream and callback:
            response_text = ""
            stream_response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                stream=True
            )
            
            for chunk in stream_response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    response_text += text
                    callback(text)
            
            # Estimar tokens
            tokens_in = len(prompt["system"] + prompt["user"]) // 4
            tokens_out = len(response_text) // 4
            
            return response_text, tokens_in, tokens_out
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens
            )
            
            return (
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
    
    def _generate_anthropic(
        self,
        prompt: Dict[str, str],
        stream: bool,
        callback: Optional[Callable]
    ) -> tuple:
        """Genera con Anthropic."""
        client = self._get_client("anthropic")
        
        if stream and callback:
            response_text = ""
            
            with client.messages.stream(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}]
            ) as stream_response:
                for text in stream_response.text_stream:
                    response_text += text
                    callback(text)
            
            # Obtener uso de tokens del mensaje final
            final_message = stream_response.get_final_message()
            tokens_in = final_message.usage.input_tokens
            tokens_out = final_message.usage.output_tokens
            
            return response_text, tokens_in, tokens_out
        else:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}]
            )
            
            return (
                response.content[0].text,
                response.usage.input_tokens,
                response.usage.output_tokens
            )
    
    def _generate_ollama(
        self,
        prompt: Dict[str, str],
        stream: bool,
        callback: Optional[Callable]
    ) -> tuple:
        """Genera con Ollama (modelo local)."""
        client = self._get_client("ollama")
        
        full_prompt = f"{prompt['system']}\n\n{prompt['user']}"
        
        if stream and callback:
            response_text = ""
            
            with client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_output_tokens
                    }
                }
            ) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            text = data["response"]
                            response_text += text
                            callback(text)
            
            tokens_in = len(full_prompt) // 4
            tokens_out = len(response_text) // 4
            
            return response_text, tokens_in, tokens_out
        else:
            response = client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_output_tokens
                    }
                }
            )
            
            data = response.json()
            return (
                data.get("response", ""),
                data.get("prompt_eval_count", len(full_prompt) // 4),
                data.get("eval_count", len(data.get("response", "")) // 4)
            )
    
    def estimate_cost(
        self,
        tokens_input: int,
        tokens_output: int
    ) -> float:
        """
        Estima el coste de una generación.
        
        Args:
            tokens_input: Tokens de entrada
            tokens_output: Tokens de salida
            
        Returns:
            Coste estimado en USD
        """
        config = self.MODEL_CONFIGS.get(self.model, {})
        
        cost_in = (tokens_input / 1000) * config.get("cost_per_1k_input", 0)
        cost_out = (tokens_output / 1000) * config.get("cost_per_1k_output", 0)
        
        return cost_in + cost_out
