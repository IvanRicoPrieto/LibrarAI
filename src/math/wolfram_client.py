"""
WolframClient — Cliente de Wolfram Alpha API para LibrarAI.

Usado como fallback cuando SymPy no puede resolver un problema.
Corre FUERA del sandbox (necesita acceso a red).

Usa el endpoint LLM-optimized de Wolfram Alpha:
https://api.wolframalpha.com/v1/llm-api

Requiere: WOLFRAM_MCP_API_KEY en .env
"""

import os
import logging
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WolframClient:
    """
    Cliente para Wolfram Alpha LLM API.

    Proporciona acceso a las capacidades de cálculo de Wolfram Alpha
    como fallback para problemas que SymPy no puede resolver.
    """

    LLM_API_URL = "https://api.wolframalpha.com/v1/llm-api"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or os.getenv("WOLFRAM_MCP_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            logger.warning("WOLFRAM_MCP_API_KEY no configurada")

    @property
    def available(self) -> bool:
        """Comprueba si la API está disponible (key configurada)."""
        return bool(self.api_key)

    def query(self, input_expr: str) -> Dict[str, Any]:
        """
        Envía una consulta a Wolfram Alpha LLM API.

        Args:
            input_expr: Expresión o pregunta (ej: "integrate sin(x)^2 dx")

        Returns:
            Dict con:
              - success: bool
              - result: str (texto completo de la respuesta)
              - execution_time_ms: float
              - error: str | None
        """
        if not self.api_key:
            return {
                "success": False,
                "result": "",
                "execution_time_ms": 0,
                "error": "WOLFRAM_MCP_API_KEY no configurada",
            }

        params = urllib.parse.urlencode({
            "appid": self.api_key,
            "input": input_expr,
        })
        url = f"{self.LLM_API_URL}?{params}"

        start = time.time()
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                elapsed = (time.time() - start) * 1000

                logger.info(
                    f"Wolfram query completada: {elapsed:.0f}ms, "
                    f"{len(body)} chars"
                )

                return {
                    "success": True,
                    "result": body,
                    "execution_time_ms": elapsed,
                    "error": None,
                }

        except urllib.error.HTTPError as e:
            elapsed = (time.time() - start) * 1000
            error_msg = f"HTTP {e.code}: {e.reason}"
            logger.error(f"Wolfram API error: {error_msg}")
            return {
                "success": False,
                "result": "",
                "execution_time_ms": elapsed,
                "error": error_msg,
            }
        except urllib.error.URLError as e:
            elapsed = (time.time() - start) * 1000
            error_msg = f"URL error: {e.reason}"
            logger.error(f"Wolfram API error: {error_msg}")
            return {
                "success": False,
                "result": "",
                "execution_time_ms": elapsed,
                "error": error_msg,
            }
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"Wolfram API error inesperado: {e}")
            return {
                "success": False,
                "result": "",
                "execution_time_ms": elapsed,
                "error": str(e),
            }

    def solve(self, equation: str) -> Dict[str, Any]:
        """Resuelve una ecuación via Wolfram Alpha."""
        return self.query(f"solve {equation}")

    def simplify(self, expr: str) -> Dict[str, Any]:
        """Simplifica una expresión via Wolfram Alpha."""
        return self.query(f"simplify {expr}")

    def integrate(self, expr: str, variable: str = "x") -> Dict[str, Any]:
        """Calcula una integral via Wolfram Alpha."""
        return self.query(f"integrate {expr} d{variable}")

    def differentiate(self, expr: str, variable: str = "x") -> Dict[str, Any]:
        """Calcula una derivada via Wolfram Alpha."""
        return self.query(f"derivative of {expr} with respect to {variable}")

    def eigenvalues(self, matrix_str: str) -> Dict[str, Any]:
        """Calcula eigenvalores de una matriz."""
        return self.query(f"eigenvalues {matrix_str}")

    def step_by_step(self, expr: str) -> Dict[str, Any]:
        """Solicita solución paso a paso."""
        return self.query(f"step-by-step {expr}")
