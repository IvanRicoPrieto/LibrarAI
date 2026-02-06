"""
Excepciones para claude_max_client.

Jerarquia:
    ClaudeMaxError
    ├── AuthenticationError    - Suscripcion no valida o no autenticada
    ├── RateLimitError         - Limite de rate de la suscripcion alcanzado
    ├── ModelNotAvailableError - Modelo no disponible en la suscripcion
    ├── ImageProcessingError   - Error procesando imagen para vision
    ├── FileProcessingError    - Error procesando archivo de texto/codigo
    └── CompletionError        - Error durante la generacion de respuesta
"""


class ClaudeMaxError(Exception):
    """Error base para claude_max_client."""
    pass


class AuthenticationError(ClaudeMaxError):
    """La suscripcion de Claude Code Max no esta activa o no se pudo autenticar."""
    pass


class RateLimitError(ClaudeMaxError):
    """Se alcanzo el limite de rate de la suscripcion."""

    def __init__(self, message: str = "Rate limit alcanzado", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelNotAvailableError(ClaudeMaxError):
    """El modelo solicitado no esta disponible en la suscripcion actual."""
    pass


class ImageProcessingError(ClaudeMaxError):
    """Error al procesar una imagen para enviarla a Claude."""
    pass


class FileProcessingError(ClaudeMaxError):
    """Error al procesar un archivo de texto/codigo para incluirlo como contexto."""
    pass


class CompletionError(ClaudeMaxError):
    """Error durante la generacion de una respuesta."""
    pass
