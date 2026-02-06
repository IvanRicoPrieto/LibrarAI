"""
Utilidades para claude_max_client.

Funciones auxiliares para procesamiento de archivos, imagenes y estimacion de tokens.
"""

import base64
import mimetypes
from pathlib import Path

from .exceptions import ImageProcessingError, FileProcessingError


# ============================================================================
# Imagenes - formatos soportados por Claude Vision
# ============================================================================

SUPPORTED_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


# ============================================================================
# Archivos de texto - extensiones reconocidas como legibles
# ============================================================================

TEXT_FILE_EXTENSIONS = {
    # Codigo fuente
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".lua", ".pl", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    # Web
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    # Datos y config
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".csv", ".tsv", ".sql",
    # Documentacion
    ".md", ".markdown", ".rst", ".txt", ".tex", ".bib", ".org", ".adoc",
    # Otros
    ".dockerfile", ".makefile", ".gitignore", ".editorconfig",
    ".ipynb",  # Jupyter notebooks (son JSON)
    ".log",
}

# Tamano maximo para leer archivos de texto (en bytes) - 1MB
MAX_TEXT_FILE_SIZE = 1_048_576


def validate_image_path(image_path: str) -> Path:
    """
    Valida que una ruta de imagen existe y es de un tipo soportado.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Path resuelto y validado.

    Raises:
        ImageProcessingError: Si el archivo no existe o no es un tipo soportado.
    """
    path = Path(image_path).resolve()

    if not path.exists():
        raise ImageProcessingError(f"Imagen no encontrada: {image_path}")

    if not path.is_file():
        raise ImageProcessingError(f"La ruta no es un archivo: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_TYPES:
        supported = ", ".join(SUPPORTED_IMAGE_TYPES.keys())
        raise ImageProcessingError(
            f"Tipo de imagen no soportado: {suffix}. Soportados: {supported}"
        )

    return path


def validate_file_path(file_path: str) -> Path:
    """
    Valida que una ruta de archivo existe y es legible.

    Args:
        file_path: Ruta al archivo.

    Returns:
        Path resuelto y validado.

    Raises:
        FileProcessingError: Si el archivo no existe, no es un archivo,
            es demasiado grande, o no es de un tipo reconocido.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileProcessingError(f"Archivo no encontrado: {file_path}")

    if not path.is_file():
        raise FileProcessingError(f"La ruta no es un archivo: {file_path}")

    suffix = path.suffix.lower()
    # Archivos sin extension o con extension no reconocida:
    # intentar leerlos de todas formas (podrian ser scripts sin extension)
    # Solo bloquear binarios conocidos
    binary_extensions = {
        ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
        ".pdf",  # PDF requiere parser especial, no texto plano
    }
    if suffix in binary_extensions:
        raise FileProcessingError(
            f"Tipo de archivo binario no soportado como texto: {suffix}. "
            f"Usa 'images' para imagenes o procesa el archivo externamente."
        )

    file_size = path.stat().st_size
    if file_size > MAX_TEXT_FILE_SIZE:
        size_mb = file_size / 1_048_576
        raise FileProcessingError(
            f"Archivo demasiado grande: {size_mb:.1f}MB (max {MAX_TEXT_FILE_SIZE // 1_048_576}MB). "
            f"Considera dividir el archivo o pasar solo el fragmento relevante como texto."
        )

    return path


def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Lee el contenido de un archivo de texto.

    Args:
        file_path: Ruta al archivo.
        encoding: Codificacion del archivo (default: utf-8).

    Returns:
        Contenido del archivo como string.

    Raises:
        FileProcessingError: Si no se puede leer el archivo.
    """
    path = validate_file_path(file_path)

    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # Intentar con latin-1 como fallback
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            raise FileProcessingError(
                f"No se pudo decodificar el archivo {file_path}: {e}"
            ) from e
    except Exception as e:
        raise FileProcessingError(
            f"Error leyendo archivo {file_path}: {e}"
        ) from e


def is_image_file(file_path: str) -> bool:
    """Comprueba si un archivo es una imagen soportada por Claude Vision."""
    suffix = Path(file_path).suffix.lower()
    return suffix in SUPPORTED_IMAGE_TYPES


def is_text_file(file_path: str) -> bool:
    """Comprueba si un archivo es reconocido como texto legible."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    # Sin extension: probablemente script
    if not suffix:
        return True
    return suffix in TEXT_FILE_EXTENSIONS


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Lee una imagen y la codifica en base64.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Tupla (base64_data, media_type).

    Raises:
        ImageProcessingError: Si no se puede leer o codificar la imagen.
    """
    path = validate_image_path(image_path)

    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        media_type = SUPPORTED_IMAGE_TYPES[path.suffix.lower()]
        return b64, media_type
    except Exception as e:
        raise ImageProcessingError(f"Error codificando imagen {image_path}: {e}") from e


def estimate_tokens(text: str) -> int:
    """
    Estimacion rapida de tokens para un texto.

    Usa la heuristica de ~4 caracteres por token, que es razonablemente
    precisa para texto en ingles/espanol con modelos Claude.

    Args:
        text: Texto a estimar.

    Returns:
        Numero estimado de tokens.
    """
    if not text:
        return 0
    return max(1, len(text) // 3)


def estimate_image_tokens(image_path: str) -> int:
    """
    Estima los tokens que consumira una imagen.

    Claude usa aproximadamente 1 token por cada 750 pixeles.
    Sin leer la imagen completa, estimamos basandonos en el tamano del archivo.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Numero estimado de tokens.
    """
    path = validate_image_path(image_path)
    file_size = path.stat().st_size

    if file_size < 100_000:
        return 500
    elif file_size < 500_000:
        return 1000
    elif file_size < 1_000_000:
        return 1500
    else:
        return 2000


def normalize_prompt_input(
    prompt_input: str | dict,
    shared_system: str | None = None,
) -> dict:
    """
    Normaliza un input de prompt a formato dict para batch processing.

    Args:
        prompt_input: String simple o dict con campos {prompt, system, model, ...}.
        shared_system: System prompt compartido (se usa si el input no tiene uno propio).

    Returns:
        Dict normalizado con al menos la clave "prompt".
    """
    if isinstance(prompt_input, str):
        result = {"prompt": prompt_input}
    elif isinstance(prompt_input, dict):
        result = dict(prompt_input)
        if "prompt" not in result:
            raise ValueError("El dict de prompt debe contener la clave 'prompt'")
    else:
        raise TypeError(f"Tipo de prompt no soportado: {type(prompt_input)}")

    # Aplicar system prompt compartido si no tiene uno propio
    if shared_system and "system" not in result:
        result["system"] = shared_system

    return result
