"""
Image Describer - Describe imágenes referenciadas en Markdown para indexación.

Utiliza Claude Max (vision) para generar descripciones textuales de las
imágenes encontradas en los documentos Markdown, permitiendo que el contenido
visual sea indexado y recuperable via RAG.

Las descripciones se cachean en disco para no reprocesar imágenes ya vistas.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Regex para encontrar referencias a imágenes en Markdown
IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')


class ImageDescriptionCache:
    """Cache persistente de descripciones de imágenes."""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info(f"Cache de imágenes cargado: {len(self._cache)} entradas")
            except Exception as e:
                logger.warning(f"Error cargando cache de imágenes: {e}")
                self._cache = {}

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def get(self, image_path: str) -> Optional[str]:
        return self._cache.get(image_path)

    def set(self, image_path: str, description: str):
        self._cache[image_path] = description

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, image_path: str) -> bool:
        return image_path in self._cache


def extract_image_references(content: str) -> List[Tuple[str, str, str]]:
    """
    Extrae referencias a imágenes del contenido Markdown.

    Returns:
        Lista de (full_match, alt_text, image_path)
    """
    results = []
    for match in IMAGE_PATTERN.finditer(content):
        results.append((match.group(0), match.group(1), match.group(2)))
    return results


class ImageDescriber:
    """
    Describe imágenes usando Claude Max (vision).

    Usa ClaudeMaxClient directamente (no llm_provider) porque
    la vision requiere el parámetro `images` que solo soporta Claude Max.
    """

    def __init__(self, cache_dir: Path):
        """
        Args:
            cache_dir: Directorio para almacenar la cache de descripciones
        """
        self.cache = ImageDescriptionCache(cache_dir / "image_descriptions.json")
        self._client = None

    def _init_client(self):
        if self._client is not None:
            return
        try:
            from ..claude_max_client import ClaudeMaxClient
            self._client = ClaudeMaxClient()
        except Exception as e:
            logger.error(f"No se pudo inicializar ClaudeMaxClient para vision: {e}")
            raise

    def describe_image(self, image_path: str) -> str:
        """
        Genera una descripción textual de una imagen.

        Args:
            image_path: Ruta absoluta a la imagen

        Returns:
            Descripción textual de la imagen
        """
        self._init_client()

        prompt = (
            "Describe esta imagen de forma concisa y precisa para un contexto académico "
            "de computación cuántica, matemáticas o finanzas cuantitativas. "
            "Si es un diagrama de circuito cuántico, describe las puertas y qubits. "
            "Si es una gráfica, describe los ejes, tendencias y datos relevantes. "
            "Si es una fórmula o ecuación, transcríbela en LaTeX. "
            "Si es una tabla, describe su contenido. "
            "Máximo 3 oraciones."
        )

        try:
            response = self._client.complete(
                prompt=prompt,
                system="Eres un asistente que describe imágenes académicas de forma precisa.",
                images=[image_path],
                temperature=0.1,
                max_tokens=300,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error describiendo imagen {image_path}: {e}")
            return f"[Imagen no procesada: {Path(image_path).name}]"

    def process_document_images(
        self,
        content: str,
        file_path: Path,
    ) -> str:
        """
        Reemplaza referencias de imágenes en el contenido Markdown con
        descripciones textuales.

        Args:
            content: Contenido Markdown original
            file_path: Ruta al archivo .md (para resolver rutas relativas)

        Returns:
            Contenido con imágenes reemplazadas por descripciones
        """
        file_dir = Path(file_path).parent
        refs = extract_image_references(content)

        if not refs:
            return content

        processed = 0
        skipped_remote = 0
        skipped_missing = 0
        cached = 0

        for full_match, alt_text, img_path in refs:
            # Ignorar URLs remotas
            if img_path.startswith(("http://", "https://")):
                placeholder = f"\n[IMAGEN REMOTA: {alt_text or img_path}]\n"
                content = content.replace(full_match, placeholder, 1)
                skipped_remote += 1
                continue

            # Resolver ruta relativa
            abs_path = (file_dir / img_path).resolve()

            if not abs_path.exists():
                logger.debug(f"Imagen no encontrada: {abs_path}")
                placeholder = f"\n[IMAGEN NO ENCONTRADA: {alt_text or img_path}]\n"
                content = content.replace(full_match, placeholder, 1)
                skipped_missing += 1
                continue

            # Verificar extensión soportada
            if abs_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
                placeholder = f"\n[IMAGEN: {alt_text or abs_path.name}]\n"
                content = content.replace(full_match, placeholder, 1)
                continue

            # Buscar en cache
            cache_key = str(abs_path)
            description = self.cache.get(cache_key)

            if description is None:
                # Describir con vision
                description = self.describe_image(str(abs_path))
                self.cache.set(cache_key, description)
                processed += 1
            else:
                cached += 1

            # Reemplazar en contenido
            block = (
                f"\n[DESCRIPCIÓN DE IMAGEN: {alt_text}]\n"
                f"{description}\n"
                f"[FIN DESCRIPCIÓN]\n"
            )
            content = content.replace(full_match, block, 1)

        # Guardar cache periódicamente
        if processed > 0:
            self.cache.save()

        logger.info(
            f"Imágenes procesadas: {processed} nuevas, {cached} cacheadas, "
            f"{skipped_remote} remotas, {skipped_missing} no encontradas"
        )

        return content
