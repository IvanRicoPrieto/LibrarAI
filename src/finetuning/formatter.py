"""
Fine-tune Formatter — Formatea datos para OpenAI fine-tuning API.

Soporta:
- Formato JSONL para embeddings fine-tuning
- Validación de formato
- Split train/validation
"""

import json
import hashlib
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class OpenAIFineTuneFormatter:
    """
    Formatea pares de entrenamiento al formato JSONL de OpenAI.

    Formato de salida (embedding fine-tuning):
    {"prompt": "query text", "completion": "relevant passage"}

    Con hard negatives (triplet format):
    {"query": "...", "pos": "...", "neg": "..."}
    """

    # ------------------------------------------------------------------
    # Formateo
    # ------------------------------------------------------------------

    def format_pairs(
        self,
        pairs: list,
        output_path: Path,
        negatives: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Escribe pares de entrenamiento en formato JSONL.

        Args:
            pairs: Lista de TrainingPair.
            output_path: Ruta del archivo JSONL de salida.
            negatives: Hard negatives (opcional).

        Returns:
            Estadísticas del formateo.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_pairs": 0,
            "total_with_negatives": 0,
            "output_file": str(output_path),
            "format": "triplet" if negatives else "pair",
        }

        with open(output_path, "w", encoding="utf-8") as f:
            if negatives:
                # Formato triplet con hard negatives
                for neg_item in negatives:
                    for neg in neg_item.get("negatives", []):
                        entry = {
                            "query": neg_item["query"],
                            "pos": neg_item["positive"],
                            "neg": neg,
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        stats["total_pairs"] += 1
                        stats["total_with_negatives"] += 1
            else:
                # Formato pares simples
                for pair in pairs:
                    if isinstance(pair, dict):
                        query = pair.get("query", "")
                        passage = pair.get("positive_passage", "")
                    else:
                        query = getattr(pair, "query", "")
                        passage = getattr(pair, "positive_passage", "")
                    entry = {
                        "prompt": query,
                        "completion": passage,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    stats["total_pairs"] += 1

        logger.info(
            f"Formateados {stats['total_pairs']} pares → {output_path}"
        )
        return stats

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def validate_format(self, file_path: Path) -> Dict:
        """
        Valida un archivo JSONL de entrenamiento.

        Returns:
            Dict con resultados de validación.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"valid": False, "error": "Archivo no encontrado"}

        stats = {
            "valid": True,
            "total_lines": 0,
            "valid_lines": 0,
            "errors": [],
            "duplicates": 0,
            "format_detected": None,
        }

        seen_hashes = set()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats["total_lines"] += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    stats["errors"].append(
                        f"Línea {line_num}: JSON inválido: {e}"
                    )
                    stats["valid"] = False
                    continue

                if not isinstance(entry, dict):
                    stats["errors"].append(
                        f"Línea {line_num}: No es un objeto JSON"
                    )
                    stats["valid"] = False
                    continue

                # Detectar formato
                if "query" in entry and "pos" in entry:
                    fmt = "triplet"
                elif "prompt" in entry and "completion" in entry:
                    fmt = "pair"
                else:
                    stats["errors"].append(
                        f"Línea {line_num}: Campos desconocidos: "
                        f"{list(entry.keys())}"
                    )
                    stats["valid"] = False
                    continue

                if stats["format_detected"] is None:
                    stats["format_detected"] = fmt
                elif stats["format_detected"] != fmt:
                    stats["errors"].append(
                        f"Línea {line_num}: Formato mixto ({fmt} vs "
                        f"{stats['format_detected']})"
                    )

                # Verificar duplicados
                content_hash = hashlib.sha256(
                    line.encode()
                ).hexdigest()[:16]
                if content_hash in seen_hashes:
                    stats["duplicates"] += 1
                else:
                    seen_hashes.add(content_hash)

                # Verificar campos no vacíos
                if fmt == "pair":
                    if not entry.get("prompt") or not entry.get("completion"):
                        stats["errors"].append(
                            f"Línea {line_num}: Campos vacíos"
                        )
                elif fmt == "triplet":
                    if not entry.get("query") or not entry.get("pos"):
                        stats["errors"].append(
                            f"Línea {line_num}: Campos vacíos"
                        )

                stats["valid_lines"] += 1

        if stats["errors"]:
            stats["valid"] = len(stats["errors"]) == 0

        return stats

    # ------------------------------------------------------------------
    # Split train/val
    # ------------------------------------------------------------------

    def split_train_val(
        self,
        input_path: Path,
        train_path: Path,
        val_path: Path,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> Dict:
        """
        Divide datos en train y validation.

        Args:
            input_path: Archivo JSONL de entrada.
            train_path: Archivo JSONL de train.
            val_path: Archivo JSONL de validation.
            val_fraction: Fracción para validation (0-1).
            seed: Semilla para reproducibilidad.

        Returns:
            Estadísticas del split.
        """
        input_path = Path(input_path)
        train_path = Path(train_path)
        val_path = Path(val_path)

        train_path.parent.mkdir(parents=True, exist_ok=True)
        val_path.parent.mkdir(parents=True, exist_ok=True)

        # Leer todas las líneas
        with open(input_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Shuffle reproducible
        random.seed(seed)
        random.shuffle(lines)

        # Split
        n_val = max(1, int(len(lines) * val_fraction))
        val_lines = lines[:n_val]
        train_lines = lines[n_val:]

        # Escribir
        with open(train_path, "w", encoding="utf-8") as f:
            for line in train_lines:
                f.write(line + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for line in val_lines:
                f.write(line + "\n")

        stats = {
            "total": len(lines),
            "train": len(train_lines),
            "val": len(val_lines),
            "val_fraction_actual": len(val_lines) / len(lines) if lines else 0,
            "train_path": str(train_path),
            "val_path": str(val_path),
        }

        logger.info(
            f"Split: {stats['train']} train + {stats['val']} val "
            f"({stats['val_fraction_actual']:.1%})"
        )
        return stats
