"""
Fine-tune Launcher — Gestión de jobs de fine-tuning en OpenAI.

Permite:
- Subir archivos de entrenamiento
- Crear y gestionar jobs de fine-tuning
- Monitorizar estado
- Comparar modelo fine-tuned vs base
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FineTuneLauncher:
    """
    Gestiona fine-tuning de modelos de embeddings en OpenAI.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API key. Si None, usa OPENAI_API_KEY.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def _init_client(self):
        """Inicializa cliente de OpenAI."""
        if self._client is not None:
            return

        from openai import OpenAI

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no configurada.")
        self._client = OpenAI(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_training_file(self, path: Path) -> str:
        """
        Sube archivo de entrenamiento a OpenAI.

        Args:
            path: Ruta al archivo JSONL.

        Returns:
            file_id de OpenAI.
        """
        self._init_client()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        with open(path, "rb") as f:
            response = self._client.files.create(
                file=f,
                purpose="fine-tune",
            )

        file_id = response.id
        logger.info(f"Archivo subido: {path.name} → {file_id}")
        return file_id

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_job(
        self,
        training_file_id: str,
        model: str = "text-embedding-3-small",
        suffix: Optional[str] = None,
        n_epochs: Optional[int] = None,
        validation_file_id: Optional[str] = None,
    ) -> str:
        """
        Crea un job de fine-tuning.

        Args:
            training_file_id: ID del archivo de entrenamiento.
            model: Modelo base.
            suffix: Sufijo para el modelo resultante.
            n_epochs: Número de epochs (None = auto).
            validation_file_id: ID del archivo de validación.

        Returns:
            job_id.
        """
        self._init_client()

        params = {
            "training_file": training_file_id,
            "model": model,
        }
        if suffix:
            params["suffix"] = suffix
        if validation_file_id:
            params["validation_file"] = validation_file_id

        hyperparameters = {}
        if n_epochs is not None:
            hyperparameters["n_epochs"] = n_epochs
        if hyperparameters:
            params["hyperparameters"] = hyperparameters

        response = self._client.fine_tuning.jobs.create(**params)
        job_id = response.id
        logger.info(f"Job creado: {job_id} (modelo: {model})")
        return job_id

    def get_job_status(self, job_id: str) -> Dict:
        """Obtiene estado de un job."""
        self._init_client()
        job = self._client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "trained_tokens": job.trained_tokens,
            "error": job.error.message if job.error else None,
        }

    def list_jobs(self, limit: int = 10) -> List[Dict]:
        """Lista los jobs recientes."""
        self._init_client()
        jobs = self._client.fine_tuning.jobs.list(limit=limit)
        return [
            {
                "id": j.id,
                "status": j.status,
                "model": j.model,
                "fine_tuned_model": j.fine_tuned_model,
                "created_at": j.created_at,
            }
            for j in jobs.data
        ]

    def cancel_job(self, job_id: str) -> Dict:
        """Cancela un job."""
        self._init_client()
        job = self._client.fine_tuning.jobs.cancel(job_id)
        logger.info(f"Job cancelado: {job_id}")
        return {"id": job.id, "status": job.status}

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_model(
        self,
        model_id: str,
        test_queries: List[str],
        base_model: str = "text-embedding-3-large",
    ) -> Dict:
        """
        Compara embeddings del modelo fine-tuned vs base.

        Args:
            model_id: ID del modelo fine-tuned.
            test_queries: Lista de queries de test.
            base_model: Modelo base para comparación.

        Returns:
            Estadísticas de comparación.
        """
        self._init_client()

        if not test_queries:
            return {"error": "No hay queries de test"}

        # Generar embeddings con ambos modelos
        base_response = self._client.embeddings.create(
            input=test_queries,
            model=base_model,
        )
        ft_response = self._client.embeddings.create(
            input=test_queries,
            model=model_id,
        )

        base_embeddings = [d.embedding for d in base_response.data]
        ft_embeddings = [d.embedding for d in ft_response.data]

        # Calcular similitudes coseno entre pares consecutivos
        import math

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

        similarities = []
        for base_emb, ft_emb in zip(base_embeddings, ft_embeddings):
            sim = cosine_sim(base_emb, ft_emb)
            similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0

        return {
            "base_model": base_model,
            "finetuned_model": model_id,
            "num_queries": len(test_queries),
            "avg_similarity_base_vs_ft": round(avg_sim, 4),
            "base_dimensions": len(base_embeddings[0]) if base_embeddings else 0,
            "ft_dimensions": len(ft_embeddings[0]) if ft_embeddings else 0,
        }
