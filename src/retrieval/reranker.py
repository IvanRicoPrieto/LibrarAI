"""
Reranker - Re-ranking de resultados con Cross-Encoder.

Mejora la precisión del retrieval aplicando un modelo cross-encoder
que evalúa query-documento de forma conjunta.

Modelos soportados:
- ms-marco-MiniLM-L-6-v2 (default, rápido, ~22M params)
- bge-reranker-base (mejor calidad, ~278M params)
- bge-reranker-large (máxima calidad, ~560M params)
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuración del reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Alternativas:
    # - "BAAI/bge-reranker-base"
    # - "BAAI/bge-reranker-large"
    # - "cross-encoder/ms-marco-TinyBERT-L-2-v2" (muy rápido, menor calidad)
    
    batch_size: int = 32
    max_length: int = 512  # Máximo tokens por par query-doc
    normalize_scores: bool = True  # Normalizar scores a [0, 1]
    
    # Combinación con score RRF original
    reranker_weight: float = 0.7  # Peso del score del reranker
    original_weight: float = 0.3  # Peso del score RRF original


class CrossEncoderReranker:
    """
    Re-ranker basado en Cross-Encoder.
    
    El cross-encoder evalúa query-documento de forma conjunta,
    capturando interacciones que los bi-encoders (embeddings) pierden.
    
    Típicamente mejora la precisión 15-25% en el ranking final.
    """
    
    # Cache del modelo a nivel de clase para evitar recargar
    _model_cache: Dict[str, Any] = {}
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Args:
            config: Configuración del reranker
        """
        self.config = config or RerankerConfig()
        self._model = None
    
    def _load_model(self):
        """Carga el modelo cross-encoder (lazy loading con cache)."""
        if self._model is not None:
            return
        
        model_name = self.config.model_name
        
        # Usar cache a nivel de clase
        if model_name in CrossEncoderReranker._model_cache:
            self._model = CrossEncoderReranker._model_cache[model_name]
            logger.debug(f"Reranker: usando modelo cacheado {model_name}")
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Cargando modelo cross-encoder: {model_name}")
            start_time = time.time()
            
            self._model = CrossEncoder(
                model_name,
                max_length=self.config.max_length
            )
            
            # Cachear para futuras instancias
            CrossEncoderReranker._model_cache[model_name] = self._model
            
            load_time = time.time() - start_time
            logger.info(f"Modelo cargado en {load_time:.2f}s")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers no instalado. "
                "Ejecuta: pip install sentence-transformers"
            )
    
    def rerank(
        self,
        query: str,
        results: List,  # List[RetrievalResult]
        top_k: int = 10
    ) -> List:
        """
        Re-rankea resultados usando el cross-encoder.
        
        Args:
            query: Consulta original
            results: Resultados del retrieval (RetrievalResult)
            top_k: Número de resultados a devolver
            
        Returns:
            Resultados re-ordenados con scores actualizados
        """
        if not results:
            return []
        
        self._load_model()
        
        start_time = time.time()
        
        # Preparar pares query-documento
        pairs = [(query, r.content) for r in results]
        
        # Obtener scores del cross-encoder
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False
        )
        
        # Normalizar scores si está configurado
        if self.config.normalize_scores and len(scores) > 1:
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            if score_range > 0:
                scores = [(s - min_score) / score_range for s in scores]
            else:
                scores = [0.5] * len(scores)
        
        # Combinar con score original (RRF)
        for result, ce_score in zip(results, scores):
            original_score = result.score
            
            # Score combinado
            combined_score = (
                self.config.reranker_weight * float(ce_score) +
                self.config.original_weight * original_score
            )
            
            # Guardar scores en metadata para debugging
            result.metadata["reranker_score"] = float(ce_score)
            result.metadata["original_rrf_score"] = original_score
            result.metadata["combined_score"] = combined_score
            
            # Actualizar score principal
            result.score = combined_score
        
        # Re-ordenar por score combinado
        results.sort(key=lambda x: x.score, reverse=True)
        
        rerank_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Re-ranking completado: {len(results)} docs en {rerank_time:.0f}ms, "
            f"top score: {results[0].score:.4f}"
        )
        
        return results[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Devuelve información sobre el modelo cargado."""
        return {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "reranker_weight": self.config.reranker_weight,
            "original_weight": self.config.original_weight,
            "loaded": self._model is not None
        }


class RerankerFactory:
    """Factory para crear rerankers con diferentes configuraciones."""
    
    PRESETS = {
        "fast": RerankerConfig(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
            reranker_weight=0.6,
            original_weight=0.4
        ),
        "balanced": RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_weight=0.7,
            original_weight=0.3
        ),
        "quality": RerankerConfig(
            model_name="BAAI/bge-reranker-base",
            reranker_weight=0.75,
            original_weight=0.25
        ),
        "max_quality": RerankerConfig(
            model_name="BAAI/bge-reranker-large",
            reranker_weight=0.8,
            original_weight=0.2
        )
    }
    
    @classmethod
    def create(cls, preset: str = "balanced") -> CrossEncoderReranker:
        """
        Crea un reranker con una configuración predefinida.
        
        Args:
            preset: "fast", "balanced", "quality", o "max_quality"
            
        Returns:
            CrossEncoderReranker configurado
        """
        if preset not in cls.PRESETS:
            raise ValueError(f"Preset desconocido: {preset}. Usa: {list(cls.PRESETS.keys())}")
        
        return CrossEncoderReranker(config=cls.PRESETS[preset])
    
    @classmethod
    def create_custom(cls, model_name: str, **kwargs) -> CrossEncoderReranker:
        """
        Crea un reranker con configuración personalizada.
        
        Args:
            model_name: Nombre del modelo cross-encoder
            **kwargs: Otros parámetros de RerankerConfig
            
        Returns:
            CrossEncoderReranker configurado
        """
        config = RerankerConfig(model_name=model_name, **kwargs)
        return CrossEncoderReranker(config=config)
