"""
Cost Tracker - Sistema de registro de costes por uso.

Registra todos los costes de API en formato CSV para análisis fácil.
Separa costes de indexación (build) de consultas (query).
"""

import csv
import logging
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Literal, Union
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class UsageType(Enum):
    """Tipo de uso del sistema."""
    BUILD = "build"      # Indexación: embeddings, construcción de índices
    QUERY = "query"      # Consultas: retrieval + generación


@dataclass
class UsageRecord:
    """Registro de uso individual."""
    timestamp: str
    usage_type: str          # "build" o "query"
    provider: str            # "openai", "anthropic", "ollama"
    model: str               # Modelo específico usado
    operation: str           # "embedding", "generation", "routing", etc.
    tokens_input: int
    tokens_output: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    total_cost: float
    query: Optional[str] = None  # Solo para queries
    
    def to_row(self) -> list:
        """Convierte a fila CSV."""
        return [
            self.timestamp,
            self.usage_type,
            self.provider,
            self.model,
            self.operation,
            self.tokens_input,
            self.tokens_output,
            f"{self.cost_per_1k_input:.6f}",
            f"{self.cost_per_1k_output:.6f}",
            f"{self.total_cost:.6f}",
            self.query or ""
        ]


class CostTracker:
    """
    Tracker de costes centralizado.
    
    Registra todos los usos de API en un archivo CSV para
    fácil análisis y seguimiento de gastos.
    """
    
    # Precios por 1K tokens (actualizados a enero 2026)
    PRICING = {
        # OpenAI
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        
        # Anthropic (modelos 2025-2026)
        "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},  # Claude 4.5 Sonnet
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},            # Alias Claude 4.5
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},      # Claude 4 Sonnet
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        
        # Claude Max (suscripcion tarifa plana, coste 0 por token)
        "claude-opus-4-5-20251101": {"input": 0.0, "output": 0.0},
        "claude-haiku-4-5-20251001": {"input": 0.0, "output": 0.0},

        # Ollama (gratis, coste local)
        "llama3.2": {"input": 0.0, "output": 0.0},
        "mistral": {"input": 0.0, "output": 0.0},
        "qwen2.5": {"input": 0.0, "output": 0.0},
    }
    
    CSV_HEADERS = [
        "timestamp",
        "usage_type",
        "provider",
        "model",
        "operation",
        "tokens_input",
        "tokens_output",
        "cost_per_1k_input",
        "cost_per_1k_output",
        "total_cost",
        "query"
    ]
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern para tracker global."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, logs_dir: Optional[Path] = None):
        """
        Args:
            logs_dir: Directorio para archivos de costes
        """
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        if logs_dir is None:
            # Detectar directorio del proyecto
            project_root = Path(__file__).parent.parent.parent
            logs_dir = project_root / "logs"
        
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.logs_dir / "cost_tracking.csv"
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """Crea el archivo CSV con headers si no existe."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)
    
    def get_pricing(self, model: str) -> dict:
        """Obtiene precios para un modelo."""
        if model not in self.PRICING:
            logger.warning(
                f"Modelo '{model}' no encontrado en tabla de precios. "
                f"Usando coste $0. Añádelo a CostTracker.PRICING si es de pago."
            )
        return self.PRICING.get(model, {"input": 0.0, "output": 0.0})
    
    def calculate_cost(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int = 0
    ) -> float:
        """Calcula el coste total de una operación."""
        pricing = self.get_pricing(model)
        
        cost_input = (tokens_input / 1000) * pricing["input"]
        cost_output = (tokens_output / 1000) * pricing["output"]
        
        return cost_input + cost_output
    
    def record(
        self,
        usage_type: UsageType,
        provider: str,
        model: str,
        operation: str,
        tokens_input: int,
        tokens_output: int = 0,
        query: Optional[str] = None
    ) -> UsageRecord:
        """
        Registra un uso en el CSV.
        
        Args:
            usage_type: BUILD o QUERY
            provider: "openai", "anthropic", "ollama"
            model: Nombre del modelo
            operation: "embedding", "generation", "routing", etc.
            tokens_input: Tokens de entrada
            tokens_output: Tokens de salida
            query: Consulta (solo para QUERY)
            
        Returns:
            Registro creado
        """
        pricing = self.get_pricing(model)
        total_cost = self.calculate_cost(model, tokens_input, tokens_output)
        
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            usage_type=usage_type.value,
            provider=provider,
            model=model,
            operation=operation,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_per_1k_input=pricing["input"],
            cost_per_1k_output=pricing["output"],
            total_cost=total_cost,
            query=query[:100] if query else None  # Truncar query larga
        )
        
        # Escribir al CSV
        with self._lock:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(record.to_row())
        
        return record
    
    def record_embedding(
        self,
        model: str,
        tokens: int,
        usage_type: UsageType = UsageType.BUILD
    ) -> UsageRecord:
        """Registra uso de embeddings."""
        provider = "openai" if "embedding" in model else "unknown"
        return self.record(
            usage_type=usage_type,
            provider=provider,
            model=model,
            operation="embedding",
            tokens_input=tokens,
            tokens_output=0
        )
    
    def record_generation(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int,
        query: str
    ) -> UsageRecord:
        """Registra uso de generación."""
        if "claude" in model.lower():
            provider = "anthropic"
        elif "gpt" in model.lower():
            provider = "openai"
        else:
            provider = "ollama"
        
        return self.record(
            usage_type=UsageType.QUERY,
            provider=provider,
            model=model,
            operation="generation",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            query=query
        )
    
    def record_routing(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int
    ) -> UsageRecord:
        """Registra uso de routing."""
        provider = "openai" if "gpt" in model.lower() else "anthropic"
        return self.record(
            usage_type=UsageType.QUERY,
            provider=provider,
            model=model,
            operation="routing",
            tokens_input=tokens_input,
            tokens_output=tokens_output
        )
    
    def get_summary(
        self,
        usage_type: Optional[Union[UsageType, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> dict:
        """
        Obtiene resumen de costes.
        
        Args:
            usage_type: Filtrar por tipo (BUILD/QUERY) - acepta UsageType o string
            start_date: Fecha inicio (ISO format)
            end_date: Fecha fin (ISO format)
            
        Returns:
            Diccionario con estadísticas
        """
        summary = {
            "total_cost": 0.0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "by_type": {"build": 0.0, "query": 0.0},
            "by_provider": {},
            "by_model": {},
            "by_operation": {},
            "record_count": 0
        }
        
        if not self.csv_path.exists():
            return summary
        
        # Normalizar usage_type a string
        filter_type = None
        if usage_type:
            if isinstance(usage_type, UsageType):
                filter_type = usage_type.value
            else:
                filter_type = str(usage_type).lower()
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Filtrar por tipo
                if filter_type and row["usage_type"] != filter_type:
                    continue
                
                # Filtrar por fecha
                if start_date and row["timestamp"] < start_date:
                    continue
                if end_date and row["timestamp"] > end_date:
                    continue
                
                cost = float(row["total_cost"])
                tokens_in = int(row["tokens_input"])
                tokens_out = int(row["tokens_output"])
                
                summary["total_cost"] += cost
                summary["total_tokens_input"] += tokens_in
                summary["total_tokens_output"] += tokens_out
                summary["record_count"] += 1
                
                # Por tipo
                t = row["usage_type"]
                summary["by_type"][t] = summary["by_type"].get(t, 0) + cost
                
                # Por proveedor
                p = row["provider"]
                summary["by_provider"][p] = summary["by_provider"].get(p, 0) + cost
                
                # Por modelo
                m = row["model"]
                summary["by_model"][m] = summary["by_model"].get(m, 0) + cost
                
                # Por operación
                o = row["operation"]
                summary["by_operation"][o] = summary["by_operation"].get(o, 0) + cost
        
        return summary
    
    def print_summary(self, usage_type: Optional[Union[UsageType, str]] = None):
        """Imprime resumen formateado."""
        summary = self.get_summary(usage_type)
        
        print("\n" + "=" * 60)
        print("💰 RESUMEN DE COSTES")
        print("=" * 60)
        
        print(f"\n📊 Total: ${summary['total_cost']:.4f}")
        print(f"   Tokens entrada: {summary['total_tokens_input']:,}")
        print(f"   Tokens salida:  {summary['total_tokens_output']:,}")
        print(f"   Registros:      {summary['record_count']}")
        
        print(f"\n📁 Por tipo:")
        print(f"   🔧 Build (indexación): ${summary['by_type'].get('build', 0):.4f}")
        print(f"   🔍 Query (consultas):  ${summary['by_type'].get('query', 0):.4f}")
        
        if summary["by_provider"]:
            print(f"\n🏢 Por proveedor:")
            for provider, cost in sorted(summary["by_provider"].items()):
                print(f"   {provider}: ${cost:.4f}")
        
        if summary["by_model"]:
            print(f"\n🤖 Por modelo:")
            for model, cost in sorted(summary["by_model"].items(), key=lambda x: -x[1]):
                print(f"   {model}: ${cost:.4f}")
        
        print("\n" + "=" * 60)


# Instancia global
_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Obtiene el tracker global."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


if __name__ == "__main__":
    # Test
    tracker = get_tracker()
    
    # Simular uso
    tracker.record_embedding("text-embedding-3-large", 5000)
    tracker.record_generation(
        "claude-3-5-sonnet-20241022",
        tokens_input=2000,
        tokens_output=500,
        query="¿Qué es el algoritmo de Shor?"
    )
    
    tracker.print_summary()
