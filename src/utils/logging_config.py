"""
Logging Estructurado para LibrarAI.

Implementa logging con estructlog para:
- Tracing de requests/queries
- Métricas de rendimiento
- Debugging de problemas de retrieval
- Correlación de logs entre componentes

Uso:
    from src.utils.logging_config import get_logger, trace_context
    
    logger = get_logger(__name__)
    
    with trace_context(query="¿Qué es BB84?"):
        logger.info("processing_query", top_k=5, filters={"domain": "qkd"})
"""

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field
import json

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


# =============================================================================
# Configuración de Logging
# =============================================================================

@dataclass
class LogConfig:
    """Configuración del sistema de logging."""
    level: str = "INFO"
    format: str = "json"  # "json" o "console"
    log_file: Optional[Path] = None
    include_timestamp: bool = True
    include_caller: bool = True
    include_trace_id: bool = True
    
    # Métricas
    log_latencies: bool = True
    log_token_counts: bool = True
    
    # Rotación de logs
    max_file_size_mb: int = 50
    backup_count: int = 5


# Context variables para tracing
_trace_context: Dict[str, Any] = {}


def set_trace_context(**kwargs):
    """Establece contexto de tracing global."""
    global _trace_context
    _trace_context.update(kwargs)


def clear_trace_context():
    """Limpia el contexto de tracing."""
    global _trace_context
    _trace_context = {}


def get_trace_context() -> Dict[str, Any]:
    """Obtiene el contexto de tracing actual."""
    return _trace_context.copy()


@contextmanager
def trace_context(**kwargs):
    """
    Context manager para establecer contexto de tracing.
    
    Ejemplo:
        with trace_context(query="¿Qué es BB84?", session_id="abc123"):
            # Todos los logs dentro tendrán query y session_id
            logger.info("processing")
    """
    global _trace_context
    old_context = _trace_context.copy()
    
    # Generar trace_id si no existe
    if "trace_id" not in _trace_context and "trace_id" not in kwargs:
        kwargs["trace_id"] = str(uuid.uuid4())[:8]
    
    _trace_context.update(kwargs)
    try:
        yield _trace_context
    finally:
        _trace_context = old_context


# =============================================================================
# Processors para structlog
# =============================================================================

def add_trace_context(logger, method_name, event_dict):
    """Añade contexto de tracing a cada log."""
    context = get_trace_context()
    for key, value in context.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Añade timestamp ISO 8601."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_service_info(logger, method_name, event_dict):
    """Añade información del servicio."""
    event_dict["service"] = "librar_ai"
    event_dict["version"] = "1.0.0"
    return event_dict


def format_for_humans(logger, method_name, event_dict):
    """Formatea logs para lectura humana en consola."""
    timestamp = event_dict.pop("timestamp", "")
    event = event_dict.pop("event", "")
    level = event_dict.pop("level", "INFO")
    trace_id = event_dict.pop("trace_id", "")
    
    # Color según nivel
    colors = {
        "debug": "\033[36m",    # Cyan
        "info": "\033[32m",     # Green
        "warning": "\033[33m",  # Yellow
        "error": "\033[31m",    # Red
        "critical": "\033[35m", # Magenta
    }
    reset = "\033[0m"
    color = colors.get(level.lower(), "")
    
    # Formatear extras
    extras = " ".join(f"{k}={v}" for k, v in event_dict.items() if k not in ["service", "version"])
    
    trace_str = f"[{trace_id}] " if trace_id else ""
    extras_str = f" | {extras}" if extras else ""
    
    return f"{timestamp[:19]} {color}{level.upper():8}{reset} {trace_str}{event}{extras_str}"


# =============================================================================
# Configuración de structlog
# =============================================================================

def configure_logging(config: Optional[LogConfig] = None):
    """
    Configura el sistema de logging estructurado.
    
    Args:
        config: Configuración de logging (usa defaults si None)
    """
    if config is None:
        config = LogConfig()
    
    if not STRUCTLOG_AVAILABLE:
        # Fallback a logging estándar
        logging.basicConfig(
            level=getattr(logging, config.level),
            format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return
    
    # Processors comunes
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_trace_context,
        add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Formato de salida
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(format_for_humans)
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configurar nivel de logging estándar
    logging.basicConfig(
        level=getattr(logging, config.level),
        format="%(message)s",
        stream=sys.stdout,
    )
    
    # Configurar archivo si se especifica
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(getattr(logging, config.level))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str = None):
    """
    Obtiene un logger estructurado.
    
    Args:
        name: Nombre del módulo (usa __name__ normalmente)
    
    Returns:
        Logger configurado
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# =============================================================================
# Decoradores de Logging
# =============================================================================

def log_execution_time(
    operation: str = None,
    log_args: bool = False,
    log_result: bool = False,
):
    """
    Decorador para loguear tiempo de ejecución de funciones.
    
    Args:
        operation: Nombre de la operación (usa nombre de función si None)
        log_args: Si loguear argumentos de entrada
        log_result: Si loguear resultado (cuidado con datos grandes)
    
    Ejemplo:
        @log_execution_time("vector_search", log_args=True)
        def search_vectors(query: str, top_k: int):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            op_name = operation or func.__name__
            
            # Preparar contexto de log
            log_context = {"operation": op_name}
            if log_args:
                # Evitar loguear self/cls
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                filtered_args = {
                    name: repr(arg)[:100]
                    for name, arg in zip(arg_names, args)
                    if name not in ("self", "cls")
                }
                filtered_args.update({k: repr(v)[:100] for k, v in kwargs.items()})
                log_context["args"] = filtered_args
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                log_context["duration_ms"] = round(elapsed_ms, 2)
                log_context["status"] = "success"
                
                if log_result and result is not None:
                    log_context["result_type"] = type(result).__name__
                    if hasattr(result, "__len__"):
                        log_context["result_length"] = len(result)
                
                logger.info(f"{op_name}_completed", **log_context)
                return result
                
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                log_context["duration_ms"] = round(elapsed_ms, 2)
                log_context["status"] = "error"
                log_context["error_type"] = type(e).__name__
                log_context["error_message"] = str(e)[:200]
                
                logger.error(f"{op_name}_failed", **log_context)
                raise
        
        return wrapper
    return decorator


def log_async_execution_time(
    operation: str = None,
    log_args: bool = False,
):
    """Versión async del decorador log_execution_time."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            op_name = operation or func.__name__
            
            log_context = {"operation": op_name}
            if log_args:
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                filtered_args = {
                    name: repr(arg)[:100]
                    for name, arg in zip(arg_names, args)
                    if name not in ("self", "cls")
                }
                log_context["args"] = filtered_args
            
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                log_context["duration_ms"] = round(elapsed_ms, 2)
                log_context["status"] = "success"
                
                logger.info(f"{op_name}_completed", **log_context)
                return result
                
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                log_context["duration_ms"] = round(elapsed_ms, 2)
                log_context["status"] = "error"
                log_context["error_type"] = type(e).__name__
                
                logger.error(f"{op_name}_failed", **log_context)
                raise
        
        return wrapper
    return decorator


# =============================================================================
# Helpers para métricas específicas de RAG
# =============================================================================

@dataclass
class RAGMetrics:
    """Métricas de una query RAG."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    
    # Tiempos (ms)
    embedding_time_ms: float = 0
    vector_search_time_ms: float = 0
    bm25_search_time_ms: float = 0
    graph_search_time_ms: float = 0
    fusion_time_ms: float = 0
    rerank_time_ms: float = 0
    generation_time_ms: float = 0
    total_time_ms: float = 0
    
    # Conteos
    chunks_retrieved: int = 0
    chunks_after_fusion: int = 0
    chunks_after_rerank: int = 0
    
    # Tokens
    context_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    
    # Cache
    embedding_cache_hit: bool = False
    semantic_cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para logging."""
        return {
            "trace_id": self.trace_id,
            "query_length": len(self.query),
            "timing": {
                "embedding_ms": self.embedding_time_ms,
                "vector_search_ms": self.vector_search_time_ms,
                "bm25_search_ms": self.bm25_search_time_ms,
                "graph_search_ms": self.graph_search_time_ms,
                "fusion_ms": self.fusion_time_ms,
                "rerank_ms": self.rerank_time_ms,
                "generation_ms": self.generation_time_ms,
                "total_ms": self.total_time_ms,
            },
            "retrieval": {
                "chunks_retrieved": self.chunks_retrieved,
                "chunks_after_fusion": self.chunks_after_fusion,
                "chunks_after_rerank": self.chunks_after_rerank,
            },
            "tokens": {
                "context": self.context_tokens,
                "response": self.response_tokens,
                "total": self.total_tokens,
            },
            "cache": {
                "embedding_hit": self.embedding_cache_hit,
                "semantic_hit": self.semantic_cache_hit,
            },
        }
    
    def log(self, logger=None):
        """Loguea las métricas."""
        if logger is None:
            logger = get_logger("rag_metrics")
        logger.info("rag_query_metrics", **self.to_dict())


def log_rag_query(
    query: str,
    metrics: RAGMetrics,
    response_length: int = 0,
    sources_count: int = 0,
):
    """
    Loguea una query RAG completa con todas sus métricas.
    
    Args:
        query: Query del usuario
        metrics: Métricas recopiladas
        response_length: Longitud de la respuesta generada
        sources_count: Número de fuentes citadas
    """
    logger = get_logger("rag")
    
    logger.info(
        "query_processed",
        trace_id=metrics.trace_id,
        query_preview=query[:100] + "..." if len(query) > 100 else query,
        total_time_ms=metrics.total_time_ms,
        chunks_used=metrics.chunks_after_rerank or metrics.chunks_after_fusion,
        tokens_total=metrics.total_tokens,
        response_length=response_length,
        sources_count=sources_count,
        cache_hits={
            "embedding": metrics.embedding_cache_hit,
            "semantic": metrics.semantic_cache_hit,
        },
    )


# =============================================================================
# Inicialización
# =============================================================================

# Configurar logging al importar (con defaults)
configure_logging()
