"""
Caché Semántica para LibrarAI.

Cachea respuestas completas del LLM basándose en similitud semántica
de las queries, no solo coincidencia exacta. Si una query nueva es
semánticamente similar a una cacheada (>umbral), devuelve la respuesta
cacheada directamente.

Beneficios:
- Reduce costes de LLM dramáticamente para queries similares
- Latencia ~0ms para cache hits
- Útil para preguntas frecuentes o variaciones de las mismas

Ejemplo:
- Query 1: "¿Qué es el entrelazamiento cuántico?"
- Query 2: "Explícame el entrelazamiento cuántico"  → Cache hit!
- Query 3: "Define quantum entanglement"  → Cache hit!
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple
import sqlite3
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Respuesta cacheada con metadatos."""
    query: str
    query_embedding: List[float]
    response: str
    model: str
    sources: List[Dict[str, Any]]
    routing_info: Dict[str, Any]
    tokens_input: int
    tokens_output: int
    created_at: str
    access_count: int = 0
    last_accessed: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResponse":
        return cls(**data)


@dataclass 
class SemanticCacheConfig:
    """Configuración del caché semántico."""
    # Umbral de similitud para considerar cache hit (0.0-1.0)
    similarity_threshold: float = 0.92
    
    # Máximo de entradas en caché
    max_entries: int = 1000
    
    # TTL en días (0 = sin expiración)
    ttl_days: int = 30
    
    # Modelo de embeddings para queries
    embedding_model: str = "text-embedding-3-small"
    
    # Dimensiones del embedding
    embedding_dimensions: int = 1536
    
    # Usar caché en memoria además de disco
    use_memory_cache: bool = True
    
    # Tamaño máximo de caché en memoria
    memory_cache_size: int = 100


class SemanticCache:
    """
    Caché semántica para respuestas de LLM.
    
    Usa embeddings para detectar queries semánticamente similares
    y devolver respuestas cacheadas sin llamar al LLM.
    """
    
    def __init__(
        self,
        cache_dir: Path,
        config: Optional[SemanticCacheConfig] = None
    ):
        """
        Args:
            cache_dir: Directorio para almacenar el caché
            config: Configuración del caché
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or SemanticCacheConfig()
        self.db_path = self.cache_dir / "semantic_cache.db"
        
        # Caché en memoria (LRU simple)
        self._memory_cache: Dict[str, CachedResponse] = {}
        self._memory_order: List[str] = []  # Para LRU
        
        # Cliente de embeddings (lazy init)
        self._embedding_client = None
        
        # Matriz de embeddings para búsqueda rápida
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._cache_ids: List[str] = []
        
        # Inicializar DB
        self._init_db()
        
        # Cargar embeddings en memoria
        self._load_embeddings_matrix()
        
        # Estadísticas
        self.stats = {
            "hits": 0,
            "misses": 0,
            "insertions": 0
        }
    
    def _init_db(self):
        """Inicializa la base de datos SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                query_embedding BLOB NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                sources TEXT NOT NULL,
                routing_info TEXT NOT NULL,
                tokens_input INTEGER,
                tokens_output INTEGER,
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON cache_entries(created_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_embeddings_matrix(self):
        """Carga todos los embeddings en una matriz numpy para búsqueda rápida."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT cache_id, query_embedding FROM cache_entries"
        )
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            self._embeddings_matrix = None
            self._cache_ids = []
            return
        
        embeddings = []
        self._cache_ids = []
        
        for cache_id, emb_blob in rows:
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
            embeddings.append(embedding)
            self._cache_ids.append(cache_id)
        
        self._embeddings_matrix = np.array(embeddings)
        logger.debug(
            f"Cargados {len(self._cache_ids)} embeddings en matriz"
        )
    
    def _get_embedding_client(self):
        """Obtiene el cliente de embeddings (lazy init)."""
        if self._embedding_client is None:
            from openai import OpenAI
            self._embedding_client = OpenAI()
        return self._embedding_client
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Calcula el embedding de un texto."""
        client = self._get_embedding_client()
        
        response = client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
            dimensions=self.config.embedding_dimensions
        )
        
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def _cosine_similarity(
        self, 
        query_embedding: np.ndarray,
        matrix: np.ndarray
    ) -> np.ndarray:
        """Calcula similitud coseno entre query y matriz de embeddings."""
        # Normalizar
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        
        # Producto escalar = similitud coseno (ya normalizados)
        similarities = np.dot(matrix_norms, query_norm)
        
        return similarities
    
    def _generate_cache_id(self, query: str) -> str:
        """Genera ID único para una entrada de caché."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def get(
        self, 
        query: str,
        compute_embedding: bool = True
    ) -> Tuple[Optional[CachedResponse], float]:
        """
        Busca una respuesta cacheada semánticamente similar.
        
        Args:
            query: Query a buscar
            compute_embedding: Si calcular embedding (False para testing)
            
        Returns:
            Tupla (CachedResponse o None, similarity_score)
        """
        # Verificar caché en memoria primero (exacto)
        cache_id = self._generate_cache_id(query)
        if cache_id in self._memory_cache:
            self.stats["hits"] += 1
            cached = self._memory_cache[cache_id]
            self._update_access(cache_id)
            return cached, 1.0
        
        # Si no hay embeddings cacheados, es miss
        if self._embeddings_matrix is None or len(self._cache_ids) == 0:
            self.stats["misses"] += 1
            return None, 0.0
        
        # Calcular embedding de la query
        if compute_embedding:
            try:
                query_embedding = self._compute_embedding(query)
            except Exception as e:
                logger.warning(f"Error calculando embedding: {e}")
                self.stats["misses"] += 1
                return None, 0.0
        else:
            self.stats["misses"] += 1
            return None, 0.0
        
        # Buscar similares
        similarities = self._cosine_similarity(
            query_embedding, 
            self._embeddings_matrix
        )
        
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        logger.debug(
            f"Mejor similitud: {max_similarity:.4f} "
            f"(umbral: {self.config.similarity_threshold})"
        )
        
        # Verificar si supera el umbral
        if max_similarity >= self.config.similarity_threshold:
            cache_id = self._cache_ids[max_idx]
            cached = self._load_entry(cache_id)
            
            if cached:
                self.stats["hits"] += 1
                self._update_access(cache_id)
                
                # Añadir a memoria
                if self.config.use_memory_cache:
                    self._add_to_memory_cache(cache_id, cached)
                
                logger.info(
                    f"Cache HIT semántico (sim={max_similarity:.3f}): "
                    f"'{query[:50]}...' → '{cached.query[:50]}...'"
                )
                return cached, float(max_similarity)
        
        self.stats["misses"] += 1
        return None, float(max_similarity)
    
    def put(
        self,
        query: str,
        response: str,
        model: str,
        sources: List[Dict[str, Any]],
        routing_info: Dict[str, Any],
        tokens_input: int = 0,
        tokens_output: int = 0
    ) -> str:
        """
        Añade una respuesta al caché.
        
        Args:
            query: Query original
            response: Respuesta del LLM
            model: Modelo usado
            sources: Fuentes usadas
            routing_info: Info de routing
            tokens_input: Tokens de entrada
            tokens_output: Tokens de salida
            
        Returns:
            cache_id de la entrada
        """
        cache_id = self._generate_cache_id(query)
        
        # Calcular embedding
        try:
            query_embedding = self._compute_embedding(query)
        except Exception as e:
            logger.error(f"Error calculando embedding para caché: {e}")
            return ""
        
        now = datetime.now().isoformat()
        
        cached = CachedResponse(
            query=query,
            query_embedding=query_embedding.tolist(),
            response=response,
            model=model,
            sources=sources,
            routing_info=routing_info,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            created_at=now,
            access_count=1,
            last_accessed=now
        )
        
        # Guardar en DB
        self._save_entry(cache_id, cached, query_embedding)
        
        # Actualizar matriz de embeddings
        if self._embeddings_matrix is None:
            self._embeddings_matrix = query_embedding.reshape(1, -1)
            self._cache_ids = [cache_id]
        else:
            self._embeddings_matrix = np.vstack([
                self._embeddings_matrix,
                query_embedding.reshape(1, -1)
            ])
            self._cache_ids.append(cache_id)
        
        # Añadir a memoria
        if self.config.use_memory_cache:
            self._add_to_memory_cache(cache_id, cached)
        
        self.stats["insertions"] += 1
        
        # Limpiar entradas antiguas si es necesario
        self._cleanup_if_needed()
        
        logger.debug(f"Cacheado query: '{query[:50]}...'")
        return cache_id
    
    def _save_entry(
        self, 
        cache_id: str, 
        cached: CachedResponse,
        embedding: np.ndarray
    ):
        """Guarda una entrada en la DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache_entries 
            (cache_id, query, query_embedding, response, model, sources,
             routing_info, tokens_input, tokens_output, created_at, 
             access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cache_id,
            cached.query,
            embedding.tobytes(),
            cached.response,
            cached.model,
            json.dumps(cached.sources),
            json.dumps(cached.routing_info),
            cached.tokens_input,
            cached.tokens_output,
            cached.created_at,
            cached.access_count,
            cached.last_accessed
        ))
        
        conn.commit()
        conn.close()
    
    def _load_entry(self, cache_id: str) -> Optional[CachedResponse]:
        """Carga una entrada de la DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query, query_embedding, response, model, sources,
                   routing_info, tokens_input, tokens_output, created_at,
                   access_count, last_accessed
            FROM cache_entries WHERE cache_id = ?
        """, (cache_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return CachedResponse(
            query=row[0],
            query_embedding=np.frombuffer(row[1], dtype=np.float32).tolist(),
            response=row[2],
            model=row[3],
            sources=json.loads(row[4]),
            routing_info=json.loads(row[5]),
            tokens_input=row[6],
            tokens_output=row[7],
            created_at=row[8],
            access_count=row[9],
            last_accessed=row[10] or ""
        )
    
    def _update_access(self, cache_id: str):
        """Actualiza contador de acceso y timestamp."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE cache_entries 
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE cache_id = ?
        """, (datetime.now().isoformat(), cache_id))
        
        conn.commit()
        conn.close()
    
    def _add_to_memory_cache(self, cache_id: str, cached: CachedResponse):
        """Añade entrada al caché en memoria (LRU)."""
        # Si ya existe, mover al final
        if cache_id in self._memory_cache:
            self._memory_order.remove(cache_id)
            self._memory_order.append(cache_id)
            return
        
        # Si está lleno, eliminar el más antiguo
        while len(self._memory_order) >= self.config.memory_cache_size:
            old_id = self._memory_order.pop(0)
            del self._memory_cache[old_id]
        
        self._memory_cache[cache_id] = cached
        self._memory_order.append(cache_id)
    
    def _cleanup_if_needed(self):
        """Limpia entradas antiguas si se excede el límite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Contar entradas
        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        count = cursor.fetchone()[0]
        
        if count > self.config.max_entries:
            # Eliminar las más antiguas (sin acceso reciente)
            to_delete = count - self.config.max_entries
            cursor.execute("""
                DELETE FROM cache_entries WHERE cache_id IN (
                    SELECT cache_id FROM cache_entries
                    ORDER BY last_accessed ASC, created_at ASC
                    LIMIT ?
                )
            """, (to_delete,))
            
            conn.commit()
            logger.info(f"Limpiadas {to_delete} entradas antiguas del caché")
            
            # Recargar matriz
            conn.close()
            self._load_embeddings_matrix()
            return
        
        # Limpiar por TTL si está configurado
        if self.config.ttl_days > 0:
            cutoff = (
                datetime.now() - timedelta(days=self.config.ttl_days)
            ).isoformat()
            
            cursor.execute("""
                DELETE FROM cache_entries 
                WHERE created_at < ? AND access_count < 5
            """, (cutoff,))
            
            if cursor.rowcount > 0:
                logger.info(f"Expiradas {cursor.rowcount} entradas por TTL")
                conn.commit()
                conn.close()
                self._load_embeddings_matrix()
                return
        
        conn.close()
    
    def invalidate(self, query: str) -> bool:
        """
        Invalida una entrada específica del caché.
        
        Args:
            query: Query a invalidar
            
        Returns:
            True si se eliminó, False si no existía
        """
        cache_id = self._generate_cache_id(query)
        
        # Eliminar de memoria
        if cache_id in self._memory_cache:
            del self._memory_cache[cache_id]
            self._memory_order.remove(cache_id)
        
        # Eliminar de DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM cache_entries WHERE cache_id = ?", 
            (cache_id,)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            self._load_embeddings_matrix()
        
        return deleted
    
    def clear(self):
        """Limpia todo el caché."""
        self._memory_cache.clear()
        self._memory_order.clear()
        self._embeddings_matrix = None
        self._cache_ids = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache_entries")
        conn.commit()
        conn.close()
        
        self.stats = {"hits": 0, "misses": 0, "insertions": 0}
        logger.info("Caché semántico limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas del caché."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        total_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(access_count) FROM cache_entries")
        total_accesses = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT SUM(tokens_input), SUM(tokens_output) 
            FROM cache_entries
        """)
        row = cursor.fetchone()
        total_tokens_saved = (row[0] or 0) + (row[1] or 0)
        
        conn.close()
        
        hit_rate = 0.0
        total_requests = self.stats["hits"] + self.stats["misses"]
        if total_requests > 0:
            hit_rate = self.stats["hits"] / total_requests
        
        return {
            "total_entries": total_entries,
            "memory_entries": len(self._memory_cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "insertions": self.stats["insertions"],
            "hit_rate": hit_rate,
            "total_accesses": total_accesses,
            "estimated_tokens_saved": total_tokens_saved * self.stats["hits"],
            "similarity_threshold": self.config.similarity_threshold
        }


# Singleton
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache(
    cache_dir: Optional[Path] = None,
    config: Optional[SemanticCacheConfig] = None
) -> SemanticCache:
    """
    Obtiene la instancia singleton del caché semántico.
    
    Args:
        cache_dir: Directorio para el caché
        config: Configuración
        
    Returns:
        Instancia de SemanticCache
    """
    global _semantic_cache
    
    if _semantic_cache is None:
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "indices" / "semantic_cache"
        _semantic_cache = SemanticCache(cache_dir, config)
    
    return _semantic_cache


if __name__ == "__main__":
    # Test básico
    import tempfile
    
    logging.basicConfig(level=logging.DEBUG)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(
            cache_dir=Path(tmpdir),
            config=SemanticCacheConfig(similarity_threshold=0.9)
        )
        
        # Simular caching
        print("\n📝 Test de Caché Semántico")
        print("=" * 50)
        
        # Nota: Este test requiere OPENAI_API_KEY
        try:
            # Cachear una respuesta
            cache.put(
                query="¿Qué es el entrelazamiento cuántico?",
                response="El entrelazamiento cuántico es un fenómeno...",
                model="claude",
                sources=[{"doc": "Nielsen", "section": "2.3"}],
                routing_info={"type": "conceptual"},
                tokens_input=100,
                tokens_output=200
            )
            
            # Buscar con query similar
            cached, similarity = cache.get(
                "Explícame el entrelazamiento cuántico"
            )
            
            if cached:
                print(f"✅ Cache HIT con similitud {similarity:.3f}")
                print(f"   Query original: {cached.query}")
            else:
                print(f"❌ Cache MISS (similitud: {similarity:.3f})")
            
            # Stats
            print(f"\n📊 Estadísticas: {cache.get_stats()}")
            
        except Exception as e:
            print(f"⚠️  Test requiere OPENAI_API_KEY: {e}")
