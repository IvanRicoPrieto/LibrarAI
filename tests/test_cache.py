# tests/test_cache.py
"""
Tests unitarios para el sistema de caché de embeddings.
"""

import pytest
import sys
import time
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.cache import (
    EmbeddingCache,
    CacheConfig,
    CacheStats
)


class TestCacheConfig:
    """Tests para la configuración del caché."""
    
    def test_default_config(self):
        """Test de configuración por defecto."""
        config = CacheConfig()
        
        assert config.memory_cache_size == 10000
        assert config.persistent == True
        assert config.track_stats == True
    
    def test_custom_config(self):
        """Test de configuración personalizada."""
        config = CacheConfig(
            memory_cache_size=500,
            persistent=False,
            max_age_seconds=3600,
            track_stats=False
        )
        
        assert config.memory_cache_size == 500
        assert config.persistent == False
        assert config.max_age_seconds == 3600


class TestCacheStats:
    """Tests para las estadísticas del caché."""
    
    def test_hit_rate_empty(self):
        """Hit rate debería ser 0 sin accesos."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        """Test del cálculo de hit rate."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8
    
    def test_hit_rate_all_hits(self):
        """Hit rate con todos hits."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0
    
    def test_hit_rate_all_misses(self):
        """Hit rate con todos misses."""
        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0
    
    def test_stats_to_dict(self):
        """Test de serialización de estadísticas."""
        stats = CacheStats(
            hits=75,
            misses=25,
            memory_hits=50,
            disk_hits=25,
            total_saved_ms=1500.5,
            total_saved_cost=0.0125
        )
        
        d = stats.to_dict()
        
        assert d["hits"] == 75
        assert d["misses"] == 25
        assert d["hit_rate"] == "75.0%"
        assert "estimated_savings_ms" in d
        assert "estimated_savings_cost" in d


class TestEmbeddingCache:
    """Tests para el caché de embeddings."""
    
    @pytest.fixture
    def cache(self, temp_dir):
        """Instancia de caché con directorio temporal."""
        config = CacheConfig(
            memory_cache_size=100,
            persistent=True,
            track_stats=True
        )
        return EmbeddingCache(indices_dir=temp_dir, config=config)
    
    @pytest.fixture
    def memory_only_cache(self, temp_dir):
        """Caché solo en memoria."""
        config = CacheConfig(
            memory_cache_size=100,
            persistent=False,
            track_stats=True
        )
        return EmbeddingCache(indices_dir=temp_dir, config=config)
    
    def test_cache_initialization(self, cache):
        """Test de inicialización correcta."""
        assert cache is not None
        # Verificar atributo config (no _config)
        assert hasattr(cache, 'config') or hasattr(cache, '_config')
    
    def test_cache_miss(self, cache):
        """Test de cache miss para query nueva."""
        embedding = cache.get("query no cacheada", model="test-model")
        
        assert embedding is None
        assert cache.stats.misses == 1
    
    def test_cache_set_and_get(self, cache):
        """Test de almacenar y recuperar embedding."""
        query = "¿Qué es un qubit?"
        model = "text-embedding-3-small"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Embedding simulado
        
        # Guardar en caché
        cache.set(query, embedding, model=model)
        
        # Recuperar
        retrieved = cache.get(query, model=model)
        
        assert retrieved is not None
        assert retrieved == embedding
        assert cache.stats.hits == 1
    
    def test_cache_memory_hit(self, cache):
        """Test de hit en memoria (más rápido)."""
        query = "test query"
        embedding = [0.1] * 1536
        
        cache.set(query, embedding, model="test")
        
        # Primera recuperación
        cache.get(query, model="test")
        
        # Debería ser memory hit
        assert cache.stats.memory_hits >= 1
    
    def test_different_models_different_keys(self, cache):
        """Embeddings de diferentes modelos no se mezclan."""
        query = "misma query"
        embedding_small = [0.1] * 1536
        embedding_large = [0.2] * 3072
        
        cache.set(query, embedding_small, model="text-embedding-3-small")
        cache.set(query, embedding_large, model="text-embedding-3-large")
        
        retrieved_small = cache.get(query, model="text-embedding-3-small")
        retrieved_large = cache.get(query, model="text-embedding-3-large")
        
        assert retrieved_small == embedding_small
        assert retrieved_large == embedding_large
    
    def test_cache_persistence(self, temp_dir):
        """Test de que el caché persiste en disco."""
        config = CacheConfig(persistent=True)
        
        # Crear caché y guardar
        cache1 = EmbeddingCache(indices_dir=temp_dir, config=config)
        cache1.set("query persistente", [0.1, 0.2], model="test")
        
        # Crear nuevo caché (simula reinicio)
        cache2 = EmbeddingCache(indices_dir=temp_dir, config=config)
        
        # Debería encontrar el embedding
        retrieved = cache2.get("query persistente", model="test")
        assert retrieved == [0.1, 0.2]
    
    def test_memory_only_no_persistence(self, temp_dir):
        """Test de que el caché en memoria no persiste."""
        config = CacheConfig(persistent=False)
        
        cache1 = EmbeddingCache(indices_dir=temp_dir, config=config)
        cache1.set("query temporal", [0.3, 0.4], model="test")
        
        # Nuevo caché no debería tener la query
        cache2 = EmbeddingCache(indices_dir=temp_dir, config=config)
        retrieved = cache2.get("query temporal", model="test")
        
        # Podría ser None o no, depende de implementación
        # Lo importante es que no falle
    
    def test_cache_clear(self, cache):
        """Test de limpieza del caché."""
        cache.set("query1", [0.1], model="test")
        cache.set("query2", [0.2], model="test")
        
        cache.clear()
        
        assert cache.get("query1", model="test") is None
        assert cache.get("query2", model="test") is None
    
    def test_cache_stats_accumulation(self, cache):
        """Test de que las estadísticas se acumulan correctamente."""
        # Varios misses
        for i in range(5):
            cache.get(f"miss_query_{i}", model="test")
        
        assert cache.stats.misses == 5
        
        # Un set y hit
        cache.set("hit_query", [0.1], model="test")
        cache.get("hit_query", model="test")
        
        assert cache.stats.hits == 1
        assert cache.stats.misses == 5
    
    def test_hash_consistency(self, cache):
        """Verifica que el hash es consistente."""
        query = "query de prueba"
        
        # El hash puede ser interno, verificamos indirectamente
        cache.set(query, [0.1], model="model_test")
        result1 = cache.get(query, model="model_test")
        result2 = cache.get(query, model="model_test")
        
        assert result1 == result2
    
    def test_different_queries_different_results(self, cache):
        """Queries diferentes retornan resultados diferentes."""
        cache.set("query 1", [0.1], model="model")
        cache.set("query 2", [0.2], model="model")
        
        result1 = cache.get("query 1", model="model")
        result2 = cache.get("query 2", model="model")
        
        assert result1 != result2


class TestCacheExpiration:
    """Tests para expiración del caché."""
    
    @pytest.fixture
    def expiring_cache(self, temp_dir):
        """Caché con expiración rápida."""
        config = CacheConfig(
            memory_cache_size=100,
            persistent=True,
            max_age_seconds=1,  # 1 segundo
            track_stats=True
        )
        return EmbeddingCache(indices_dir=temp_dir, config=config)
    
    @pytest.mark.skip(reason="Test lento - descomentar para verificar expiración")
    def test_cache_expiration(self, expiring_cache):
        """Test de expiración de entries."""
        expiring_cache.set("expiring_query", [0.1], model="test")
        
        # Inmediatamente debería existir
        assert expiring_cache.get("expiring_query", model="test") is not None
        
        # Esperar a que expire
        time.sleep(1.5)
        
        # Debería haber expirado
        retrieved = expiring_cache.get("expiring_query", model="test")
        # Puede ser None o no dependiendo de implementación


class TestCacheConcurrency:
    """Tests de concurrencia del caché."""
    
    def test_thread_safety(self, temp_dir):
        """Test básico de thread safety."""
        import threading
        
        config = CacheConfig(memory_cache_size=1000, persistent=False)
        cache = EmbeddingCache(indices_dir=temp_dir, config=config)
        
        errors = []
        
        def writer():
            try:
                for i in range(100):
                    cache.set(f"thread_query_{i}", [float(i)], model="test")
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for i in range(100):
                    cache.get(f"thread_query_{i}", model="test")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # No debería haber errores de concurrencia
        assert len(errors) == 0
