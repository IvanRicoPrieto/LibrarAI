"""
Indexador de Biblioteca - Crea índices vectoriales, BM25 y grafo.

Este módulo se encarga de:
1. Generar embeddings para chunks (con paralelización opcional)
2. Almacenar en Qdrant (vector DB) - local o remoto (Docker/Cloud)
3. Construir índice BM25 para búsqueda léxica
4. Construir grafo de conocimiento (opcional)
5. Gestionar manifest para indexación incremental
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

from tqdm import tqdm

from .parser import MarkdownParser, ParsedDocument, parse_library
from .chunker import HierarchicalChunker, Chunk, ChunkLevel
from ..utils.cost_tracker import get_tracker, UsageType

logger = logging.getLogger(__name__)


@dataclass
class DocumentManifest:
    """Registro de un documento indexado."""
    doc_id: str
    file_path: str
    content_hash: str
    chunk_count: int
    indexed_at: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentManifest":
        return cls(**data)


@dataclass 
class LibraryManifest:
    """Manifest completo de la biblioteca indexada."""
    last_updated: str
    total_documents: int
    total_chunks: int
    documents: Dict[str, DocumentManifest]
    
    def to_dict(self) -> Dict:
        return {
            "last_updated": self.last_updated,
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "documents": {
                k: v.to_dict() for k, v in self.documents.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LibraryManifest":
        documents = {
            k: DocumentManifest.from_dict(v) 
            for k, v in data.get("documents", {}).items()
        }
        return cls(
            last_updated=data.get("last_updated", ""),
            total_documents=data.get("total_documents", 0),
            total_chunks=data.get("total_chunks", 0),
            documents=documents
        )
    
    @classmethod
    def empty(cls) -> "LibraryManifest":
        return cls(
            last_updated="",
            total_documents=0,
            total_chunks=0,
            documents={}
        )


class LibraryIndexer:
    """
    Indexador principal que coordina todos los índices.
    
    Componentes:
    - Qdrant: índice vectorial
    - BM25: índice léxico
    - NetworkX: grafo de conocimiento
    - Manifest: tracking de documentos indexados
    """
    
    def __init__(
        self,
        indices_dir: Path,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 1536,
        qdrant_collection: str = "quantum_library",
        use_graph: bool = True,
        qdrant_url: str = None,
        use_semantic_chunking: bool = False,
        parallel_workers: int = 4,
        parallel_batch_size: int = 50
    ):
        """
        Args:
            indices_dir: Directorio para almacenar índices
            embedding_provider: "openai" o "local"
            embedding_model: Modelo de embeddings
            embedding_dimensions: Dimensiones del vector
            qdrant_collection: Nombre de la colección en Qdrant
            use_graph: Si crear grafo de conocimiento
            qdrant_url: URL de Qdrant remoto (ej: http://localhost:6333). Si None, usa local.
            use_semantic_chunking: Si usar chunking semántico adaptativo
            parallel_workers: Número de workers para embeddings paralelos (default: 4)
            parallel_batch_size: Tamaño de batch por worker (default: 50)
        """
        self.indices_dir = Path(indices_dir)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.qdrant_collection = qdrant_collection
        self.use_graph = use_graph
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.use_semantic_chunking = use_semantic_chunking
        self.parallel_workers = parallel_workers
        self.parallel_batch_size = parallel_batch_size
        
        # Rutas de archivos
        self.manifest_path = self.indices_dir / "manifest.json"
        self.bm25_path = self.indices_dir / "bm25_index.pkl"
        self.chunks_path = self.indices_dir / "chunks.pkl"
        self.graph_path = self.indices_dir / "knowledge_graph.gpickle"
        
        # Componentes (se inicializan lazy)
        self._qdrant_client = None
        self._embedding_client = None
        self._bm25_index = None
        self._chunks_store: Dict[str, Chunk] = {}
        self._graph = None
        
        # Parser y chunker
        self.parser = MarkdownParser()
        if use_semantic_chunking:
            from .semantic_chunker import SemanticChunker
            self.chunker = SemanticChunker()
            logger.info("Usando chunking semántico adaptativo")
        else:
            self.chunker = HierarchicalChunker()
        
        # Cargar manifest existente
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> LibraryManifest:
        """Carga el manifest existente o crea uno nuevo."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)
                return LibraryManifest.from_dict(data)
        return LibraryManifest.empty()
    
    def _save_manifest(self):
        """Guarda el manifest actual."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest.to_dict(), f, indent=2)
    
    def _init_qdrant(self):
        """Inicializa cliente de Qdrant (local o remoto)."""
        if self._qdrant_client is not None:
            return
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, PointStruct
            )
            
            if self.qdrant_url:
                # Conexión remota (Docker o cloud)
                self._qdrant_client = QdrantClient(url=self.qdrant_url)
                logger.info(f"Conectado a Qdrant remoto: {self.qdrant_url}")
            else:
                # Almacenamiento local (file-based)
                qdrant_path = self.indices_dir / "qdrant"
                qdrant_path.mkdir(exist_ok=True)
                self._qdrant_client = QdrantClient(path=str(qdrant_path))
                logger.debug("Qdrant local inicializado")
            
            # Crear colección si no existe
            collections = self._qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.qdrant_collection not in collection_names:
                self._qdrant_client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.embedding_dimensions,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Creada colección Qdrant: {self.qdrant_collection}")
            
        except ImportError:
            logger.error("qdrant-client no instalado. Ejecuta: pip install qdrant-client")
            raise
    
    def _init_embeddings(self):
        """Inicializa cliente de embeddings."""
        if self._embedding_client is not None:
            return
        
        if self.embedding_provider == "openai":
            try:
                from openai import OpenAI
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY no configurada. "
                        "Copia .env.example a .env y configura tu API key."
                    )
                
                self._embedding_client = OpenAI(api_key=api_key)
                logger.info(f"Embeddings: OpenAI {self.embedding_model}")
                
            except ImportError:
                logger.error("openai no instalado. Ejecuta: pip install openai")
                raise
        else:
            # Embeddings locales con sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                
                self._embedding_client = SentenceTransformer(
                    self.embedding_model
                )
                logger.info(f"Embeddings: Local {self.embedding_model}")
                
            except ImportError:
                logger.error(
                    "sentence-transformers no instalado. "
                    "Ejecuta: pip install sentence-transformers"
                )
                raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Genera embedding para un texto."""
        self._init_embeddings()
        
        if self.embedding_provider == "openai":
            response = self._embedding_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        else:
            # Sentence transformers
            return self._embedding_client.encode(text).tolist()
    
    def _get_embeddings_single_batch(
        self,
        batch_texts: List[str],
        batch_idx: int
    ) -> Tuple[int, List[List[float]], int]:
        """
        Procesa un solo batch de embeddings. Usado por workers paralelos.
        
        Args:
            batch_texts: Textos del batch
            batch_idx: Índice del batch para ordenación
            
        Returns:
            Tuple (batch_idx, embeddings, tokens_used)
        """
        if self.embedding_provider == "openai":
            response = self._embedding_client.embeddings.create(
                input=batch_texts,
                model=self.embedding_model
            )
            embeddings = [d.embedding for d in response.data]
            tokens = response.usage.total_tokens
        else:
            embeddings = self._embedding_client.encode(batch_texts).tolist()
            tokens = sum(len(t) // 4 for t in batch_texts)
        
        return batch_idx, embeddings, tokens
    
    def _get_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        use_parallel: bool = True
    ) -> List[List[float]]:
        """
        Genera embeddings en batch con soporte para paralelización.
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño de batch para API (default: 100)
            use_parallel: Si usar procesamiento paralelo (default: True)
            
        Returns:
            Lista de embeddings
        """
        self._init_embeddings()
        
        tracker = get_tracker()
        total_tokens = 0
        
        # Dividir en batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append((i // batch_size, texts[i:i + batch_size]))
        
        # Modo paralelo
        if use_parallel and len(batches) > 1 and self.parallel_workers > 1:
            logger.info(f"⚡ Embeddings paralelos: {len(batches)} batches × {self.parallel_workers} workers")
            
            # Usar batch_size más pequeño para paralelización
            parallel_batch_size = min(batch_size, self.parallel_batch_size)
            batches = []
            for i in range(0, len(texts), parallel_batch_size):
                batches.append((i // parallel_batch_size, texts[i:i + parallel_batch_size]))
            
            results = {}
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                # Enviar todos los batches
                futures = {
                    executor.submit(
                        self._get_embeddings_single_batch, 
                        batch_texts, 
                        batch_idx
                    ): batch_idx
                    for batch_idx, batch_texts in batches
                }
                
                # Recolectar resultados con barra de progreso
                with tqdm(total=len(batches), desc="Generando embeddings (paralelo)") as pbar:
                    for future in as_completed(futures):
                        try:
                            batch_idx, embeddings, tokens = future.result()
                            results[batch_idx] = embeddings
                            total_tokens += tokens
                            pbar.update(1)
                        except Exception as e:
                            batch_idx = futures[future]
                            logger.error(f"Error en batch {batch_idx}: {e}")
                            # Reintentar de forma secuencial
                            batch_texts = batches[batch_idx][1]
                            _, embeddings, tokens = self._get_embeddings_single_batch(
                                batch_texts, batch_idx
                            )
                            results[batch_idx] = embeddings
                            total_tokens += tokens
                            pbar.update(1)
            
            elapsed = time.time() - start_time
            logger.info(f"⚡ Embeddings completados en {elapsed:.1f}s ({len(texts)/elapsed:.1f} textos/s)")
            
            # Ordenar resultados por batch_idx
            all_embeddings = []
            for i in range(len(batches)):
                all_embeddings.extend(results[i])
        
        # Modo secuencial (fallback o batches pequeños)
        else:
            all_embeddings = []
            for batch_idx, batch_texts in tqdm(batches, desc="Generando embeddings"):
                if self.embedding_provider == "openai":
                    response = self._embedding_client.embeddings.create(
                        input=batch_texts,
                        model=self.embedding_model
                    )
                    batch_embeddings = [d.embedding for d in response.data]
                    total_tokens += response.usage.total_tokens
                else:
                    batch_embeddings = self._embedding_client.encode(batch_texts).tolist()
                    total_tokens += sum(len(t) // 4 for t in batch_texts)
                
                all_embeddings.extend(batch_embeddings)
        
        # Registrar coste total de embeddings (BUILD)
        if total_tokens > 0:
            tracker.record_embedding(
                model=self.embedding_model,
                tokens=total_tokens,
                usage_type=UsageType.BUILD
            )
        
        return all_embeddings
    
    def _init_bm25(self, chunks: Optional[List[Chunk]] = None):
        """Inicializa o carga índice BM25."""
        if self._bm25_index is not None:
            return
        
        if self.bm25_path.exists() and chunks is None:
            with open(self.bm25_path, 'rb') as f:
                self._bm25_index = pickle.load(f)
            logger.info("Índice BM25 cargado desde disco")
        elif chunks:
            self._build_bm25(chunks)
    
    def _build_bm25(self, chunks: List[Chunk]):
        """Construye índice BM25 desde chunks."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 no instalado. Ejecuta: pip install rank-bm25")
            raise
        
        # Tokenizar documentos (simple split)
        tokenized_corpus = [
            chunk.content.lower().split() 
            for chunk in chunks
        ]
        
        self._bm25_index = BM25Okapi(tokenized_corpus)
        
        # Guardar
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self._bm25_index, f)
        
        logger.info(f"Índice BM25 construido: {len(chunks)} documentos")
    
    def _init_graph(self):
        """Inicializa grafo de conocimiento."""
        if not self.use_graph:
            return
        
        if self._graph is not None:
            return
        
        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx no instalado. Ejecuta: pip install networkx")
            raise
        
        if self.graph_path.exists():
            with open(self.graph_path, 'rb') as f:
                self._graph = pickle.load(f)
            logger.info("Grafo de conocimiento cargado desde disco")
        else:
            self._graph = nx.DiGraph()
            logger.info("Grafo de conocimiento inicializado (vacío)")
    
    def _save_graph(self):
        """Guarda grafo a disco."""
        if self._graph is not None:
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self._graph, f)
    
    def _load_chunks_store(self):
        """Carga chunks almacenados."""
        if self.chunks_path.exists():
            with open(self.chunks_path, 'rb') as f:
                self._chunks_store = pickle.load(f)
    
    def _save_chunks_store(self):
        """Guarda chunks a disco."""
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self._chunks_store, f)
    
    def index_library(
        self,
        markdown_dir: Path,
        incremental: bool = True,
        force: bool = False,
        use_parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Indexa toda la biblioteca de documentos.
        
        Args:
            markdown_dir: Directorio con archivos .md
            incremental: Si True, solo procesa documentos nuevos/modificados
            force: Si True, reindexar todo ignorando manifest
            use_parallel: Si True, usa embeddings paralelos (3-5x más rápido)
            
        Returns:
            Estadísticas de indexación
        """
        import time
        start_time = time.time()
        
        markdown_dir = Path(markdown_dir)
        stats = {
            "documents_processed": 0,
            "documents_skipped": 0,
            "chunks_created": 0,
            "errors": [],
            "elapsed_seconds": 0
        }
        
        # Inicializar componentes
        self._init_qdrant()
        self._load_chunks_store()
        self._init_graph()
        
        # Encontrar archivos Markdown
        md_files = list(markdown_dir.rglob("*.md"))
        logger.info(f"Encontrados {len(md_files)} archivos Markdown")
        
        all_new_chunks = []
        
        for file_path in tqdm(md_files, desc="Procesando documentos"):
            try:
                # Calcular hash del archivo
                content = file_path.read_text(encoding='utf-8')
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                
                # Verificar si necesita re-indexar
                relative_path = str(file_path.relative_to(markdown_dir))
                
                if incremental and not force:
                    if relative_path in self.manifest.documents:
                        existing = self.manifest.documents[relative_path]
                        if existing.content_hash == file_hash:
                            stats["documents_skipped"] += 1
                            continue
                
                # Parsear y chunkear
                doc = self.parser.parse_file(file_path)
                chunks = self.chunker.chunk_document(doc)
                
                # Solo indexar chunks MICRO
                micro_chunks = self.chunker.get_micro_chunks(chunks)
                
                # Guardar todos los chunks (para auto-merge)
                for chunk in chunks:
                    self._chunks_store[chunk.chunk_id] = chunk
                
                all_new_chunks.extend(micro_chunks)
                
                # Actualizar manifest
                self.manifest.documents[relative_path] = DocumentManifest(
                    doc_id=doc.doc_id,
                    file_path=relative_path,
                    content_hash=file_hash,
                    chunk_count=len(micro_chunks),
                    indexed_at=datetime.now().isoformat()
                )
                
                stats["documents_processed"] += 1
                stats["chunks_created"] += len(micro_chunks)
                
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {e}")
                stats["errors"].append(str(file_path))
        
        # Generar embeddings e indexar en Qdrant
        if all_new_chunks:
            logger.info(f"Indexando {len(all_new_chunks)} chunks en Qdrant...")
            # Construir mapeo doc_id -> file_path para categorías
            file_paths = {
                m.doc_id: m.file_path 
                for m in self.manifest.documents.values()
            }
            self._index_chunks_to_qdrant(all_new_chunks, file_paths, use_parallel=use_parallel)
        
        # Construir/actualizar BM25
        all_micro_chunks = [
            c for c in self._chunks_store.values() 
            if c.level == ChunkLevel.MICRO
        ]
        self._build_bm25(all_micro_chunks)
        
        # Actualizar manifest
        self.manifest.last_updated = datetime.now().isoformat()
        self.manifest.total_documents = len(self.manifest.documents)
        self.manifest.total_chunks = len(all_micro_chunks)
        
        # Guardar todo
        self._save_manifest()
        self._save_chunks_store()
        self._save_graph()
        
        # Calcular tiempo total
        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)
        
        logger.info(
            f"Indexación completada en {elapsed:.1f}s: "
            f"{stats['documents_processed']} procesados, "
            f"{stats['documents_skipped']} omitidos, "
            f"{stats['chunks_created']} chunks creados"
        )
        
        return stats
    
    def _index_chunks_to_qdrant(
        self, 
        chunks: List[Chunk], 
        file_paths: Dict[str, str] = None,
        use_parallel: bool = True
    ):
        """
        Indexa chunks en Qdrant.
        
        Args:
            chunks: Lista de chunks a indexar
            file_paths: Diccionario doc_id -> file_path para extraer categoría
            use_parallel: Si usar embeddings paralelos
        """
        from qdrant_client.models import PointStruct
        
        # Generar embeddings (paralelo o secuencial)
        texts = [c.content for c in chunks]
        embeddings = self._get_embeddings_batch(texts, use_parallel=use_parallel)
        
        # Crear puntos para Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Extraer categoría del file_path (primera carpeta)
            category = "general"
            if file_paths and chunk.doc_id in file_paths:
                path = file_paths[chunk.doc_id]
                parts = path.split("/")
                if len(parts) > 1:
                    # Usar primera subcarpeta como categoría
                    category = parts[0]
            
            point = PointStruct(
                id=hash(chunk.chunk_id) % (2**63),  # ID numérico
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "header_path": chunk.header_path,
                    "parent_id": chunk.parent_id,
                    "level": chunk.level.value,
                    "token_count": chunk.token_count,
                    "category": category
                }
            )
            points.append(point)
        
        # Insertar en batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._qdrant_client.upsert(
                collection_name=self.qdrant_collection,
                points=batch
            )
        
        logger.info(f"Indexados {len(points)} vectores en Qdrant")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la biblioteca indexada."""
        return {
            "last_updated": self.manifest.last_updated,
            "total_documents": self.manifest.total_documents,
            "total_chunks": self.manifest.total_chunks,
            "documents": list(self.manifest.documents.keys())
        }


if __name__ == "__main__":
    import sys
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    # Cargar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    if len(sys.argv) > 1:
        markdown_dir = Path(sys.argv[1])
        indices_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./indices")
        
        indexer = LibraryIndexer(indices_dir)
        stats = indexer.index_library(markdown_dir)
        
        print("\n📊 Estadísticas de indexación:")
        print(f"   Documentos procesados: {stats['documents_processed']}")
        print(f"   Documentos omitidos: {stats['documents_skipped']}")
        print(f"   Chunks creados: {stats['chunks_created']}")
        if stats['errors']:
            print(f"   Errores: {len(stats['errors'])}")
