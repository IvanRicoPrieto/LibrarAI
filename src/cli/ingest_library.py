#!/usr/bin/env python3
"""
ingest_library - Indexa la biblioteca de documentos Markdown.

Uso:
    python -m src.cli.ingest_library [opciones]
    
Ejemplos:
    python -m src.cli.ingest_library                    # Indexar todo
    python -m src.cli.ingest_library --force            # Reindexar todo
    python -m src.cli.ingest_library --stats            # Ver estadísticas
    python -m src.cli.ingest_library --costs            # Ver costes
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Configura rutas del proyecto."""
    project_root = Path(__file__).parent.parent.parent
    
    return {
        "data_dir": project_root / "data",
        "indices_dir": project_root / "indices",
        "config_dir": project_root / "config",
        "logs_dir": project_root / "logs"
    }


def load_config(config_path: Path) -> dict:
    """Carga configuración desde YAML."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"No se pudo cargar config: {e}")
        return {}


def print_banner():
    """Muestra banner del CLI."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║        � LibrarAI - Library Indexer                       ║
║        Indexación de tu biblioteca inteligente            ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_stats(indexer):
    """Muestra estadísticas de la biblioteca."""
    stats = indexer.get_stats()
    
    print("\n📊 Estadísticas de la Biblioteca Indexada")
    print("=" * 50)
    print(f"📅 Última actualización: {stats['last_updated']}")
    print(f"📚 Documentos totales:   {stats['total_documents']}")
    print(f"🧩 Chunks totales:       {stats['total_chunks']}")
    print("\n📑 Documentos indexados:")
    
    for doc in sorted(stats['documents']):
        print(f"   • {doc}")


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Indexa la biblioteca de documentos Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s                     Indexación incremental
  %(prog)s --force             Reindexar todo desde cero
  %(prog)s --stats             Ver estadísticas actuales
  %(prog)s --costs             Ver costes acumulados
  %(prog)s --build-graph       Construir grafo de conocimiento
"""
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Forzar reindexación completa'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Mostrar estadísticas sin indexar'
    )
    
    parser.add_argument(
        '--costs', '-c',
        action='store_true',
        help='Mostrar resumen de costes acumulados'
    )
    
    parser.add_argument(
        '--build-graph', '-g',
        action='store_true',
        help='Construir/reconstruir grafo de conocimiento'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        help='Directorio con archivos Markdown (data/)'
    )
    
    parser.add_argument(
        '--indices-dir', '-i',
        type=Path,
        help='Directorio para índices'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Modo verboso'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simular sin hacer cambios'
    )
    
    parser.add_argument(
        '--semantic-chunking',
        action='store_true',
        help='Usar chunking semántico adaptativo (detecta definiciones, teoremas, etc.)'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        default=True,
        help='Usar indexación paralela (default: activado, 3-5x más rápido)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Desactivar indexación paralela (modo secuencial)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Número de workers para paralelización (default: 4)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=50,
        help='Tamaño de batch para embeddings paralelos (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Cargar variables de entorno
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Variables de entorno cargadas")
    except ImportError:
        logger.warning("python-dotenv no instalado, usando variables del sistema")
    
    # Configurar rutas
    paths = setup_paths()
    data_dir = args.data_dir or paths["data_dir"]
    indices_dir = args.indices_dir or paths["indices_dir"]
    
    # Cargar configuración
    config_path = paths["config_dir"] / "settings.yaml"
    config = load_config(config_path)
    
    # Crear directorios si no existen
    indices_dir.mkdir(parents=True, exist_ok=True)
    
    # Ver costes
    if args.costs:
        try:
            from ..utils.cost_tracker import get_tracker, UsageType
            tracker = get_tracker()
            tracker.print_summary()
            print("\n📊 Costes de indexación (BUILD):")
            tracker.print_summary(UsageType.BUILD)
        except Exception as e:
            logger.error(f"Error mostrando costes: {e}")
        return
    
    print(f"📂 Directorio datos:   {data_dir}")
    print(f"📂 Directorio índices: {indices_dir}")
    
    # Verificar que existe el directorio de datos
    if not data_dir.exists():
        logger.error(f"Directorio no encontrado: {data_dir}")
        sys.exit(1)
    
    # Importar indexador
    try:
        from ..ingestion.indexer import LibraryIndexer
    except ImportError:
        # Fallback para ejecución directa
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.ingestion.indexer import LibraryIndexer
    
    # Configurar indexador
    embedding_config = config.get("embedding", {})
    
    # Determinar si usar paralelización
    use_parallel = not args.no_parallel
    
    indexer = LibraryIndexer(
        indices_dir=indices_dir,
        embedding_provider=embedding_config.get("provider", "openai"),
        embedding_model=embedding_config.get("model", "text-embedding-3-large"),
        embedding_dimensions=embedding_config.get("dimensions", 1536),
        use_graph=config.get("graph", {}).get("enabled", True),
        use_semantic_chunking=args.semantic_chunking,
        parallel_workers=args.workers,
        parallel_batch_size=args.batch_size
    )
    
    if args.semantic_chunking:
        print("🧠 Chunking semántico activado (detecta definiciones, teoremas, etc.)")
    
    if use_parallel:
        print(f"⚡ Indexación paralela activada ({args.workers} workers, batch {args.batch_size})")
    
    # Solo estadísticas
    if args.stats:
        print_stats(indexer)
        return
    
    # Dry run
    if args.dry_run:
        md_files = list(data_dir.rglob("*.md"))
        print(f"\n🔍 Dry run - se encontraron {len(md_files)} archivos:")
        for f in md_files[:20]:
            print(f"   • {f.relative_to(data_dir)}")
        if len(md_files) > 20:
            print(f"   ... y {len(md_files) - 20} más")
        return
    
    # Indexar
    print("\n🚀 Iniciando indexación...")
    start_time = datetime.now()
    
    try:
        stats = indexer.index_library(
            markdown_dir=data_dir,
            incremental=not args.force,
            force=args.force,
            use_parallel=use_parallel
        )
        
        elapsed = stats.get("elapsed_seconds", (datetime.now() - start_time).total_seconds())
        
        print("\n✅ Indexación completada")
        print("=" * 50)
        print(f"⏱️  Tiempo total:         {elapsed:.1f}s")
        print(f"📄 Documentos procesados: {stats['documents_processed']}")
        print(f"⏭️  Documentos omitidos:  {stats['documents_skipped']}")
        print(f"🧩 Chunks creados:        {stats['chunks_created']}")
        
        # Mostrar coste de esta sesión
        try:
            from ..utils.cost_tracker import get_tracker
            tracker = get_tracker()
            summary = tracker.get_summary()
            if summary["by_type"].get("build", 0) > 0:
                print(f"\n💰 Coste de indexación:   ${summary['by_type']['build']:.4f}")
        except Exception:
            pass
        
        if stats['errors']:
            print(f"\n⚠️  Errores ({len(stats['errors'])}):")
            for err in stats['errors'][:5]:
                print(f"   • {err}")
        
    except Exception as e:
        logger.error(f"Error durante indexación: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Construir grafo si se solicita
    if args.build_graph:
        print("\n🕸️  Construyendo grafo de conocimiento...")
        try:
            from ..retrieval.graph_retriever import GraphRetriever
            
            graph_retriever = GraphRetriever(indices_dir)
            graph_retriever.build_graph_from_chunks()
            
            stats = graph_retriever.get_graph_stats()
            print(f"   Nodos: {stats['nodes']}")
            print(f"   Aristas: {stats['edges']}")
            print(f"   Tipos: {stats['entity_types']}")
            
        except Exception as e:
            logger.error(f"Error construyendo grafo: {e}")
    
    print("\n✨ ¡Listo! Usa 'ask_library' para consultar.")


if __name__ == "__main__":
    main()
