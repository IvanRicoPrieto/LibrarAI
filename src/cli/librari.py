#!/usr/bin/env python3
"""
librari - CLI optimizado para agentes IA.

Provee 5 modos de operación con output JSON estructurado:
- explore: Descubrir qué contenido existe sobre un tema
- retrieve: Obtener contenido exhaustivo
- query: Responder preguntas con citas
- verify: Verificar afirmaciones contra fuentes
- cite: Generar citas formateadas

Uso:
    python -m src.cli.librari explore "algoritmo de Shor"
    python -m src.cli.librari retrieve "QFT" --exhaustive
    python -m src.cli.librari query "¿Qué es el entrelazamiento?"
    python -m src.cli.librari verify --claim "Shor factoriza en O(n³)"
    python -m src.cli.librari cite --chunks nc_5.2.1,nc_5.2.2 --style apa
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Configurar logging mínimo por defecto (solo errores)
logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Configura rutas del proyecto."""
    project_root = Path(__file__).parent.parent.parent
    return {
        "indices_dir": project_root / "indices",
        "config_dir": project_root / "config",
        "outputs_dir": project_root / "outputs",
    }


def load_env():
    """Carga variables de entorno."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def print_json(data, pretty: bool = True):
    """Imprime datos como JSON."""
    indent = 2 if pretty else None
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def cmd_explore(args):
    """Comando explore: descubrir contenido sobre un tema."""
    from ..api import AgentAPI

    paths = setup_paths()
    api = AgentAPI(indices_dir=paths["indices_dir"])

    result = api.explore(
        topic=args.topic,
        max_depth=args.depth,
        include_summaries=not args.no_summaries,
    )

    print_json(result.to_dict(), not args.compact)


def cmd_retrieve(args):
    """Comando retrieve: obtener contenido."""
    from ..api import AgentAPI

    paths = setup_paths()
    api = AgentAPI(indices_dir=paths["indices_dir"])

    result = api.retrieve(
        query=args.query,
        exhaustive=args.exhaustive,
        max_chunks=args.max_chunks,
        min_score=args.min_score,
        expand_context=not args.no_expand,
        difficulty_level=args.level,
        math_aware=args.math_aware,
    )

    print_json(result.to_dict(), not args.compact)


def cmd_query(args):
    """Comando query: responder preguntas."""
    from ..api import AgentAPI

    paths = setup_paths()
    api = AgentAPI(
        indices_dir=paths["indices_dir"],
        use_grounding=args.grounded,
    )

    result = api.query(
        question=args.question,
        top_k=args.top_k,
        require_citations=args.grounded,
        min_confidence=args.min_confidence,
        difficulty_level=args.level,
        math_aware=args.math_aware,
    )

    print_json(result.to_dict(), not args.compact)


def cmd_verify(args):
    """Comando verify: verificar afirmaciones."""
    from ..api import AgentAPI

    paths = setup_paths()
    api = AgentAPI(indices_dir=paths["indices_dir"])

    result = api.verify(
        claim=args.claim,
        max_sources=args.max_sources,
    )

    print_json(result.to_dict(), not args.compact)


def cmd_cite(args):
    """Comando cite: generar citas formateadas."""
    from ..api import AgentAPI

    paths = setup_paths()
    api = AgentAPI(indices_dir=paths["indices_dir"])

    chunk_ids = [c.strip() for c in args.chunks.split(",")]

    result = api.cite(
        chunk_ids=chunk_ids,
        style=args.style,
    )

    print_json(result.to_dict(), not args.compact)


def cmd_stats(args):
    """Comando stats: mostrar estadísticas de la biblioteca."""
    paths = setup_paths()

    stats = {
        "timestamp": datetime.now().isoformat(),
        "indices_dir": str(paths["indices_dir"]),
    }

    # Contar chunks
    chunks_path = paths["indices_dir"] / "chunks.pkl"
    if chunks_path.exists():
        import pickle
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        stats["total_chunks"] = len(chunks)

        # Contar por documento
        docs = {}
        for c in chunks.values():
            doc_id = getattr(c, 'doc_id', 'unknown')
            docs[doc_id] = docs.get(doc_id, 0) + 1
        stats["documents"] = len(docs)
        stats["chunks_per_document"] = docs

    # Contar en Qdrant
    try:
        from qdrant_client import QdrantClient
        import os

        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            client = QdrantClient(url=qdrant_url)
        else:
            client = QdrantClient(path=str(paths["indices_dir"] / "qdrant"))

        collection_info = client.get_collection("quantum_library")
        stats["qdrant"] = {
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
        }
    except Exception as e:
        stats["qdrant_error"] = str(e)

    # Verificar grafo
    graph_path = paths["indices_dir"] / "knowledge_graph.gpickle"
    stats["knowledge_graph_exists"] = graph_path.exists()

    # Verificar índice jerárquico
    hierarchical_path = paths["indices_dir"] / "hierarchical_index.json"
    stats["hierarchical_index_exists"] = hierarchical_path.exists()

    print_json(stats, not args.compact)


def main():
    """Punto de entrada principal."""
    load_env()

    parser = argparse.ArgumentParser(
        prog="librari",
        description="LibrarAI - API CLI para agentes IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  librari explore "algoritmo de Shor"
  librari retrieve "QFT" --exhaustive --format json
  librari query "¿Qué es el entrelazamiento?" --grounded
  librari verify --claim "Shor factoriza en O(n³)"
  librari cite --chunks nc_5.2.1,nc_5.2.2 --style apa
  librari stats

Output: Todos los comandos producen JSON estructurado para fácil parsing.
""",
    )

    # Flags globales
    parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Output JSON compacto (sin indentación)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verboso (logs adicionales)",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Comandos disponibles",
        metavar="COMANDO",
    )

    # =========================================================================
    # EXPLORE
    # =========================================================================
    explore_parser = subparsers.add_parser(
        "explore",
        help="Descubrir qué contenido existe sobre un tema",
        description="""
Explora la biblioteca para descubrir qué contenido existe sobre un tema.
Devuelve una estructura jerárquica de documentos, secciones y chunks relevantes.
        """,
    )
    explore_parser.add_argument(
        "topic",
        help="Tema a explorar",
    )
    explore_parser.add_argument(
        "--depth", "-d",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Profundidad de exploración: 1=docs, 2=+chapters, 3=+sections (default: 3)",
    )
    explore_parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="No incluir resúmenes de contenido",
    )
    explore_parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Output JSON compacto",
    )
    explore_parser.set_defaults(func=cmd_explore)

    # =========================================================================
    # RETRIEVE
    # =========================================================================
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Obtener contenido relevante",
        description="""
Recupera chunks de contenido relevantes para un tema.
En modo exhaustivo, recupera TODO el contenido relevante sin límite tradicional.
        """,
    )
    retrieve_parser.add_argument(
        "query",
        help="Query de búsqueda",
    )
    retrieve_parser.add_argument(
        "--exhaustive", "-e",
        action="store_true",
        help="Modo exhaustivo: recuperar todo el contenido relevante",
    )
    retrieve_parser.add_argument(
        "--max-chunks", "-n",
        type=int,
        default=50,
        help="Máximo de chunks a devolver (default: 50, ignorado si --exhaustive)",
    )
    retrieve_parser.add_argument(
        "--min-score",
        type=float,
        default=0.001,
        help="Score mínimo de relevancia (default: 0.001)",
    )
    retrieve_parser.add_argument(
        "--no-expand",
        action="store_true",
        help="No expandir chunks a contexto coherente",
    )
    retrieve_parser.add_argument(
        "--level", "-l",
        type=str,
        default=None,
        help="Filtrar por nivel: introductory, intermediate, advanced, research (o combinaciones: 'introductory,intermediate')",
    )
    retrieve_parser.add_argument(
        "--math-aware", "-m",
        action="store_true",
        help="Expandir query con términos matemáticos equivalentes",
    )
    retrieve_parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Output JSON compacto",
    )
    retrieve_parser.set_defaults(func=cmd_retrieve)

    # =========================================================================
    # QUERY
    # =========================================================================
    query_parser = subparsers.add_parser(
        "query",
        help="Responder preguntas con citas",
        description="""
Genera una respuesta a una pregunta con afirmaciones citadas.
Usa --grounded para forzar citas verificables en cada afirmación.
        """,
    )
    query_parser.add_argument(
        "question",
        help="Pregunta a responder",
    )
    query_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Número de fuentes a considerar (default: 10)",
    )
    query_parser.add_argument(
        "--grounded", "-g",
        action="store_true",
        help="Forzar citas verificables en cada afirmación",
    )
    query_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Confianza mínima para responder (default: 0.5)",
    )
    query_parser.add_argument(
        "--level", "-l",
        type=str,
        default=None,
        help="Filtrar por nivel: introductory, intermediate, advanced, research",
    )
    query_parser.add_argument(
        "--math-aware", "-m",
        action="store_true",
        help="Expandir query con términos matemáticos equivalentes",
    )
    query_parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Output JSON compacto",
    )
    query_parser.set_defaults(func=cmd_query)

    # =========================================================================
    # VERIFY
    # =========================================================================
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verificar afirmaciones contra fuentes",
        description="""
Verifica si una afirmación está soportada por las fuentes.
Devuelve estado (supported/contradicted/not_found/partial) y evidencia.
        """,
    )
    verify_parser.add_argument(
        "--claim", "-c",
        required=True,
        help="Afirmación a verificar",
    )
    verify_parser.add_argument(
        "--max-sources",
        type=int,
        default=20,
        help="Máximo de fuentes a revisar (default: 20)",
    )
    verify_parser.add_argument(
        "--compact",
        action="store_true",
        help="Output JSON compacto",
    )
    verify_parser.set_defaults(func=cmd_verify)

    # =========================================================================
    # CITE
    # =========================================================================
    cite_parser = subparsers.add_parser(
        "cite",
        help="Generar citas formateadas",
        description="""
Genera citas formateadas para chunks específicos.
Estilos disponibles: apa, ieee, chicago, markdown, inline.
        """,
    )
    cite_parser.add_argument(
        "--chunks",
        required=True,
        help="IDs de chunks separados por comas (ej: nc_5.2.1,nc_5.2.2)",
    )
    cite_parser.add_argument(
        "--style", "-s",
        choices=["apa", "ieee", "chicago", "markdown", "inline"],
        default="apa",
        help="Estilo de citación (default: apa)",
    )
    cite_parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Output JSON compacto",
    )
    cite_parser.set_defaults(func=cmd_cite)

    # =========================================================================
    # STATS
    # =========================================================================
    stats_parser = subparsers.add_parser(
        "stats",
        help="Mostrar estadísticas de la biblioteca",
        description="Muestra estadísticas sobre la biblioteca indexada.",
    )
    stats_parser.add_argument(
        "--compact", "-c",
        action="store_true",
        help="Output JSON compacto",
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Parsear argumentos
    args = parser.parse_args()

    # Configurar logging si verbose
    if getattr(args, 'verbose', False):
        logging.getLogger().setLevel(logging.INFO)

    # Ejecutar comando
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        error_output = {
            "error": str(e),
            "command": args.command,
            "timestamp": datetime.now().isoformat(),
        }
        print_json(error_output)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
