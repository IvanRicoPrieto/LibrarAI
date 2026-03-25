#!/usr/bin/env python3
"""
ask_library - Consulta tu biblioteca inteligente con LibrarAI.

Uso:
    python -m src.cli.ask_library "tu pregunta"
    python -m src.cli.ask_library --interactive

Ejemplos:
    python -m src.cli.ask_library "¿Qué es el algoritmo de Shor?"
    python -m src.cli.ask_library "Compara BB84 con E91" --model claude
    python -m src.cli.ask_library --interactive
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Re-exportar componentes extraídos para compatibilidad con importadores existentes
from .rag_pipeline import RAGPipeline
from .formatters import print_banner, format_response, save_session
from .code_executor import execute_code_blocks
from .interactive import interactive_mode


def setup_paths():
    """Configura rutas del proyecto."""
    project_root = Path(__file__).parent.parent.parent

    return {
        "indices_dir": project_root / "indices",
        "config_dir": project_root / "config",
        "logs_dir": project_root / "logs",
        "outputs_dir": project_root / "outputs"
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


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Consulta tu biblioteca inteligente usando LibrarAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s "¿Qué es el algoritmo de Shor?"
  %(prog)s "Compara BB84 con E91" --model gpt-4.1 --deep
  %(prog)s "Entrelazamiento cuántico" --sources
  %(prog)s "Calcula la entropía de von Neumann" --exec
  %(prog)s --interactive
  %(prog)s "Explica el entrelazamiento" --critic --save
"""
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Pregunta a realizar'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Modo interactivo'
    )

    parser.add_argument(
        '--model', '-m',
        choices=['claude', 'gpt-4.1', 'gpt-4.1-mini', 'local'],
        default='claude',
        help='Modelo a usar (default: claude = Claude Sonnet 4.5)'
    )

    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=10,
        help='Número de documentos a recuperar (default: 10)'
    )

    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='No mostrar fuentes'
    )

    parser.add_argument(
        '--sources',
        action='store_true',
        help='Solo mostrar fuentes relevantes sin generar respuesta (ahorra costes)'
    )

    parser.add_argument(
        '--deep',
        action='store_true',
        help='Modo Deep Research: descompone queries complejas y busca iterativamente'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Streaming de respuesta'
    )

    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Guardar sesión'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Salida en formato JSON'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Modo verboso'
    )

    parser.add_argument(
        '--no-router',
        action='store_true',
        help='Desactivar router inteligente'
    )

    parser.add_argument(
        '--critic',
        action='store_true',
        help='Activar crítico de respuestas con validación de citas'
    )

    parser.add_argument(
        '--exec',
        action='store_true',
        help='Permitir ejecución de código (cálculos, gráficas, simulaciones)'
    )

    parser.add_argument(
        '--rerank',
        action='store_true',
        help='Aplicar re-ranking con cross-encoder para mejorar precisión (+15-25%%)'
    )

    parser.add_argument(
        '--rerank-preset',
        choices=['fast', 'balanced', 'quality', 'max_quality'],
        default='balanced',
        help='Preset del reranker: fast (rápido), balanced (default), quality, max_quality'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Deshabilitar cache de embeddings'
    )

    parser.add_argument(
        '--filter', '-f',
        type=str,
        action='append',
        dest='filters',
        metavar='KEY:VALUE',
        help='Filtrar por metadatos (ej: --filter category:computacion_cuantica). Puede usarse múltiples veces'
    )

    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='Listar categorías disponibles y salir'
    )

    parser.add_argument(
        '--hyde',
        action='store_true',
        help='Activar HyDE (Hypothetical Document Embeddings) para mejorar recall (+10-20%% en queries abstractas)'
    )

    parser.add_argument(
        '--hyde-domain',
        choices=['quantum_computing', 'quantum_information', 'quantum_cryptography', 'general_physics', 'mathematics'],
        default='quantum_computing',
        help='Dominio para generación HyDE (default: quantum_computing)'
    )

    parser.add_argument(
        '--compress',
        action='store_true',
        help='Comprimir contexto para incluir más chunks en el presupuesto de tokens (reduce 30-50%%)'
    )

    parser.add_argument(
        '--compress-level',
        choices=['light', 'medium', 'aggressive'],
        default='medium',
        help='Nivel de compresión de contexto: light (~20%%), medium (~40%%), aggressive (~60%%)'
    )

    parser.add_argument(
        '--cache-stats',
        action='store_true',
        help='Mostrar estadísticas del cache de embeddings'
    )

    parser.add_argument(
        '--semantic-cache',
        action='store_true',
        default=True,
        help='Activar caché semántico (default: activado). Reutiliza respuestas para queries similares'
    )

    parser.add_argument(
        '--no-semantic-cache',
        action='store_true',
        help='Desactivar caché semántico (fuerza generación nueva)'
    )

    parser.add_argument(
        '--cache-threshold',
        type=float,
        default=0.92,
        help='Umbral de similitud para cache semántico (default: 0.92). Valores más altos = más estricto'
    )

    parser.add_argument(
        '--semantic-cache-stats',
        action='store_true',
        help='Mostrar estadísticas del caché semántico y salir'
    )

    parser.add_argument(
        '--clear-semantic-cache',
        action='store_true',
        help='Limpiar el caché semántico y salir'
    )

    parser.add_argument(
        '--crag',
        action='store_true',
        help='Activar Corrective RAG: evalúa relevancia de resultados y reformula si necesario'
    )

    parser.add_argument(
        '--agentic',
        action='store_true',
        help='Activar Agentic RAG: loop iterativo Retrieve→Reflect→Decide para queries complejas'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=4,
        help='Máximo de iteraciones para Agentic RAG (default: 4)'
    )

    parser.add_argument(
        '--use-propositions',
        action='store_true',
        help='Incluir búsqueda en proposiciones atómicas (requiere indexación con --propositions)'
    )

    parser.add_argument(
        '--export-context',
        action='store_true',
        help='Exportar fragmentos de contexto coherentes usados como fuentes (usa LLM para cortes inteligentes)'
    )

    parser.add_argument(
        '--context-output',
        type=str,
        default=None,
        help='Archivo para guardar contextos exportados (default: stdout o auto-generado)'
    )

    parser.add_argument(
        '--costs', '-c',
        action='store_true',
        help='Mostrar resumen de costes acumulados de consultas'
    )

    # === Nuevas mejoras avanzadas ===

    parser.add_argument(
        '--multi-query',
        action='store_true',
        help='Multi-Query RAG: expande query en variaciones para mejorar recall'
    )

    parser.add_argument(
        '--query-variations',
        type=int,
        default=4,
        help='Número de variaciones de query para Multi-Query RAG (default: 4)'
    )

    parser.add_argument(
        '--self-rag',
        action='store_true',
        help='Self-RAG: modo auto-reflexivo que evalúa y refina retrieval/respuesta'
    )

    parser.add_argument(
        '--colbert-rerank',
        action='store_true',
        help='Usar ColBERT para re-rankear resultados (mejor precisión, requiere ragatouille)'
    )

    parser.add_argument(
        '--grounded',
        action='store_true',
        help='Citation Grounding: fuerza citas verificables [n] en cada afirmación'
    )

    parser.add_argument(
        '--min-grounding',
        type=float,
        default=0.8,
        help='Score mínimo de grounding para Citation Grounding (default: 0.8)'
    )

    parser.add_argument(
        '--hierarchical',
        action='store_true',
        help='Usar routing jerárquico por resúmenes antes de búsqueda fina'
    )

    args = parser.parse_args()

    # Cargar variables de entorno ANTES de cualquier operación
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Si solo pide costes, mostrar y salir
    if args.costs and not args.query and not args.interactive:
        try:
            from ..utils.cost_tracker import get_tracker
            tracker = get_tracker()
            tracker.print_summary()  # Sin filtro, muestra todo
        except Exception as e:
            print(f"❌ Error mostrando costes: {e}")
        sys.exit(0)

    # Si solo pide estadísticas de cache, mostrar y salir
    if args.cache_stats and not args.query and not args.interactive:
        try:
            paths = setup_paths()
            from ..retrieval.cache import EmbeddingCache
            cache = EmbeddingCache(indices_dir=paths["indices_dir"])
            stats = cache.get_stats()
            print("\n📊 Estadísticas del Cache de Embeddings")
            print("=" * 45)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            cache.close()
        except Exception as e:
            print(f"❌ Error mostrando estadísticas de cache: {e}")
        sys.exit(0)

    # Si pide estadísticas del caché semántico
    if args.semantic_cache_stats and not args.query and not args.interactive:
        try:
            paths = setup_paths()
            from ..retrieval.semantic_cache import SemanticCache
            cache_dir = paths["indices_dir"] / "semantic_cache"
            if cache_dir.exists():
                cache = SemanticCache(cache_dir)
                stats = cache.get_stats()
                print("\n💾 Estadísticas del Caché Semántico")
                print("=" * 45)
                for key, value in stats.items():
                    if key == "hit_rate" and value is not None:
                        print(f"  {key}: {value:.1%}")
                    else:
                        print(f"  {key}: {value}")
                cache.close()
            else:
                print("\n💾 Caché semántico no inicializado aún")
                print("   Se creará con la primera consulta")
        except Exception as e:
            print(f"❌ Error mostrando estadísticas: {e}")
        sys.exit(0)

    # Si pide limpiar el caché semántico
    if args.clear_semantic_cache and not args.query and not args.interactive:
        try:
            paths = setup_paths()
            from ..retrieval.semantic_cache import SemanticCache
            cache_dir = paths["indices_dir"] / "semantic_cache"
            if cache_dir.exists():
                cache = SemanticCache(cache_dir)
                cleared = cache.clear()
                print(f"\n🗑️ Caché semántico limpiado: {cleared} entradas eliminadas")
                cache.close()
            else:
                print("\n💾 Caché semántico no existe")
        except Exception as e:
            print(f"❌ Error limpiando caché: {e}")
        sys.exit(0)

    # Si pide listar categorías, mostrar y salir
    if args.list_categories:
        try:
            paths = setup_paths()
            from qdrant_client import QdrantClient
            import os

            # Usar QDRANT_URL si está definido (Docker), sino local
            qdrant_url = os.getenv("QDRANT_URL")
            if qdrant_url:
                client = QdrantClient(url=qdrant_url)
            else:
                client = QdrantClient(path=str(paths["indices_dir"] / "qdrant"))

            # Obtener categorías únicas haciendo scroll
            categories = set()
            offset = None
            while True:
                results, offset = client.scroll(
                    collection_name="quantum_library",
                    limit=1000,
                    offset=offset,
                    with_payload=["category"]
                )
                for point in results:
                    if point.payload and "category" in point.payload:
                        categories.add(point.payload["category"])
                if offset is None:
                    break

            print("\n📁 Categorías Disponibles")
            print("=" * 35)
            for cat in sorted(categories):
                print(f"  • {cat}")
            print(f"\nTotal: {len(categories)} categorías")
            print("\nUso: --filter category:CATEGORIA")
        except Exception as e:
            print(f"❌ Error listando categorías: {e}")
        sys.exit(0)

    # Verificar que hay query o modo interactivo
    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(1)

    # Configurar rutas
    paths = setup_paths()

    # Verificar que existen los índices
    if not paths["indices_dir"].exists():
        print("❌ No se encontraron índices. Ejecuta primero:")
        print("   python -m src.cli.ingest_library")
        sys.exit(1)

    # Cargar configuración
    config = load_config(paths["config_dir"] / "settings.yaml")

    # Mapear modelo
    model_map = {
        "claude": "claude-sonnet-4-5-20250929",  # Claude Sonnet 4.5
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "local": "llama3.2"  # Ollama
    }
    model = model_map.get(args.model, args.model)

    # Determinar si usar caché semántico
    use_semantic_cache = not args.no_semantic_cache

    # Determinar si usar compresión
    compress_context = args.compress
    compress_level = args.compress_level

    # Determinar si usar CRAG y agentic desde config o CLI
    use_crag = args.crag or config.get("corrective_rag", {}).get("enabled", False)
    use_agentic = args.agentic or config.get("agentic_rag", {}).get("enabled", False)
    use_propositions = args.use_propositions or config.get("proposition_indexing", {}).get("enabled", False)

    # Determinar nuevas mejoras desde config o CLI
    use_multi_query = args.multi_query or config.get("multi_query", {}).get("enabled", False)
    use_self_rag = args.self_rag or config.get("self_rag", {}).get("enabled", False)
    use_colbert_rerank = args.colbert_rerank or config.get("colbert", {}).get("enabled", False)
    use_grounded = args.grounded or config.get("citation_grounding", {}).get("enabled", False)
    use_hierarchical = args.hierarchical or config.get("hierarchical_index", {}).get("enabled", False)

    # Crear pipeline
    try:
        pipeline = RAGPipeline(
            indices_dir=paths["indices_dir"],
            config=config,
            model=model,
            use_router=not args.no_router,
            use_critic=args.critic,
            use_reranker=args.rerank,
            reranker_preset=args.rerank_preset,
            use_cache=not args.no_cache,
            use_hyde=args.hyde,
            hyde_domain=args.hyde_domain,
            use_semantic_cache=use_semantic_cache,
            semantic_cache_threshold=args.cache_threshold,
            compress_context=compress_context,
            compress_level=compress_level,
            use_crag=use_crag,
            use_agentic=use_agentic,
            agentic_max_iterations=args.max_iterations,
            use_propositions=use_propositions,
            # Nuevas mejoras avanzadas
            use_multi_query=use_multi_query,
            multi_query_variations=args.query_variations,
            use_self_rag=use_self_rag,
            use_colbert_rerank=use_colbert_rerank,
            use_grounded_citations=use_grounded,
            min_grounding_score=args.min_grounding,
            use_hierarchical_routing=use_hierarchical
        )
    except Exception as e:
        logger.error(f"Error inicializando pipeline: {e}")
        sys.exit(1)

    # Verificar redundancias y advertir
    redundancy_warnings = []
    if use_agentic and use_crag:
        redundancy_warnings.append("⚠️  CRAG es redundante con Agentic RAG (Agentic ya incluye corrección)")
        use_crag = False  # Desactivar para evitar doble procesamiento
    if use_agentic and use_self_rag:
        redundancy_warnings.append("⚠️  Self-RAG es redundante con Agentic RAG (Agentic ya incluye auto-reflexión)")
        use_self_rag = False

    # Mostrar modos activos (solo si no JSON)
    if not args.json and (args.query or args.interactive):
        # Mostrar warnings de redundancia
        for warn in redundancy_warnings:
            print(warn)

        if use_crag:
            print("🔍 Corrective RAG activado")
        if use_agentic:
            print(f"🤖 Agentic RAG activado (max {args.max_iterations} iteraciones)")
        if use_propositions:
            print("💎 Búsqueda en proposiciones activada")
        # Nuevas mejoras
        if use_multi_query:
            print(f"🔀 Multi-Query RAG activado ({args.query_variations} variaciones)")
        if use_self_rag:
            print("🪞 Self-RAG activado (modo auto-reflexivo)")
        if use_colbert_rerank:
            print("🎯 ColBERT reranking activado")
        if use_grounded:
            print(f"📎 Citation Grounding activado (min: {args.min_grounding})")
        if use_hierarchical:
            print("📊 Hierarchical routing activado")

    # Modo interactivo
    if args.interactive:
        print_banner()
        interactive_mode(
            pipeline,
            args.save,
            paths["outputs_dir"]
        )
        return

    # Modo single query
    if not args.json:
        print_banner()
        print(f"❓ Pregunta: {args.query}")
        if args.sources:
            print("\n🔍 Buscando fuentes (sin generar respuesta)...")
        elif args.deep:
            print("\n🔬 Modo Deep Research: analizando consulta...")
        else:
            print("\n🔍 Buscando en la biblioteca...")

    try:
        # Callback para streaming
        stream_callback = None
        if args.stream and not args.json and not args.sources:
            def stream_callback(text):
                print(text, end='', flush=True)
            print("\n📝 Respuesta:")
            print("─" * 60)

        # Parsear filtros de args
        filters = None
        if args.filters:
            filters = {}
            for f in args.filters:
                if ':' in f:
                    key, value = f.split(':', 1)
                    filters[key.strip()] = value.strip()
                else:
                    logger.warning(f"Filtro ignorado (formato inválido): {f}")

        # Llamar al pipeline con modo deep research si aplica
        if args.deep:
            response, sources, routing = pipeline.ask_deep(
                args.query,
                top_k=args.top_k,
                max_iterations=3
            )
        else:
            response, sources, routing = pipeline.ask(
                args.query,
                top_k=args.top_k,
                stream=args.stream,
                stream_callback=stream_callback,
                sources_only=args.sources,
                filters=filters
            )

        # Export context: expandir fuentes a fragmentos coherentes con LLM
        expanded_contexts = None
        if args.export_context and sources:
            try:
                from ..retrieval.context_expander import ContextExpander

                if not args.json:
                    print("\n🔄 Expandiendo contextos con LLM...")

                expander = ContextExpander(
                    indices_dir=paths["indices_dir"],
                    chunks_before=3,
                    chunks_after=3,
                    max_context_tokens=2000
                )

                expanded_contexts = expander.expand_retrieval_results(
                    sources,
                    args.query
                )

                # Preparar output
                context_output_data = {
                    "query": args.query,
                    "timestamp": datetime.now().isoformat(),
                    "total_expanded": len(expanded_contexts),
                    "contexts": [ctx.to_dict() for ctx in expanded_contexts]
                }

                # Determinar dónde escribir
                if args.context_output:
                    context_path = Path(args.context_output)
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    context_path = paths["outputs_dir"] / f"context_{timestamp}.json"

                context_path.parent.mkdir(parents=True, exist_ok=True)

                with open(context_path, 'w', encoding='utf-8') as f:
                    json.dump(context_output_data, ensure_ascii=False, indent=2, fp=f)

                if not args.json:
                    print(f"✅ Contextos exportados a: {context_path}")
                    print(f"   {len(expanded_contexts)} fragmentos coherentes extraídos")

                    # Mostrar resumen de cada contexto
                    print("\n📄 Resumen de contextos exportados:")
                    print("─" * 50)
                    for i, ctx in enumerate(expanded_contexts, 1):
                        print(f"\n[{i}] {ctx.source_citation}")
                        print(f"    📝 {ctx.topic_summary}")
                        print(f"    🎯 {ctx.relevance_to_query[:100]}..." if len(ctx.relevance_to_query) > 100 else f"    🎯 {ctx.relevance_to_query}")
                        print(f"    📊 {ctx.token_count} tokens")
                    print("─" * 50)

            except Exception as e:
                logger.warning(f"Error expandiendo contextos: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        if args.json:
            output = {
                "query": args.query,
                "response": response.to_dict(),
                "sources": [s.to_dict() for s in sources],
                "routing": routing.to_dict() if routing else None
            }
            # Incluir contextos expandidos si se generaron
            if expanded_contexts:
                output["expanded_contexts"] = [ctx.to_dict() for ctx in expanded_contexts]
            print(json.dumps(output, ensure_ascii=False, indent=2))
        elif args.sources:
            # Modo solo fuentes: mostrar fuentes detalladas
            print("\n" + "═" * 60)
            print("📚 FUENTES RELEVANTES")
            print("═" * 60)
            for i, s in enumerate(sources, 1):
                print(f"\n[{i}] {s.doc_title}")
                print(f"    📍 {s.header_path}")
                print(f"    📊 Score: {s.score:.3f} | Nivel: {s.level}")
                print(f"    📝 Preview:")
                preview = s.content[:300].replace('\n', ' ')
                print(f"       {preview}...")
            print("\n" + "═" * 60)
            print(f"📈 Total: {len(sources)} fuentes | Max score: {max(s.score for s in sources):.3f}")
            print("💡 Usa sin --sources para generar una respuesta completa")
        else:
            if not args.stream:
                print(format_response(response, sources, not args.no_sources))
            else:
                print("\n" + "─" * 60)
                print(f"\n📊 Modelo: {response.model}")
                print(f"⏱️  Latencia: {response.latency_ms:.0f}ms")

        # Ejecutar código si se solicita y la respuesta contiene bloques de código
        if args.exec and not args.sources:
            execute_code_blocks(response.content, paths["outputs_dir"])

        # Guardar sesión
        if args.save:
            save_session(args.query, response, sources, paths["outputs_dir"])
            if not args.json:
                print(f"\n💾 Sesión guardada en {paths['outputs_dir']}")

        # Mostrar costes si se solicita
        if args.costs and not args.json:
            try:
                from ..utils.cost_tracker import get_tracker
                tracker = get_tracker()
                print("\n" + "─" * 60)
                tracker.print_summary(usage_type="QUERY")
            except Exception:
                pass

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            logger.error(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
