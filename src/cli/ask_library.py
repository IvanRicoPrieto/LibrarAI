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
from typing import Optional, List

# Configurar logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


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


def execute_code_blocks(content: str, outputs_dir: Path):
    """
    Detecta y ejecuta bloques de código Python en la respuesta.
    
    Args:
        content: Contenido de la respuesta con posibles bloques ```python
        outputs_dir: Directorio para guardar figuras generadas
    """
    import re
    
    # Buscar bloques de código Python
    code_pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
    code_blocks = code_pattern.findall(content)
    
    if not code_blocks:
        return
    
    print("\n" + "═" * 60)
    print("🖥️  EJECUCIÓN DE CÓDIGO")
    print("═" * 60)
    
    try:
        from ..execution.sandbox import CodeSandbox
        sandbox = CodeSandbox(
            timeout_seconds=30,
            outputs_dir=outputs_dir
        )
        
        for i, code in enumerate(code_blocks, 1):
            print(f"\n[Bloque {i}]")
            print(f"{'─' * 40}")
            
            # Validar antes de ejecutar
            is_valid, error = sandbox.validate_code(code)
            if not is_valid:
                print(f"⚠️ Código no ejecutado (seguridad): {error}")
                continue
            
            # Preguntar al usuario si quiere ejecutar
            print(f"Código a ejecutar:")
            print(f"```python\n{code[:500]}{'...' if len(code) > 500 else ''}\n```")
            
            try:
                confirm = input("\n¿Ejecutar este código? [s/N]: ").strip().lower()
            except EOFError:
                confirm = 'n'
            
            if confirm not in ['s', 'si', 'sí', 'y', 'yes']:
                print("⏭️ Omitido")
                continue
            
            print("\n⏳ Ejecutando...")
            result = sandbox.execute(code)
            
            if result.success:
                print("✅ Ejecución exitosa")
                if result.stdout:
                    print(f"\n📤 Salida:\n{result.stdout}")
                if result.figures:
                    print(f"\n📊 {len(result.figures)} figura(s) generada(s)")
                    # Guardar figuras
                    for j, fig_b64 in enumerate(result.figures, 1):
                        fig_path = outputs_dir / f"figure_{i}_{j}.png"
                        import base64
                        with open(fig_path, 'wb') as f:
                            f.write(base64.b64decode(fig_b64))
                        print(f"   Guardada: {fig_path}")
            else:
                print(f"❌ Error: {result.error_type}")
                if result.stderr:
                    print(f"   {result.stderr[:200]}")
            
            print(f"⏱️ Tiempo: {result.execution_time_ms:.0f}ms")
    
    except ImportError as e:
        print(f"⚠️ Módulo de ejecución no disponible: {e}")
    except Exception as e:
        print(f"❌ Error ejecutando código: {e}")
    
    print("\n" + "═" * 60)


def print_banner():
    """Muestra banner del CLI."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║        � LibrarAI - Ask Library                           ║
║        Tu biblioteca inteligente con IA                   ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def format_response(response, sources, show_sources: bool = True, show_cost: bool = True) -> str:
    """Formatea la respuesta para mostrar."""
    output = []
    
    # Respuesta principal
    output.append("\n📝 Respuesta:")
    output.append("─" * 60)
    output.append(response.content)
    output.append("─" * 60)
    
    # Mostrar crítica si existe
    if response.metadata and "critique" in response.metadata:
        critique = response.metadata["critique"]
        output.append("\n🔍 EVALUACIÓN DE CALIDAD:")
        output.append("─" * 40)
        
        # Score general
        if "overall_score" in critique:
            score = critique["overall_score"]
            stars = "★" * int(score * 5) + "☆" * (5 - int(score * 5))
            output.append(f"  📊 Score: {score:.2f} {stars}")
        
        # Cobertura de citas
        if "citation_coverage" in critique:
            cov = critique["citation_coverage"]
            output.append(f"  📖 Citas usadas: {cov*100:.0f}%")
        
        # Fundamentación
        if "grounded_score" in critique:
            gs = critique["grounded_score"]
            output.append(f"  🎯 Fundamentación: {gs*100:.0f}%")
        
        # Fortalezas
        if critique.get("strengths"):
            output.append("\n  ✅ Fortalezas:")
            for s in critique["strengths"][:3]:
                output.append(f"     • {s}")
        
        # Problemas
        if critique.get("issues"):
            output.append("\n  ⚠️  Problemas detectados:")
            for issue in critique["issues"][:3]:
                if isinstance(issue, dict):
                    severity = issue.get("severity", "?")
                    desc = issue.get("description", "Sin descripción")
                    icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                    output.append(f"     {icon} {desc}")
                else:
                    output.append(f"     • {issue}")
        
        # Sugerencias
        if critique.get("suggestions"):
            output.append("\n  💡 Sugerencias:")
            for sug in critique["suggestions"][:2]:
                output.append(f"     • {sug}")
        
        output.append("─" * 40)
    
    # Metadatos
    output.append(f"\n📊 Modelo: {response.model}")
    output.append(f"🔢 Tokens: {response.tokens_input} entrada, {response.tokens_output} salida")
    output.append(f"⏱️  Latencia: {response.latency_ms:.0f}ms")
    
    # Coste estimado
    if show_cost:
        try:
            from ..utils.cost_tracker import get_tracker
            tracker = get_tracker()
            cost = tracker.calculate_cost(
                response.model,
                response.tokens_input,
                response.tokens_output
            )
            output.append(f"💰 Coste: ${cost:.4f}")
        except Exception:
            pass
    
    # Fuentes
    if show_sources and sources:
        output.append(f"\n📚 Fuentes ({len(sources)}):")
        for i, source in enumerate(sources[:5], 1):
            title = source.doc_title[:40] + "..." if len(source.doc_title) > 40 else source.doc_title
            output.append(f"   [{i}] {title}")
            output.append(f"       📍 {source.header_path}")
    
    return "\n".join(output)


def save_session(
    query: str,
    response,
    sources,
    output_path: Path
):
    """Guarda la sesión a un archivo."""
    session = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response.to_dict(),
        "sources": [s.to_dict() for s in sources]
    }
    
    # Crear directorio si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Añadir a archivo de sesión
    sessions_file = output_path / "sessions.jsonl"
    with open(sessions_file, 'a') as f:
        f.write(json.dumps(session, ensure_ascii=False) + "\n")


class RAGPipeline:
    """Pipeline RAG completo."""

    def __init__(
        self,
        indices_dir: Path,
        config: dict,
        model: str = "claude-sonnet-4-5-20250929",
        use_router: bool = True,
        use_critic: bool = False,
        use_reranker: bool = False,
        reranker_preset: str = "balanced",
        use_cache: bool = True,
        use_hyde: bool = False,
        hyde_domain: str = "quantum_computing",
        use_semantic_cache: bool = True,
        semantic_cache_threshold: float = 0.92,
        compress_context: bool = False,
        compress_level: str = "medium",
        use_crag: bool = False,
        use_agentic: bool = False,
        agentic_max_iterations: int = 4,
        use_propositions: bool = False,
        # Nuevas mejoras avanzadas
        use_multi_query: bool = False,
        multi_query_variations: int = 4,
        use_self_rag: bool = False,
        use_colbert_rerank: bool = False,
        use_grounded_citations: bool = False,
        min_grounding_score: float = 0.8,
        use_hierarchical_routing: bool = False
    ):
        self.indices_dir = indices_dir
        self.config = config
        self.model = model
        self.use_router = use_router
        self.use_critic = use_critic
        self.use_reranker = use_reranker
        self.reranker_preset = reranker_preset
        self.use_cache = use_cache
        self.use_hyde = use_hyde
        self.hyde_domain = hyde_domain
        self.use_semantic_cache = use_semantic_cache
        self.semantic_cache_threshold = semantic_cache_threshold
        self.compress_context = compress_context
        self.compress_level = compress_level
        self.use_crag = use_crag
        self.use_agentic = use_agentic
        self.agentic_max_iterations = agentic_max_iterations
        self.use_propositions = use_propositions
        # Nuevas mejoras
        self.use_multi_query = use_multi_query
        self.multi_query_variations = multi_query_variations
        self.use_self_rag = use_self_rag
        self.use_colbert_rerank = use_colbert_rerank
        self.use_grounded_citations = use_grounded_citations
        self.min_grounding_score = min_grounding_score
        self.use_hierarchical_routing = use_hierarchical_routing
        self._context_budget = config.get("generation", {}).get("context_budget", 16000)

        # Componentes (lazy init)
        self._retriever = None
        self._synthesizer = None
        self._router = None
        self._critic = None
        self._citation_injector = None
        self._semantic_cache = None
        self._context_compressor = None
        self._corrective_rag = None
        self._agentic_pipeline = None
        # Nuevos componentes
        self._multi_query_expander = None
        self._self_rag_pipeline = None
        self._colbert_reranker = None
        self._grounding_system = None
        self._hierarchical_index = None
    
    def _init_components(self):
        """Inicializa componentes del pipeline."""
        if self._retriever is not None:
            return
        
        from ..retrieval.fusion import UnifiedRetriever
        from ..generation.synthesizer import ResponseSynthesizer
        from ..generation.citation_injector import CitationInjector
        from ..agents.router import QueryRouter
        from ..agents.critic import ResponseCritic
        
        retrieval_config = self.config.get("retrieval", {})
        
        # Determinar si usar reranker: CLI flag tiene prioridad sobre config
        use_reranker = self.use_reranker or retrieval_config.get("reranker", {}).get("enabled", False)
        reranker_preset = self.reranker_preset or retrieval_config.get("reranker", {}).get("preset", "balanced")
        
        # Determinar si usar HyDE: CLI flag tiene prioridad sobre config
        use_hyde = self.use_hyde or retrieval_config.get("hyde", {}).get("enabled", False)
        hyde_domain = self.hyde_domain or retrieval_config.get("hyde", {}).get("domain", "quantum_computing")
        
        self._retriever = UnifiedRetriever(
            indices_dir=self.indices_dir,
            vector_weight=retrieval_config.get("vector_weight", 0.5),
            bm25_weight=retrieval_config.get("bm25_weight", 0.3),
            graph_weight=retrieval_config.get("graph_weight", 0.2),
            use_graph=self.config.get("graph", {}).get("enabled", True),
            use_reranker=use_reranker,
            reranker_preset=reranker_preset,
            use_cache=self.use_cache,
            use_hyde=use_hyde,
            hyde_domain=hyde_domain
        )
        
        gen_config = self.config.get("generation", {})
        
        self._synthesizer = ResponseSynthesizer(
            temperature=gen_config.get("temperature", 0.3),
            max_output_tokens=gen_config.get("max_tokens", 2000)
        )
        
        self._citation_injector = CitationInjector()
        
        if self.use_router:
            self._router = QueryRouter(use_llm_router=False)
        
        if self.use_critic:
            self._critic = ResponseCritic(use_llm_critic=False)
        
        # Caché semántico
        if self.use_semantic_cache:
            try:
                from ..retrieval.semantic_cache import (
                    SemanticCache, SemanticCacheConfig
                )
                cache_config = SemanticCacheConfig(
                    similarity_threshold=self.semantic_cache_threshold
                )
                cache_dir = self.indices_dir / "semantic_cache"
                self._semantic_cache = SemanticCache(cache_dir, cache_config)
                logger.debug("Caché semántico inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar caché semántico: {e}")
                self._semantic_cache = None
        
        # Compresor de contexto
        if self.compress_context:
            try:
                from ..generation.context_compressor import (
                    ContextCompressor, CompressionConfig, CompressionLevel
                )
                level_map = {
                    "light": CompressionLevel.LIGHT,
                    "medium": CompressionLevel.MEDIUM,
                    "aggressive": CompressionLevel.AGGRESSIVE
                }
                config = CompressionConfig(
                    level=level_map.get(self.compress_level, CompressionLevel.MEDIUM)
                )
                self._context_compressor = ContextCompressor(config)
                logger.info(f"Compresor de contexto inicializado (nivel: {self.compress_level})")
            except Exception as e:
                logger.warning(f"No se pudo inicializar compresor: {e}")
                self._context_compressor = None

        # Corrective RAG
        if self.use_crag:
            try:
                from ..retrieval.corrective_rag import CorrectiveRAG
                crag_config = self.config.get("corrective_rag", {})
                self._corrective_rag = CorrectiveRAG(
                    retriever=self._retriever,
                    correct_threshold=crag_config.get("correct_threshold", 0.7),
                    incorrect_threshold=crag_config.get("incorrect_threshold", 0.3),
                    use_llm_assessment=crag_config.get("use_llm", True)
                )
                logger.info("Corrective RAG inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar CRAG: {e}")
                self._corrective_rag = None

        # Agentic RAG
        if self.use_agentic:
            try:
                from ..agents.agentic_rag import AgenticRAGPipeline
                agentic_config = self.config.get("agentic_rag", {})
                self._agentic_pipeline = AgenticRAGPipeline(
                    retriever=self._retriever,
                    synthesizer=self._synthesizer,
                    max_iterations=self.agentic_max_iterations,
                    confidence_threshold=agentic_config.get("confidence_threshold", 0.7),
                    min_new_results=agentic_config.get("min_new_results", 2)
                )
                logger.info("Agentic RAG inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Agentic RAG: {e}")
                self._agentic_pipeline = None

        # === Nuevas mejoras avanzadas ===

        # Multi-Query RAG
        if self.use_multi_query:
            try:
                from ..retrieval.multi_query import MultiQueryExpander
                mq_config = self.config.get("multi_query", {})
                self._multi_query_expander = MultiQueryExpander(
                    n_variations=self.multi_query_variations,
                    use_llm=mq_config.get("use_llm", True)
                )
                logger.info(f"Multi-Query RAG inicializado ({self.multi_query_variations} variaciones)")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Multi-Query: {e}")
                self._multi_query_expander = None

        # Self-RAG
        if self.use_self_rag:
            try:
                from ..agents.self_rag import SelfRAGPipeline
                self_rag_config = self.config.get("self_rag", {})
                self._self_rag_pipeline = SelfRAGPipeline(
                    retriever=self._retriever,
                    synthesizer=self._synthesizer,
                    max_iterations=self_rag_config.get("max_iterations", 3),
                    min_relevance_ratio=self_rag_config.get("min_relevance_ratio", 0.5),
                    require_support=self_rag_config.get("require_support", True)
                )
                logger.info("Self-RAG inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Self-RAG: {e}")
                self._self_rag_pipeline = None

        # ColBERT Reranker
        if self.use_colbert_rerank:
            try:
                from ..retrieval.colbert_retriever import ColBERTReranker
                colbert_config = self.config.get("colbert", {})
                self._colbert_reranker = ColBERTReranker(
                    model_name=colbert_config.get("model_name", "colbert-ir/colbertv2.0")
                )
                if self._colbert_reranker._available:
                    logger.info("ColBERT Reranker inicializado")
                else:
                    logger.warning("ColBERT no disponible (instala: pip install ragatouille)")
                    self._colbert_reranker = None
            except Exception as e:
                logger.warning(f"No se pudo inicializar ColBERT: {e}")
                self._colbert_reranker = None

        # Citation Grounding
        if self.use_grounded_citations:
            try:
                from ..generation.citation_grounding import CitationGroundingSystem
                grounding_config = self.config.get("citation_grounding", {})
                self._grounding_system = CitationGroundingSystem(
                    min_grounding_score=self.min_grounding_score,
                    require_all_cited=grounding_config.get("require_all_cited", True),
                    use_llm_verification=grounding_config.get("use_llm_verification", True)
                )
                logger.info(f"Citation Grounding inicializado (min: {self.min_grounding_score})")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Citation Grounding: {e}")
                self._grounding_system = None

        # Hierarchical Index Routing
        if self.use_hierarchical_routing:
            try:
                from ..retrieval.hierarchical_index import HierarchicalSummaryIndex
                self._hierarchical_index = HierarchicalSummaryIndex(self.indices_dir)
                stats = self._hierarchical_index.get_stats()
                if stats.get("total_summaries", 0) > 0:
                    logger.info(f"Hierarchical Index cargado: {stats['total_summaries']} resúmenes")
                else:
                    logger.warning("Hierarchical Index vacío (ejecuta indexación con --hierarchical)")
                    self._hierarchical_index = None
            except Exception as e:
                logger.warning(f"No se pudo inicializar Hierarchical Index: {e}")
                self._hierarchical_index = None

    def ask(
        self,
        query: str,
        top_k: int = 10,
        stream: bool = False,
        stream_callback=None,
        sources_only: bool = False,
        abstention_threshold: float = 0.002,
        filters: dict = None
    ):
        """
        Procesa una consulta.
        
        Args:
            query: Pregunta del usuario
            top_k: Número de fuentes a recuperar
            stream: Si usar streaming
            stream_callback: Callback para streaming
            sources_only: Solo devolver fuentes sin generar respuesta
            abstention_threshold: Umbral mínimo de score RRF para responder (default: 0.002)
                                 Nota: scores RRF típicos están en rango 0.001-0.02
            filters: Filtros de metadatos (ej: {"category": "computacion_cuantica"})
        
        Returns:
            Tuple (response, sources, routing_decision)
        """
        self._init_components()
        
        # 0. Verificar caché semántico
        if self._semantic_cache and not sources_only and not stream:
            try:
                cached, similarity = self._semantic_cache.get(query)
                if cached:
                    logger.info(f"💾 Cache semántico HIT (sim={similarity:.3f})")
                    from ..generation.synthesizer import GeneratedResponse
                    
                    # Reconstruir sources desde caché
                    from ..retrieval.fusion import RetrievalResult
                    cached_sources = [
                        RetrievalResult(
                            chunk_id=s.get("chunk_id", ""),
                            content=s.get("content", ""),
                            score=s.get("score", 0.0),
                            doc_id=s.get("doc_id", ""),
                            doc_title=s.get("doc_title", ""),
                            header_path=s.get("header_path", ""),
                            retriever_type=s.get("retriever_type", "cached")
                        )
                        for s in cached.sources
                    ]
                    
                    # Reconstruir routing desde caché
                    from ..agents.router import RoutingDecision, RetrievalStrategy
                    cached_routing = RoutingDecision(
                        query_type=cached.routing_info.get("query_type", "unknown"),
                        strategy=RetrievalStrategy.HYBRID,
                        reasoning=f"[CACHED] {cached.routing_info.get('reasoning', '')}",
                        vector_weight=cached.routing_info.get("vector_weight", 0.5),
                        bm25_weight=cached.routing_info.get("bm25_weight", 0.3),
                        graph_weight=cached.routing_info.get("graph_weight", 0.2)
                    )
                    
                    return GeneratedResponse(
                        content=cached.response,
                        query=query,
                        query_type=cached.routing_info.get("query_type", "cached"),
                        sources_used=[s.get("chunk_id", "") for s in cached.sources],
                        model=f"{cached.model} (cached)",
                        tokens_input=0,  # No se consumen tokens
                        tokens_output=0,
                        latency_ms=0,
                        metadata={"semantic_cache_hit": True, "similarity": similarity}
                    ), cached_sources, cached_routing
            except Exception as e:
                logger.debug(f"Error consultando caché semántico: {e}")
        
        # 1. Routing (si habilitado)
        routing = None
        if self._router:
            routing = self._router.route(query)
            logger.info(f"Routing: {routing.strategy.value}")

        # === Self-RAG: delegar a pipeline auto-reflexivo ===
        if self.use_self_rag and self._self_rag_pipeline:
            response, self_rag_state = self._self_rag_pipeline.ask(
                query=query,
                top_k=top_k,
                stream=stream,
                stream_callback=stream_callback
            )
            # Extraer sources del estado
            sources = self_rag_state.retrieved_contexts
            response.metadata["self_rag"] = self_rag_state.to_dict()
            return response, sources, routing

        # === Multi-Query: expandir query en variaciones ===
        expanded_query = None
        if self._multi_query_expander:
            try:
                expanded_query = self._multi_query_expander.expand(query)
                logger.info(f"Multi-Query: {len(expanded_query.all_queries())} queries")
            except Exception as e:
                logger.warning(f"Error en Multi-Query expansion: {e}")

        # === Hierarchical Routing: filtrar por resúmenes ===
        hierarchical_routing = None
        if self._hierarchical_index:
            try:
                hierarchical_routing = self._hierarchical_index.route_query(query)
                logger.info(f"Hierarchical: {len(hierarchical_routing.candidate_chunk_ids)} chunks candidatos")
            except Exception as e:
                logger.warning(f"Error en Hierarchical routing: {e}")

        # 2. Retrieval con pesos dinámicos del router
        search_kwargs = {"top_k": top_k}
        if routing:
            search_kwargs["vector_top_k"] = int(top_k * (1 + routing.vector_weight))
            search_kwargs["bm25_top_k"] = int(top_k * (1 + routing.bm25_weight))
            # Pasar pesos dinámicos para la fusión RRF
            search_kwargs["dynamic_weights"] = {
                "vector": routing.vector_weight,
                "bm25": routing.bm25_weight,
                "graph": routing.graph_weight
            }

        if filters:
            search_kwargs["filters"] = filters

        # Si agentic mode, delegar al pipeline agéntico
        if self.use_agentic and self._agentic_pipeline:
            response, agentic_sources, agent_state = self._agentic_pipeline.ask(
                query=query,
                top_k=top_k,
                stream=stream,
                stream_callback=stream_callback
            )
            response.metadata["agentic"] = {
                "iterations": agent_state.iteration,
                "confidence": agent_state.confidence,
                "strategies_used": agent_state.search_strategies_used,
                "reasoning_trace": agent_state.reasoning_trace
            }
            return response, agentic_sources, routing

        # === Retrieval (con Multi-Query si habilitado) ===
        if expanded_query and len(expanded_query.all_queries()) > 1:
            # Multi-Query: buscar con todas las variaciones y fusionar
            all_sources = []
            seen_chunk_ids = set()
            for q in expanded_query.all_queries():
                q_sources = self._retriever.search(q, **search_kwargs)
                for s in q_sources:
                    if s.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(s.chunk_id)
                        all_sources.append(s)
            # Ordenar por score y limitar
            all_sources.sort(key=lambda x: -x.score)
            sources = all_sources[:top_k * 2]
            logger.info(f"Multi-Query retrieval: {len(sources)} fuentes únicas")
        else:
            sources = self._retriever.search(query, **search_kwargs)

        # === Hierarchical filtering: priorizar chunks de secciones relevantes ===
        if hierarchical_routing and hierarchical_routing.candidate_chunk_ids:
            prioritized = []
            other = []
            for s in sources:
                if s.chunk_id in hierarchical_routing.candidate_chunk_ids:
                    s.score *= 1.15  # Boost por estar en routing
                    prioritized.append(s)
                else:
                    other.append(s)
            sources = prioritized + other
            sources.sort(key=lambda x: -x.score)
            sources = sources[:top_k]

        # === ColBERT Reranking ===
        if self._colbert_reranker and sources:
            try:
                sources = self._colbert_reranker.rerank(query, sources, top_k=top_k)
                logger.info("ColBERT reranking aplicado")
            except Exception as e:
                logger.warning(f"Error en ColBERT reranking: {e}")

        # 2.3 Corrective RAG (si habilitado)
        if self._corrective_rag and sources:
            try:
                crag_result = self._corrective_rag.correct(query, sources, top_k=top_k)
                sources = crag_result.corrected_results
                logger.info(
                    f"CRAG: action={crag_result.action_taken}, "
                    f"correct={crag_result.stats.get('correct', 0)}, "
                    f"incorrect={crag_result.stats.get('incorrect', 0)}"
                )
            except Exception as e:
                logger.warning(f"Error en CRAG, usando resultados originales: {e}")

        if not sources:
            # Sin fuentes, responder que no hay información
            from ..generation.synthesizer import GeneratedResponse
            return GeneratedResponse(
                content="No encontré información relevante en la biblioteca para responder esta consulta.",
                query=query,
                query_type="none",
                sources_used=[],
                model="n/a",
                tokens_input=0,
                tokens_output=0,
                latency_ms=0,
                abstained=True
            ), [], routing
        
        # 2.5 Política de abstención: verificar score mínimo (escala RRF)
        # Nota: Scores RRF típicos: 0.001-0.02, un buen match tiene ~0.005+
        max_score = max(s.score for s in sources) if sources else 0
        
        if max_score < abstention_threshold and not sources_only:
            from ..generation.synthesizer import GeneratedResponse
            # Construir mensaje de abstención con fuentes cercanas
            closest_sources = sorted(sources, key=lambda x: -x.score)[:3]
            sources_list = "\n".join([
                f"  - {s.doc_title} > {s.header_path} (score: {s.score:.4f})"
                for s in closest_sources
            ])
            
            abstention_msg = (
                f"⚠️ **No encontré información suficientemente relevante** para responder con certeza.\n\n"
                f"El score máximo RRF ({max_score:.4f}) está por debajo del umbral ({abstention_threshold:.4f}).\n\n"
                f"**Documentos más cercanos (pero no suficientemente relevantes):**\n{sources_list}\n\n"
                f"💡 Sugerencias:\n"
                f"- Reformula la pregunta con términos más específicos\n"
                f"- Verifica que el tema esté cubierto en tu biblioteca\n"
                f"- Usa `--sources` para explorar qué documentos hay disponibles"
            )
            
            return GeneratedResponse(
                content=abstention_msg,
                query=query,
                query_type="abstention",
                sources_used=[],
                model="n/a",
                tokens_input=0,
                tokens_output=0,
                latency_ms=0,
                abstained=True,
                metadata={"max_score": max_score, "threshold": abstention_threshold}
            ), sources, routing
        
        # 2.6 Modo solo fuentes (sin generación)
        if sources_only:
            from ..generation.synthesizer import GeneratedResponse
            return GeneratedResponse(
                content="[Modo --sources: solo se muestran fuentes, sin generar respuesta]",
                query=query,
                query_type="sources_only",
                sources_used=[s.chunk_id for s in sources],
                model="none",
                tokens_input=0,
                tokens_output=0,
                latency_ms=0
            ), sources, routing
        
        # 2.7 Compresión de contexto (si habilitada)
        compressed_sources = sources
        compression_stats = None
        if self._context_compressor and sources:
            try:
                contexts = [s.content for s in sources]
                compressed_contexts, compression_stats = self._context_compressor.compress_contexts(
                    contexts,
                    max_total_tokens=self._context_budget
                )
                
                if compression_stats.get("compression_applied"):
                    # Actualizar sources con contenido comprimido
                    from ..retrieval.fusion import RetrievalResult
                    compressed_sources = [
                        RetrievalResult(
                            chunk_id=s.chunk_id,
                            content=compressed_contexts[i],
                            score=s.score,
                            doc_id=s.doc_id,
                            doc_title=s.doc_title,
                            header_path=s.header_path,
                            retriever_type=s.retriever_type
                        )
                        for i, s in enumerate(sources) if i < len(compressed_contexts)
                    ]
                    logger.info(
                        f"📦 Contexto comprimido: {compression_stats['original_tokens']} → "
                        f"{compression_stats['compressed_tokens']} tokens "
                        f"({compression_stats['compression_ratio']:.1%})"
                    )
            except Exception as e:
                logger.warning(f"Error comprimiendo contexto: {e}")
                compressed_sources = sources
        
        # 3. Generation (usando sources comprimidas si aplica)
        # === Citation Grounding: genera con citas verificables ===
        if self._grounding_system:
            try:
                # Generar con citas obligatorias
                cited_content = self._grounding_system.generate_with_citations(
                    query, compressed_sources
                )
                # Verificar y mejorar grounding
                final_content, grounding_result = self._grounding_system.enforce_grounding(
                    cited_content, compressed_sources
                )

                from ..generation.synthesizer import GeneratedResponse
                response = GeneratedResponse(
                    content=final_content,
                    query=query,
                    query_type="grounded",
                    sources_used=[s.chunk_id for s in compressed_sources],
                    model="grounded_synthesis",
                    tokens_input=0,
                    tokens_output=0,
                    latency_ms=0,
                    metadata={
                        "grounding": grounding_result.to_dict(),
                        "grounding_score": grounding_result.grounding_score
                    }
                )
                logger.info(f"Citation Grounding: score={grounding_result.grounding_score:.2f}")
            except Exception as e:
                logger.warning(f"Error en Citation Grounding, usando generación normal: {e}")
                response = self._synthesizer.generate(
                    query=query,
                    results=compressed_sources,
                    stream=stream,
                    stream_callback=stream_callback
                )
        else:
            response = self._synthesizer.generate(
                query=query,
                results=compressed_sources,
                stream=stream,
                stream_callback=stream_callback
            )

        # Añadir metadata de mejoras avanzadas
        if expanded_query:
            response.metadata["multi_query"] = expanded_query.to_dict()
        if hierarchical_routing:
            response.metadata["hierarchical_routing"] = hierarchical_routing.to_dict()

        # Añadir stats de compresión a metadata
        if compression_stats:
            response.metadata["compression"] = compression_stats
        
        # 4. Crítica (si habilitada)
        if self._critic:
            critique = self._critic.critique(response, sources, query)
            response.metadata["critique"] = critique.to_dict()
        
        # 5. Guardar en caché semántico si está habilitado
        if self._semantic_cache and not response.abstained and not stream:
            try:
                # Preparar datos para caché
                sources_data = [
                    {
                        "chunk_id": s.chunk_id,
                        "content": s.content,
                        "score": s.score,
                        "doc_id": s.doc_id,
                        "doc_title": s.doc_title,
                        "header_path": s.header_path,
                        "retriever_type": s.retriever_type
                    }
                    for s in sources
                ]
                routing_data = {
                    "query_type": routing.query_type if routing else "unknown",
                    "reasoning": routing.reasoning if routing else "",
                    "vector_weight": routing.vector_weight if routing else 0.5,
                    "bm25_weight": routing.bm25_weight if routing else 0.3,
                    "graph_weight": routing.graph_weight if routing else 0.2
                }
                
                self._semantic_cache.put(
                    query=query,
                    response=response.content,
                    sources=sources_data,
                    routing_info=routing_data,
                    model=response.model,
                    generation_time=response.latency_ms
                )
                logger.info(f"💾 Respuesta guardada en caché semántico")
            except Exception as e:
                logger.debug(f"Error guardando en caché semántico: {e}")
        
        return response, sources, routing
    
    def ask_deep(
        self,
        query: str,
        top_k: int = 10,
        max_iterations: int = 3,
        min_score_threshold: float = 0.6
    ):
        """
        Modo Deep Research: descompone queries complejas y busca iterativamente.
        
        Args:
            query: Pregunta original
            top_k: Número de fuentes por sub-query
            max_iterations: Máximo de iteraciones de búsqueda
            min_score_threshold: Umbral mínimo de score para considerar suficiente
        
        Returns:
            Tuple (response, all_sources, routing_decision)
        """
        self._init_components()
        
        # 1. Analizar complejidad y descomponer si necesario
        routing = self._router.route(query) if self._router else None
        
        # Usar LLM para descomponer query compleja
        sub_queries = self._decompose_query(query, routing)
        
        logger.info(f"Deep Research: {len(sub_queries)} sub-queries")
        print(f"\n📋 Descomponiendo en {len(sub_queries)} sub-preguntas:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"   {i}. {sq}")
        
        # 2. Buscar para cada sub-query
        all_sources = []
        all_source_ids = set()
        
        for iteration in range(max_iterations):
            iteration_sources = []
            
            for sq in sub_queries:
                sources = self._retriever.search(sq, top_k=top_k)
                for s in sources:
                    if s.chunk_id not in all_source_ids:
                        all_source_ids.add(s.chunk_id)
                        iteration_sources.append(s)
            
            all_sources.extend(iteration_sources)
            
            # Evaluar si tenemos suficiente información
            if iteration_sources:
                max_score = max(s.score for s in iteration_sources)
                print(f"\n🔄 Iteración {iteration + 1}: {len(iteration_sources)} nuevas fuentes (max score: {max_score:.2f})")
                
                if max_score >= min_score_threshold and len(all_sources) >= top_k:
                    break
            else:
                break
        
        # 3. Ordenar y limitar fuentes
        all_sources.sort(key=lambda x: -x.score)
        final_sources = all_sources[:top_k * 2]  # Más contexto para deep research
        
        if not final_sources:
            from ..generation.synthesizer import GeneratedResponse
            return GeneratedResponse(
                content="No encontré información relevante tras búsqueda profunda.",
                query=query,
                query_type="deep_research_failed",
                sources_used=[],
                model="n/a",
                tokens_input=0,
                tokens_output=0,
                latency_ms=0,
                abstained=True
            ), [], routing
        
        print(f"\n✅ Total: {len(final_sources)} fuentes relevantes encontradas")
        
        # 4. Generar respuesta con contexto expandido
        response = self._synthesizer.generate(
            query=query,
            results=final_sources,
            stream=False
        )
        
        response.metadata["deep_research"] = {
            "sub_queries": sub_queries,
            "iterations": iteration + 1,
            "total_sources_found": len(all_sources)
        }
        
        # 5. Crítica
        if self._critic:
            critique = self._critic.critique(response, final_sources, query)
            response.metadata["critique"] = critique.to_dict()
        
        return response, final_sources, routing
    
    def _decompose_query(self, query: str, routing=None) -> List[str]:
        """
        Descompone una query compleja en sub-queries.
        
        Usa heurísticas simples primero, LLM si es necesario.
        """
        sub_queries = [query]  # Siempre incluir original
        
        # Detectar patrones de comparación
        comparison_patterns = [
            (r"(?:compara|diferencia|versus|vs\.?)\s+(.+?)\s+(?:y|con|and|with)\s+(.+)", 
             lambda m: [f"¿Qué es {m.group(1).strip()}?", f"¿Qué es {m.group(2).strip()}?", f"Diferencias entre {m.group(1).strip()} y {m.group(2).strip()}"]),
            (r"(?:relación|relationship)\s+(?:entre|between)\s+(.+?)\s+(?:y|and)\s+(.+)",
             lambda m: [f"¿Qué es {m.group(1).strip()}?", f"¿Qué es {m.group(2).strip()}?", f"¿Cómo se relacionan {m.group(1).strip()} y {m.group(2).strip()}?"]),
        ]
        
        import re
        for pattern, extractor in comparison_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                sub_queries = extractor(match)
                break
        
        # Detectar múltiples conceptos (con "y", "además", etc.)
        if len(sub_queries) == 1:
            multi_patterns = [
                r"(.+?)\s+(?:y además|y también|and also)\s+(.+)",
                r"(.+?)\s+(?:primero|first).+?(?:luego|then)\s+(.+)"
            ]
            for pattern in multi_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    sub_queries = [match.group(1).strip(), match.group(2).strip(), query]
                    break
        
        # Si el routing indica multi-hop o compleja, intentar descomponer con LLM
        if routing and routing.strategy.value in ["multi_hop", "hybrid_with_graph"]:
            if len(sub_queries) == 1:
                # Intentar descomposición con LLM ligero
                llm_decomposition = self._decompose_with_llm(query)
                if llm_decomposition:
                    sub_queries = llm_decomposition
        
        return sub_queries
    
    def _decompose_with_llm(self, query: str) -> List[str]:
        """Usa LLM para descomponer query (solo si es realmente necesario)."""
        try:
            from src.llm_provider import complete as llm_complete
            import json

            system_prompt = """Eres un asistente que descompone preguntas complejas en sub-preguntas más simples.
Devuelve SOLO una lista JSON de strings con las sub-preguntas.
Si la pregunta ya es simple, devuelve ["pregunta original"].
Máximo 4 sub-preguntas."""

            response = llm_complete(
                prompt=f"Descompón esta pregunta: {query}",
                system=system_prompt,
                temperature=0,
                max_tokens=200,
            )

            content = response.content.strip()
            # Extraer JSON array de la respuesta (robusto ante brackets en texto)
            sub_queries = self._extract_json_array(content)
            if sub_queries is not None:
                return sub_queries
        except Exception as e:
            logger.warning(f"Error en descomposición LLM: {e}")

        return None

    @staticmethod
    def _extract_json_array(text: str) -> Optional[List[str]]:
        """Extrae un JSON array de strings de un texto, robusto ante brackets sueltos."""
        import json
        # Intentar parsear desde cada '[' encontrado
        start = 0
        while True:
            idx = text.find("[", start)
            if idx == -1:
                break
            # Buscar el ']' correspondiente contando profundidad
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
            start = idx + 1
        return None


def interactive_mode(pipeline: RAGPipeline, save_sessions: bool, output_path: Path):
    """Modo interactivo con memoria conversacional."""
    from ..agents.session_manager import get_session_manager
    
    # Inicializar session manager
    session_manager = get_session_manager(output_path / "sessions")
    session_id = session_manager.create_session()
    
    print("\n🎯 Modo interactivo con memoria conversacional")
    print(f"   Sesión: {session_id[:8]}...")
    print("   Comandos especiales:")
    print("   - /sources    Ver fuentes de la última respuesta")
    print("   - /export     Exportar última respuesta a Markdown")
    print("   - /history    Ver historial de conversación")
    print("   - /clear      Limpiar pantalla")
    print("   - /new        Nueva sesión (borrar memoria)")
    print("")
    print("   💡 Soporta preguntas de seguimiento:")
    print('      "Más detalles", "Expande el punto 2", "¿Y si...?"')
    print("")
    
    last_response = None
    last_sources = None
    
    while True:
        try:
            query = input("\n❓ Tu pregunta: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['salir', 'exit', 'quit', 'q']:
                print("\n👋 ¡Hasta pronto!")
                break
            
            # Comandos especiales
            if query.startswith('/'):
                cmd = query[1:].lower()
                
                if cmd == 'sources' and last_sources:
                    print("\n📚 Fuentes de la última respuesta:")
                    for i, s in enumerate(last_sources, 1):
                        print(f"\n[{i}] {s.doc_title}")
                        print(f"    {s.header_path}")
                        print(f"    Score: {s.score:.3f}")
                        print(f"    Preview: {s.content[:200]}...")
                    continue
                
                elif cmd == 'export' and last_response:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_path = output_path / f"response_{timestamp}.md"
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(export_path, 'w') as f:
                        f.write(f"# Consulta\n\n{last_response.query}\n\n")
                        f.write(f"# Respuesta\n\n{last_response.content}\n\n")
                        f.write(f"# Fuentes\n\n")
                        for i, s in enumerate(last_sources, 1):
                            f.write(f"- [{i}] {s.doc_title} - {s.header_path}\n")
                    
                    print(f"✅ Exportado a {export_path}")
                    continue
                
                elif cmd == 'history':
                    context = session_manager.get_session(session_id)
                    if context and context.messages:
                        print("\n📜 Historial de conversación:")
                        print("─" * 50)
                        for i, msg in enumerate(context.messages, 1):
                            role = "👤 Usuario" if msg.role == "user" else "🤖 Asistente"
                            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                            print(f"\n[{i}] {role}:")
                            print(f"    {content}")
                        print("\n" + "─" * 50)
                    else:
                        print("📭 Sin historial aún")
                    continue
                
                elif cmd == 'new':
                    session_id = session_manager.create_session()
                    print(f"🆕 Nueva sesión: {session_id[:8]}...")
                    last_response = None
                    last_sources = None
                    continue
                
                elif cmd == 'clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_banner()
                    continue
                
                else:
                    print("❌ Comando no reconocido")
                    continue
            
            # Detectar si es pregunta de seguimiento
            is_followup, followup_type = session_manager.is_followup_query(query, session_id)
            
            if is_followup:
                expanded_query = session_manager.expand_query_with_context(
                    query, session_id, followup_type
                )
                if expanded_query != query:
                    print(f"📝 Interpretado como: {expanded_query}")
                query_to_search = expanded_query
            else:
                query_to_search = query
            
            # Añadir mensaje del usuario a la sesión
            session_manager.add_message(session_id, "user", query)
            
            # Procesar consulta
            print("\n🔍 Buscando en la biblioteca...")
            
            response, sources, routing = pipeline.ask(query_to_search)
            
            print(format_response(response, sources))
            
            # Añadir respuesta a la sesión
            source_ids = [s.chunk_id for s in sources]
            session_manager.add_message(
                session_id, 
                "assistant", 
                response.content,
                sources=source_ids,
                metadata={"model": response.model}
            )
            
            last_response = response
            last_sources = sources
            
            # Guardar sesión (ya se guarda automáticamente)
            if save_sessions:
                save_session(query, response, sources, output_path)
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta pronto!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error procesando consulta: {e}")


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
