"""
rag_pipeline - Clase RAGPipeline para el pipeline RAG completo.

Extraída de ask_library.py para modularidad.
"""

import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


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
