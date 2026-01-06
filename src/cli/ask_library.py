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
║        Tu biblioteca inteligente con IA                    ║
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
        hyde_domain: str = "quantum_computing"
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
        
        # Componentes (lazy init)
        self._retriever = None
        self._synthesizer = None
        self._router = None
        self._critic = None
        self._citation_injector = None
    
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
            model=self.model,
            temperature=gen_config.get("temperature", 0.3),
            max_output_tokens=gen_config.get("max_tokens", 2000)
        )
        
        self._citation_injector = CitationInjector()
        
        if self.use_router:
            self._router = QueryRouter(use_llm_router=False)
        
        if self.use_critic:
            self._critic = ResponseCritic(use_llm_critic=False)
    
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
        
        # 1. Routing (si habilitado)
        routing = None
        if self._router:
            routing = self._router.route(query)
            logger.info(f"Routing: {routing.strategy.value}")
        
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
        
        sources = self._retriever.search(query, **search_kwargs)
        
        if not sources:
            # Sin fuentes, responder que no hay información
            from ..generation.synthesizer import GeneratedResponse
            return GeneratedResponse(
                content="No encontré información relevante en la biblioteca para responder esta consulta.",
                query=query,
                query_type="none",
                sources_used=[],
                model=self.model,
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
                model=self.model,
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
        
        # 3. Generation
        response = self._synthesizer.generate(
            query=query,
            results=sources,
            stream=stream,
            stream_callback=stream_callback
        )
        
        # 4. Crítica (si habilitada)
        if self._critic:
            critique = self._critic.critique(response, sources, query)
            response.metadata["critique"] = critique.to_dict()
        
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
                model=self.model,
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
            import openai
            client = openai.OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": """Eres un asistente que descompone preguntas complejas en sub-preguntas más simples.
Devuelve SOLO una lista JSON de strings con las sub-preguntas.
Si la pregunta ya es simple, devuelve ["pregunta original"].
Máximo 4 sub-preguntas."""},
                    {"role": "user", "content": f"Descompón esta pregunta: {query}"}
                ],
                temperature=0,
                max_tokens=200
            )
            
            import json
            content = response.choices[0].message.content.strip()
            # Extraer JSON de la respuesta
            if "[" in content:
                json_str = content[content.find("["):content.rfind("]")+1]
                sub_queries = json.loads(json_str)
                if isinstance(sub_queries, list) and all(isinstance(s, str) for s in sub_queries):
                    return sub_queries
        except Exception as e:
            logger.warning(f"Error en descomposición LLM: {e}")
        
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
        '--cache-stats',
        action='store_true',
        help='Mostrar estadísticas del cache de embeddings'
    )
    
    parser.add_argument(
        '--costs', '-c',
        action='store_true',
        help='Mostrar resumen de costes acumulados de consultas'
    )
    
    args = parser.parse_args()
    
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
    
    # Si pide listar categorías, mostrar y salir
    if args.list_categories:
        try:
            paths = setup_paths()
            from qdrant_client import QdrantClient
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
    
    # Cargar variables de entorno
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass
    
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
            hyde_domain=args.hyde_domain
        )
    except Exception as e:
        logger.error(f"Error inicializando pipeline: {e}")
        sys.exit(1)
    
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
        
        if args.json:
            output = {
                "query": args.query,
                "response": response.to_dict(),
                "sources": [s.to_dict() for s in sources],
                "routing": routing.to_dict() if routing else None
            }
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
