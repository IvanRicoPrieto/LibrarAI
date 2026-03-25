"""
interactive - Modo interactivo REPL con memoria conversacional.

Extraída de ask_library.py para modularidad.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def interactive_mode(pipeline: RAGPipeline, save_sessions: bool, output_path: Path):
    """Modo interactivo con memoria conversacional."""
    from ..agents.session_manager import get_session_manager
    from .formatters import print_banner, format_response, save_session

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
