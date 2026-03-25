"""
formatters - Funciones de formateo y salida para el CLI.

Incluye print_banner(), format_response() y save_session().
Extraídas de ask_library.py para modularidad.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def print_banner():
    """Muestra banner del CLI."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║        📚 LibrarAI - Ask Library                           ║
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
