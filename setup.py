#!/usr/bin/env python3
"""
Setup script - Configura el entorno de LibrarAI.

Uso:
    python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Configura el entorno."""
    print("🔬 LibrarAI - Setup")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    # 1. Verificar Python
    print(f"\n✅ Python {sys.version.split()[0]}")
    
    # 2. Crear estructura de directorios
    print("\n📁 Creando directorios...")
    dirs = [
        "data/markdown/books",
        "data/markdown/papers", 
        "indices",
        "logs/sessions",
        "outputs/figures"
    ]
    for d in dirs:
        path = project_root / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {d}")
    
    # 3. Verificar .env
    env_path = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_path.exists():
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_path)
            print(f"\n📝 Copiado .env.example → .env")
            print("   ⚠️  Edita .env con tus API keys")
        else:
            print("\n⚠️  No se encontró .env.example")
    else:
        print("\n✅ .env ya existe")
    
    # 4. Verificar dependencias
    print("\n📦 Verificando dependencias...")
    
    required = [
        "openai",
        "anthropic", 
        "qdrant-client",
        "rank-bm25",
        "tiktoken",
        "networkx",
        "pyyaml",
        "python-dotenv",
        "tqdm"
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"   ✓ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"   ✗ {pkg}")
    
    if missing:
        print(f"\n⚠️  Paquetes faltantes: {', '.join(missing)}")
        print("   Instala con: pip install -r requirements.txt")
    else:
        print("\n✅ Todas las dependencias instaladas")
    
    # 5. Verificar API keys
    print("\n🔑 Verificando API keys...")
    
    # Cargar .env si existe
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key and openai_key.startswith("sk-"):
        print("   ✓ OPENAI_API_KEY configurada")
    else:
        print("   ✗ OPENAI_API_KEY no configurada")
    
    if anthropic_key and anthropic_key.startswith("sk-ant-"):
        print("   ✓ ANTHROPIC_API_KEY configurada")
    else:
        print("   ✗ ANTHROPIC_API_KEY no configurada")
    
    # 6. Resumen
    print("\n" + "=" * 50)
    print("📋 Próximos pasos:")
    print("")
    print("1. Configura tus API keys en .env")
    print("2. Coloca tus documentos Markdown en:")
    print("   - data/markdown/books/")
    print("   - data/markdown/papers/")
    print("3. Indexa la biblioteca:")
    print("   python -m src.cli.ingest_library")
    print("4. Consulta:")
    print("   python -m src.cli.ask_library --interactive")
    print("")


if __name__ == "__main__":
    main()
