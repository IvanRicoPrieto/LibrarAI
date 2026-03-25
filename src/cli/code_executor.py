"""
code_executor - Ejecución de bloques de código Python en respuestas.

Extraída de ask_library.py para modularidad.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
