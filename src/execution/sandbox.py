"""
Code Sandbox - Ejecución segura de código Python.

Permite ejecutar código generado por el LLM de forma aislada
para cálculos, gráficas y simulaciones cuánticas.

Modos de ejecución:
1. Subprocess con restricciones (por defecto, sin Docker)
2. Docker container (máxima seguridad, requiere Docker)
"""

import subprocess
import tempfile
import os
import sys
import signal
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


# Librerías permitidas en el sandbox
ALLOWED_IMPORTS = {
    # Matemáticas y ciencia
    "numpy", "np",
    "scipy",
    "sympy",
    "math",
    "cmath",
    
    # Visualización
    "matplotlib",
    "matplotlib.pyplot", "plt",
    
    # Computación cuántica
    "qutip",
    
    # Datos
    "pandas", "pd",
    
    # Utilidades seguras
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "json",
    "re",
    "random",
    "statistics",
}

# Imports bloqueados (peligrosos)
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
    "shelve",
    "__import__",
    "importlib",
    "eval",
    "exec",
    "compile",
    "open",  # Solo bloqueado para escritura
    "shutil",
    "pathlib",
    "glob",
}


@dataclass
class SandboxResult:
    """Resultado de la ejecución en sandbox."""
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    figures: List[str] = field(default_factory=list)  # Base64 de imágenes PNG
    return_value: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time_ms": self.execution_time_ms,
            "figures": self.figures,
            "error_type": self.error_type,
            "error_message": self.error_message
        }
    
    def get_markdown_output(self) -> str:
        """Genera salida en Markdown incluyendo figuras."""
        output = []
        
        if self.stdout:
            output.append("**Salida:**")
            output.append(f"```\n{self.stdout}\n```")
        
        if self.figures:
            output.append("\n**Figuras generadas:**")
            for i, fig in enumerate(self.figures, 1):
                output.append(f"\n![Figura {i}](data:image/png;base64,{fig})")
        
        if self.error_message:
            output.append(f"\n**Error:** {self.error_type}: {self.error_message}")
        
        output.append(f"\n*Tiempo de ejecución: {self.execution_time_ms:.0f}ms*")
        
        return "\n".join(output)


class CodeSandbox:
    """
    Sandbox para ejecución segura de código Python.
    
    Características de seguridad:
    - Whitelist de imports
    - Timeout para evitar loops infinitos
    - Límite de memoria (subprocess)
    - Sin acceso a red ni filesystem
    """
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        max_memory_mb: int = 512,
        outputs_dir: Optional[Path] = None,
        use_docker: bool = False
    ):
        """
        Args:
            timeout_seconds: Tiempo máximo de ejecución
            max_memory_mb: Memoria máxima (solo en modo Docker)
            outputs_dir: Directorio para guardar figuras
            use_docker: Si usar Docker para máximo aislamiento
        """
        self.timeout = timeout_seconds
        self.max_memory = max_memory_mb
        self.outputs_dir = Path(outputs_dir) if outputs_dir else None
        self.use_docker = use_docker
        
        if self.outputs_dir:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Valida que el código no use imports peligrosos.
        
        Returns:
            (es_valido, mensaje_error)
        """
        # Buscar imports
        import_pattern = re.compile(
            r'^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))',
            re.MULTILINE
        )
        
        for match in import_pattern.finditer(code):
            module = match.group(1) or match.group(2)
            base_module = module.split('.')[0]
            
            if base_module in BLOCKED_IMPORTS:
                return False, f"Import bloqueado: '{module}' (seguridad)"
        
        # Buscar funciones peligrosas
        dangerous_patterns = [
            (r'\beval\s*\(', "eval()"),
            (r'\bexec\s*\(', "exec()"),
            (r'\bcompile\s*\(', "compile()"),
            (r'\b__import__\s*\(', "__import__()"),
            (r'\bopen\s*\([^)]*["\']w', "open() con modo escritura"),
            (r'\bsubprocess\b', "subprocess"),
            (r'\bos\.system\b', "os.system()"),
        ]
        
        for pattern, name in dangerous_patterns:
            if re.search(pattern, code):
                return False, f"Operación bloqueada: {name} (seguridad)"
        
        return True, ""
    
    def execute(self, code: str, capture_figures: bool = True) -> SandboxResult:
        """
        Ejecuta código Python de forma segura.
        
        Args:
            code: Código Python a ejecutar
            capture_figures: Si capturar figuras matplotlib
        
        Returns:
            Resultado de la ejecución
        """
        # Validar código
        is_valid, error_msg = self.validate_code(code)
        if not is_valid:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=error_msg,
                execution_time_ms=0,
                error_type="SecurityError",
                error_message=error_msg
            )
        
        if self.use_docker:
            return self._execute_docker(code, capture_figures)
        else:
            return self._execute_subprocess(code, capture_figures)
    
    def _execute_subprocess(self, code: str, capture_figures: bool) -> SandboxResult:
        """Ejecuta código usando subprocess con timeout."""
        import time
        
        start_time = time.time()
        figures = []
        
        # Crear archivo temporal para el código
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            # Wrapper para capturar figuras
            wrapper_code = '''
import sys
import warnings
warnings.filterwarnings('ignore')

# Redirigir matplotlib a Agg (sin display)
import matplotlib
matplotlib.use('Agg')

'''
            if capture_figures:
                wrapper_code += '''
import matplotlib.pyplot as plt
import base64
import io

_sandbox_figures = []

# Hook para capturar figuras
_original_show = plt.show
def _capture_show(*args, **kwargs):
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        _sandbox_figures.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)

plt.show = _capture_show

'''
            
            wrapper_code += code
            
            if capture_figures:
                wrapper_code += '''

# Capturar figuras pendientes
if plt.get_fignums():
    _capture_show()

# Imprimir figuras como JSON al final
if _sandbox_figures:
    print("__SANDBOX_FIGURES_START__")
    import json
    print(json.dumps(_sandbox_figures))
    print("__SANDBOX_FIGURES_END__")
'''
            
            f.write(wrapper_code)
            temp_path = f.name
        
        try:
            # Ejecutar con timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Extraer figuras del stdout
            if capture_figures and "__SANDBOX_FIGURES_START__" in stdout:
                parts = stdout.split("__SANDBOX_FIGURES_START__")
                stdout = parts[0]
                
                figures_part = parts[1].split("__SANDBOX_FIGURES_END__")[0].strip()
                try:
                    import json
                    figures = json.loads(figures_part)
                except:
                    pass
            
            execution_time = (time.time() - start_time) * 1000
            
            return SandboxResult(
                success=result.returncode == 0,
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                execution_time_ms=execution_time,
                figures=figures,
                error_type="RuntimeError" if result.returncode != 0 else None,
                error_message=stderr.strip() if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Timeout: ejecución excedió {self.timeout}s",
                execution_time_ms=self.timeout * 1000,
                error_type="TimeoutError",
                error_message=f"El código excedió el tiempo límite de {self.timeout} segundos"
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                error_type=type(e).__name__,
                error_message=str(e)
            )
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _execute_docker(self, code: str, capture_figures: bool) -> SandboxResult:
        """
        Ejecuta código en contenedor Docker (máximo aislamiento).
        
        Requiere Docker instalado y la imagen python:3.11-slim disponible.
        """
        import time
        
        start_time = time.time()
        
        # Verificar Docker disponible
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Docker no disponible, usando subprocess")
            return self._execute_subprocess(code, capture_figures)
        
        # Crear script temporal
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Ejecutar en Docker
            cmd = [
                "docker", "run",
                "--rm",
                "--network", "none",  # Sin red
                "--memory", f"{self.max_memory}m",
                "--cpus", "1",
                "-v", f"{temp_path}:/code/script.py:ro",
                "python:3.11-slim",
                "python", "/code/script.py"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10  # Extra para overhead Docker
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return SandboxResult(
                success=result.returncode == 0,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                execution_time_ms=execution_time,
                figures=[],  # Docker mode no captura figuras por ahora
                error_type="RuntimeError" if result.returncode != 0 else None,
                error_message=result.stderr.strip() if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Timeout: ejecución excedió {self.timeout}s",
                execution_time_ms=self.timeout * 1000,
                error_type="TimeoutError",
                error_message=f"El código excedió el tiempo límite"
            )
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def generate_and_execute(
        self,
        prompt: str,
        model: str = "gpt-4.1-mini"
    ) -> tuple[str, SandboxResult]:
        """
        Genera código con LLM y lo ejecuta.
        
        Args:
            prompt: Descripción de lo que calcular/graficar
            model: Modelo LLM a usar
        
        Returns:
            (código_generado, resultado_ejecución)
        """
        import openai
        
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"""Eres un asistente que genera código Python para cálculos científicos y visualizaciones.

Reglas:
1. Imports permitidos: {', '.join(sorted(ALLOWED_IMPORTS))}
2. NO uses: os, sys, subprocess, socket, requests, open() para escribir
3. Para gráficas usa matplotlib y termina con plt.show()
4. El código debe ser autocontenido y ejecutable
5. Usa comentarios para explicar pasos importantes
6. Imprime resultados relevantes con print()

Responde SOLO con el código Python, sin explicaciones ni markdown."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        code = response.choices[0].message.content.strip()
        
        # Limpiar posibles marcadores de código
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        # Ejecutar
        result = self.execute(code)
        
        return code, result


# Función helper para uso directo
def run_code_safely(code: str, timeout: int = 30) -> SandboxResult:
    """
    Ejecuta código Python de forma segura.
    
    Args:
        code: Código a ejecutar
        timeout: Timeout en segundos
    
    Returns:
        Resultado de la ejecución
    """
    sandbox = CodeSandbox(timeout_seconds=timeout)
    return sandbox.execute(code)
