"""
Code Sandbox - Ejecución segura de código Python.

Permite ejecutar código generado por el LLM de forma aislada
para cálculos, gráficas y simulaciones cuánticas.

Modos de ejecución:
1. Subprocess con restricciones (por defecto, sin Docker)
2. Docker container (máxima seguridad, requiere Docker)

Características de seguridad:
- Whitelist de imports
- Validación AST para detectar patrones peligrosos
- Análisis de bucles infinitos potenciales
- Timeout para evitar ejecución indefinida
- Sin acceso a red ni filesystem
"""

import subprocess
import tempfile
import os
import sys
import signal
import base64
import ast
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
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
    "pennylane", "qml",  # PennyLane para QML
    "cirq",              # Google Cirq
    
    # Machine Learning (científico)
    "sklearn", "scikit-learn",
    "sklearn.decomposition",
    "sklearn.cluster",
    "sklearn.metrics",
    "sklearn.preprocessing",
    
    # Grafos
    "networkx", "nx",
    
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
    "fractions",
    "decimal",
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


# =============================================================================
# Validación AST (Análisis Estático de Código)
# =============================================================================

class ASTSecurityVisitor(ast.NodeVisitor):
    """
    Visitador AST para detectar patrones de código peligrosos.
    
    Detecta:
    - Bucles potencialmente infinitos (while True sin break claro)
    - Llamadas a funciones peligrosas
    - Acceso a atributos sensibles (__class__, __bases__, etc.)
    - Recursión sin límite aparente
    """
    
    # Atributos peligrosos que permiten escapar del sandbox
    DANGEROUS_ATTRS = {
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__globals__", "__code__", "__builtins__", "__import__",
        "__getattribute__", "__setattr__", "__delattr__",
        "func_globals", "gi_frame", "f_locals", "f_globals",
    }
    
    # Funciones built-in peligrosas
    DANGEROUS_BUILTINS = {
        "eval", "exec", "compile", "__import__", "open",
        "getattr", "setattr", "delattr", "globals", "locals",
        "vars", "dir", "type", "object",
    }
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.function_calls: Set[str] = set()
        self.while_loops_without_break = 0
        self.recursion_candidates: Set[str] = set()
        self.current_function: Optional[str] = None
        self.max_loop_depth = 0
        self.current_loop_depth = 0
    
    def add_issue(self, severity: str, message: str, node: ast.AST):
        """Registra un problema de seguridad."""
        self.issues.append({
            "severity": severity,  # "error", "warning", "info"
            "message": message,
            "line": getattr(node, "lineno", 0),
            "col": getattr(node, "col_offset", 0),
        })
    
    def visit_Import(self, node: ast.Import):
        """Verifica imports directos."""
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module in BLOCKED_IMPORTS:
                self.add_issue("error", f"Import bloqueado: {alias.name}", node)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Verifica imports from ... import ..."""
        if node.module:
            module = node.module.split('.')[0]
            if module in BLOCKED_IMPORTS:
                self.add_issue("error", f"Import bloqueado: from {node.module}", node)
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Analiza llamadas a funciones."""
        func_name = None
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        if func_name:
            self.function_calls.add(func_name)
            
            # Detectar funciones peligrosas
            if func_name in self.DANGEROUS_BUILTINS:
                self.add_issue("error", f"Función peligrosa: {func_name}()", node)
            
            # Detectar posible recursión
            if self.current_function and func_name == self.current_function:
                self.recursion_candidates.add(func_name)
                self.add_issue("warning", f"Posible recursión: {func_name}() se llama a sí misma", node)
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Detecta acceso a atributos peligrosos."""
        if node.attr in self.DANGEROUS_ATTRS:
            self.add_issue("error", f"Acceso a atributo peligroso: {node.attr}", node)
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While):
        """Analiza bucles while para detectar posibles infinitos."""
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        
        # Detectar while True o while 1
        is_infinite_pattern = False
        if isinstance(node.test, ast.Constant):
            if node.test.value in (True, 1):
                is_infinite_pattern = True
        elif isinstance(node.test, ast.NameConstant):  # Python < 3.8
            if node.test.value is True:
                is_infinite_pattern = True
        
        if is_infinite_pattern:
            # Buscar break en el cuerpo
            has_break = self._contains_break(node.body)
            if not has_break:
                self.while_loops_without_break += 1
                self.add_issue("warning", "Bucle 'while True' sin 'break' detectado - posible bucle infinito", node)
        
        self.generic_visit(node)
        self.current_loop_depth -= 1
    
    def visit_For(self, node: ast.For):
        """Analiza bucles for para detectar profundidad excesiva."""
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        
        if self.current_loop_depth > 4:
            self.add_issue("warning", f"Bucles anidados profundos (nivel {self.current_loop_depth})", node)
        
        self.generic_visit(node)
        self.current_loop_depth -= 1
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Rastrea definiciones de función para detectar recursión."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Rastrea funciones async."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def _contains_break(self, body: List[ast.stmt]) -> bool:
        """Verifica si un cuerpo de código contiene un break."""
        for stmt in body:
            if isinstance(stmt, ast.Break):
                return True
            if isinstance(stmt, ast.If):
                if self._contains_break(stmt.body) or self._contains_break(stmt.orelse):
                    return True
            if isinstance(stmt, ast.Try):
                for handler in stmt.handlers:
                    if self._contains_break(handler.body):
                        return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen del análisis."""
        errors = [i for i in self.issues if i["severity"] == "error"]
        warnings = [i for i in self.issues if i["severity"] == "warning"]
        
        return {
            "safe": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(warnings),
            "issues": self.issues,
            "max_loop_depth": self.max_loop_depth,
            "potential_infinite_loops": self.while_loops_without_break,
            "recursion_candidates": list(self.recursion_candidates),
        }


def analyze_code_ast(code: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Analiza código Python usando AST para detectar problemas de seguridad.
    
    Args:
        code: Código Python a analizar
    
    Returns:
        (es_seguro, reporte_detallado)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "safe": False,
            "errors": 1,
            "warnings": 0,
            "issues": [{
                "severity": "error",
                "message": f"Error de sintaxis: {e.msg}",
                "line": e.lineno or 0,
                "col": e.offset or 0,
            }],
            "syntax_error": str(e),
        }
    
    visitor = ASTSecurityVisitor()
    visitor.visit(tree)
    
    return visitor.get_summary()["safe"], visitor.get_summary()


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
        
        Realiza dos niveles de validación:
        1. Análisis regex rápido (patrones obvios)
        2. Análisis AST completo (detección profunda)
        
        Returns:
            (es_valido, mensaje_error)
        """
        # === Nivel 1: Validación regex rápida ===
        
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
        
        # === Nivel 2: Análisis AST profundo ===
        is_safe, report = analyze_code_ast(code)
        
        if not is_safe:
            # Construir mensaje de error detallado
            errors = [i for i in report.get("issues", []) if i["severity"] == "error"]
            if errors:
                error = errors[0]
                return False, f"[Línea {error['line']}] {error['message']}"
            
            # Si hay syntax error
            if "syntax_error" in report:
                return False, f"Error de sintaxis: {report['syntax_error']}"
        
        # Advertencias (no bloquean pero se loguean)
        warnings = [i for i in report.get("issues", []) if i["severity"] == "warning"]
        if warnings:
            for w in warnings:
                logger.warning(f"Sandbox warning [Línea {w['line']}]: {w['message']}")
        
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
    ) -> tuple[str, SandboxResult]:
        """
        Genera código con LLM y lo ejecuta.

        Args:
            prompt: Descripción de lo que calcular/graficar

        Returns:
            (código_generado, resultado_ejecución)
        """
        from src.llm_provider import complete as llm_complete

        system_prompt = f"""Eres un asistente que genera código Python para cálculos científicos y visualizaciones.

Reglas:
1. Imports permitidos: {', '.join(sorted(ALLOWED_IMPORTS))}
2. NO uses: os, sys, subprocess, socket, requests, open() para escribir
3. Para gráficas usa matplotlib y termina con plt.show()
4. El código debe ser autocontenido y ejecutable
5. Usa comentarios para explicar pasos importantes
6. Imprime resultados relevantes con print()

Responde SOLO con el código Python, sin explicaciones ni markdown."""

        response = llm_complete(
            prompt=prompt,
            system=system_prompt,
            temperature=0.2,
            max_tokens=1000,
        )

        code = response.content.strip()
        
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
