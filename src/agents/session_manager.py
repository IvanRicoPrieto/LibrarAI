"""
Session Manager - Gestión de memoria conversacional.

Permite preguntas de seguimiento:
- "¿Y si cambio X?"
- "Expande el punto 3"
- "Más detalles sobre lo anterior"

La memoria se mantiene en sesiones identificadas por session_id,
con un historial de mensajes y contexto recuperado.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Un mensaje en la conversación."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime
    sources: List[str] = field(default_factory=list)  # chunk_ids usados
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sources=data.get("sources", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationContext:
    """Contexto de una conversación activa."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
    active_topic: Optional[str] = None  # Tema principal detectado
    entities_mentioned: List[str] = field(default_factory=list)  # Entidades mencionadas
    source_history: List[str] = field(default_factory=list)  # Chunks usados anteriormente
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "active_topic": self.active_topic,
            "entities_mentioned": self.entities_mentioned,
            "source_history": self.source_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationContext":
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            messages=[Message.from_dict(m) for m in data["messages"]],
            active_topic=data.get("active_topic"),
            entities_mentioned=data.get("entities_mentioned", []),
            source_history=data.get("source_history", [])
        )


class SessionManager:
    """
    Gestiona sesiones de conversación con memoria.
    
    Permite:
    - Crear/recuperar sesiones
    - Mantener historial de mensajes
    - Detectar preguntas de seguimiento
    - Expandir queries con contexto previo
    """
    
    # Patrones que indican pregunta de seguimiento
    FOLLOWUP_PATTERNS = [
        # Expansión
        r"(?:más|más detalles|expande|amplía|profundiza)",
        r"(?:cuéntame más|tell me more|elaborate)",
        r"(?:qué más|what else)",
        
        # Referencia a lo anterior
        r"(?:lo anterior|the previous|lo que dijiste)",
        r"(?:el punto \d+|point \d+|el apartado)",
        r"(?:sobre eso|about that|al respecto)",
        
        # Cambio condicional
        r"(?:y si|what if|qué pasaría si)",
        r"(?:en cambio|instead|pero si)",
        r"(?:ahora con|now with)",
        
        # Comparación con anterior
        r"(?:compara con|compare with|versus lo)",
        r"(?:diferencia con|difference from)",
        
        # Clarificación
        r"(?:qué quieres decir|what do you mean)",
        r"(?:no entiendo|I don't understand)",
        r"(?:explica mejor|explain better)"
    ]
    
    def __init__(
        self,
        sessions_dir: Path,
        max_history: int = 10,
        max_context_tokens: int = 2000
    ):
        """
        Args:
            sessions_dir: Directorio para persistir sesiones
            max_history: Máximo de mensajes a mantener
            max_context_tokens: Máximo de tokens de contexto a incluir
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        
        # Cache de sesiones activas
        self._active_sessions: Dict[str, ConversationContext] = {}
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Crea una nueva sesión de conversación.
        
        Args:
            session_id: ID opcional (si no se proporciona, se genera uno)
            
        Returns:
            session_id
        """
        if session_id is None:
            session_id = self._generate_session_id()
        
        now = datetime.now()
        context = ConversationContext(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            messages=[]
        )
        
        self._active_sessions[session_id] = context
        self._save_session(context)
        
        logger.info(f"Sesión creada: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """
        Obtiene una sesión existente.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            ConversationContext o None si no existe
        """
        # Buscar en cache
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        
        # Cargar de disco
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            try:
                with open(session_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                context = ConversationContext.from_dict(data)
                self._active_sessions[session_id] = context
                return context
            except Exception as e:
                logger.error(f"Error cargando sesión {session_id}: {e}")
        
        return None
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: List[str] = None,
        metadata: Dict = None
    ):
        """
        Añade un mensaje a la sesión.
        
        Args:
            session_id: ID de la sesión
            role: "user" o "assistant"
            content: Contenido del mensaje
            sources: chunk_ids usados (si role=assistant)
            metadata: Metadatos adicionales
        """
        context = self.get_session(session_id)
        if context is None:
            logger.warning(f"Sesión {session_id} no encontrada, creando nueva")
            self.create_session(session_id)
            context = self.get_session(session_id)
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            sources=sources or [],
            metadata=metadata or {}
        )
        
        context.messages.append(message)
        context.updated_at = datetime.now()
        
        # Actualizar historial de fuentes
        if sources:
            for source in sources:
                if source not in context.source_history:
                    context.source_history.append(source)
        
        # Limitar historial
        if len(context.messages) > self.max_history:
            context.messages = context.messages[-self.max_history:]
        
        self._save_session(context)
    
    def is_followup_query(self, query: str, session_id: str) -> Tuple[bool, str]:
        """
        Detecta si una query es de seguimiento.
        
        Args:
            query: La consulta del usuario
            session_id: ID de la sesión
            
        Returns:
            Tupla (es_followup, tipo_followup)
        """
        import re
        
        query_lower = query.lower()
        
        # Verificar patrones de seguimiento
        for pattern in self.FOLLOWUP_PATTERNS:
            if re.search(pattern, query_lower):
                # Determinar tipo
                if any(p in query_lower for p in ["más", "expande", "profundiza", "more"]):
                    return True, "expansion"
                elif any(p in query_lower for p in ["y si", "what if", "qué pasaría"]):
                    return True, "conditional"
                elif any(p in query_lower for p in ["compara", "compare", "versus"]):
                    return True, "comparison"
                elif any(p in query_lower for p in ["punto", "point", "apartado"]):
                    return True, "reference"
                else:
                    return True, "clarification"
        
        # También es seguimiento si la sesión tiene contexto y la query es corta
        context = self.get_session(session_id)
        if context and len(context.messages) > 0:
            # Query muy corta probablemente es seguimiento
            if len(query.split()) < 5:
                return True, "implicit"
        
        return False, "none"
    
    def expand_query_with_context(
        self,
        query: str,
        session_id: str,
        followup_type: str = "none"
    ) -> str:
        """
        Expande una query con contexto de la conversación.
        
        Args:
            query: La consulta original
            session_id: ID de la sesión
            followup_type: Tipo de seguimiento detectado
            
        Returns:
            Query expandida con contexto
        """
        context = self.get_session(session_id)
        if context is None or len(context.messages) == 0:
            return query
        
        # Obtener última respuesta del asistente
        last_assistant_msg = None
        last_user_msg = None
        
        for msg in reversed(context.messages):
            if msg.role == "assistant" and last_assistant_msg is None:
                last_assistant_msg = msg
            elif msg.role == "user" and last_user_msg is None:
                last_user_msg = msg
            if last_assistant_msg and last_user_msg:
                break
        
        # Construir contexto según tipo de seguimiento
        if followup_type == "expansion":
            if last_assistant_msg:
                # Extraer tema principal de la respuesta anterior
                topic = self._extract_topic(last_assistant_msg.content)
                if topic:
                    return f"{query} sobre {topic}"
            
        elif followup_type == "conditional":
            if last_user_msg:
                return f"En el contexto de '{last_user_msg.content}', {query}"
            
        elif followup_type == "reference":
            # Intentar extraer referencia específica
            import re
            match = re.search(r"(?:punto|point|apartado)\s*(\d+)", query.lower())
            if match and last_assistant_msg:
                point_num = match.group(1)
                # Buscar el punto en la respuesta anterior
                points = re.findall(r'\d+\.\s*([^\n]+)', last_assistant_msg.content)
                if points and int(point_num) <= len(points):
                    referenced_point = points[int(point_num) - 1]
                    return f"Explica más sobre: {referenced_point}"
        
        elif followup_type == "implicit" or followup_type == "clarification":
            if last_user_msg:
                return f"{last_user_msg.content}: {query}"
        
        return query
    
    def get_conversation_summary(self, session_id: str) -> str:
        """
        Obtiene un resumen de la conversación para contexto del LLM.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Resumen formateado
        """
        context = self.get_session(session_id)
        if context is None or len(context.messages) == 0:
            return ""
        
        # Construir resumen
        summary_parts = []
        
        # Solo incluir últimos N mensajes
        recent = context.messages[-4:]  # Últimos 2 turnos
        
        for msg in recent:
            role_label = "Usuario" if msg.role == "user" else "Asistente"
            # Truncar contenido largo
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            summary_parts.append(f"{role_label}: {content}")
        
        return "\n".join(summary_parts)
    
    def get_relevant_source_history(
        self,
        session_id: str,
        max_sources: int = 5
    ) -> List[str]:
        """
        Obtiene chunk_ids relevantes de la conversación anterior.
        
        Útil para priorizar fuentes ya usadas en seguimientos.
        
        Args:
            session_id: ID de la sesión
            max_sources: Máximo de fuentes a retornar
            
        Returns:
            Lista de chunk_ids
        """
        context = self.get_session(session_id)
        if context is None:
            return []
        
        # Retornar las últimas fuentes usadas
        return context.source_history[-max_sources:]
    
    def update_topic(self, session_id: str, topic: str):
        """Actualiza el tema activo de la sesión."""
        context = self.get_session(session_id)
        if context:
            context.active_topic = topic
            self._save_session(context)
    
    def add_entity(self, session_id: str, entity: str):
        """Añade una entidad mencionada a la sesión."""
        context = self.get_session(session_id)
        if context and entity not in context.entities_mentioned:
            context.entities_mentioned.append(entity)
            self._save_session(context)
    
    def list_sessions(self, limit: int = 10) -> List[Dict]:
        """Lista las sesiones más recientes."""
        sessions = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "message_count": len(data["messages"]),
                    "topic": data.get("active_topic")
                })
                if len(sessions) >= limit:
                    break
            except Exception:
                pass
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            return True
        return False
    
    def _generate_session_id(self) -> str:
        """Genera un ID único de sesión."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}-{id(self)}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:12]
    
    def _save_session(self, context: ConversationContext):
        """Guarda una sesión a disco."""
        session_path = self.sessions_dir / f"{context.session_id}.json"
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(context.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _extract_topic(self, content: str) -> Optional[str]:
        """Extrae el tema principal de un contenido."""
        # Estrategia simple: buscar primer sustantivo técnico
        import re
        
        # Patrones de términos técnicos
        patterns = [
            r"(?:algoritmo|protocolo|teorema|concepto)\s+(?:de\s+)?(\w+)",
            r"(\w+)\s+(?:cuántic[oa]|quantum)",
            r"(?:el|la)\s+(\w+)\s+(?:es|consiste|permite)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return match.group(1).capitalize()
        
        return None


# Singleton para acceso global
_session_manager: Optional[SessionManager] = None


def get_session_manager(sessions_dir: Path = None) -> SessionManager:
    """Obtiene el singleton del session manager."""
    global _session_manager
    
    if _session_manager is None:
        if sessions_dir is None:
            sessions_dir = Path(__file__).parent.parent.parent / "logs" / "sessions"
        _session_manager = SessionManager(sessions_dir)
    
    return _session_manager
