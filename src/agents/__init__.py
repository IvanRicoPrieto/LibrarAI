"""
Módulo de Agents - Agentes inteligentes para RAG.

Exporta el router, planificador, crítico y session manager.
"""

from .router import QueryRouter, RoutingDecision
from .planner import QueryPlanner, ExecutionPlan
from .critic import ResponseCritic, CritiqueResult
from .session_manager import (
    SessionManager,
    ConversationContext,
    Message,
    get_session_manager
)

__all__ = [
    "QueryRouter",
    "RoutingDecision",
    "QueryPlanner",
    "ExecutionPlan",
    "ResponseCritic",
    "CritiqueResult",
    "SessionManager",
    "ConversationContext",
    "Message",
    "get_session_manager"
]
