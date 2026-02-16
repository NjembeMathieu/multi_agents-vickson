"""
Orchestrateur - Gestion du flux de travail multi-agents avec LangGraph
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from state import GraphState
from config import MAX_CORRECTION_LOOPS 
import time
from datetime import datetime

# Import des agents
from agents.agent_context import agent_context_node
from agents.agent_program import agent_program_node
from agents.agent_similarite import agent_similarite_node
from agents.agent_writer import agent_writer_node
from agents.agent_validation import agent_validation_node
from agents.agent_export import agent_export_node


class PerformanceTracker:
    """Tracker minimal des performances"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.agent_times = {}
        self.agent_calls = {}
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def track_agent(self, name, duration):
        if name not in self.agent_times:
            self.agent_times[name] = []
            self.agent_calls[name] = 0
        self.agent_times[name].append(duration)
        self.agent_calls[name] += 1
    
    def get_summary(self):
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        summary = {
            'total_time': round(total_time, 2),
            'agents': {}
        }
        for name in self.agent_times:
            avg_time = sum(self.agent_times[name]) / len(self.agent_times[name])
            summary['agents'][name] = {
                'avg_time': round(avg_time, 2),
                'calls': self.agent_calls[name]
            }
        return summary


class Orchestrateur:
    """
    Orchestrateur du système multi-agents
    Gère le flux de travail et les décisions de routage
    """
    
    def __init__(self):
        self.performance = PerformanceTracker()  # <-- DÉFINIR AVANT build_graph
        self.graph = self._build_graph()         # <-- build_graph utilise performance
    
    def _should_continue_correction(self, state: GraphState) -> Literal["writer", "export"]:
        """
        Décide si on continue les corrections ou si on exporte
        """
        validation = state.validation
        
        # Si la fiche est validée, on exporte
        if validation.valide:
            return "export"
        
        # Si on a atteint le maximum d'itérations
        if state.compteur_boucles >= MAX_CORRECTION_LOOPS:
            print(f"⚠️ Limite d'itérations atteinte ({MAX_CORRECTION_LOOPS})")
            print(f"   Score final: {validation.score_conformite}%")
            # On exporte quand même avec le meilleur score obtenu
            return "export"
        
        # Sinon, on retourne au writer pour correction
        print(f"🔄 Correction nécessaire (tentative {state.compteur_boucles + 1}/{MAX_CORRECTION_LOOPS})")
        print(f"   Score actuel: {validation.score_conformite}%")
        
        # Incrémenter le compteur de boucles
        state.compteur_boucles += 1
        
        return "writer"
    
    def _wrap_agent(self, name: str, agent_func):
        """Wrapper pour mesurer le temps des agents"""
        def wrapped(state):
            start = time.time()
            try:
                result = agent_func(state)
                duration = time.time() - start
                self.performance.track_agent(name, duration)
                return result
            except Exception as e:
                duration = time.time() - start
                self.performance.track_agent(name, duration)
                raise e
        return wrapped
    
    def _build_graph(self) -> StateGraph:
        """
        Construit le graphe de workflow
        """
        # Créer le graphe
        workflow = StateGraph(GraphState)
        
        # Ajouter les nœuds (agents) avec wrapper
        workflow.add_node("context", self._wrap_agent("context", agent_context_node))
        workflow.add_node("program", self._wrap_agent("program", agent_program_node))
        workflow.add_node("similarite", self._wrap_agent("similarite", agent_similarite_node))
        workflow.add_node("writer", self._wrap_agent("writer", agent_writer_node))
        workflow.add_node("validation", self._wrap_agent("validation", agent_validation_node))
        workflow.add_node("export", self._wrap_agent("export", agent_export_node))
        
        # Définir le point d'entrée
        workflow.set_entry_point("context")
        
        # Définir les arêtes (flux séquentiel)
        workflow.add_edge("context", "program")
        workflow.add_edge("program", "similarite")
        workflow.add_edge("similarite", "writer")
        workflow.add_edge("writer", "validation")
        
        # Branchement conditionnel après validation
        workflow.add_conditional_edges(
            "validation",
            self._should_continue_correction,
            {
                "writer": "writer",    # Retour pour correction
                "export": "export"     # Export final
            }
        )
        
        # Fin du workflow après export
        workflow.add_edge("export", END)
        
        # Compiler le graphe
        return workflow.compile()
    
    def run(self, state: GraphState) -> GraphState:
        """
        Exécute le workflow complet
        """
        print("="*60)
        print("🚀 DÉMARRAGE DE LA GÉNÉRATION DE FICHE DE COURS")
        print("="*60)
        
        self.performance.start()  # Démarrer le chrono
        
        # Exécuter le graphe
        final_state = self.graph.invoke(state)
        
        self.performance.stop()  # Arrêter le chrono
        summary = self.performance.get_summary()
        
        print("="*60)
        print("✅ GÉNÉRATION TERMINÉE")
        print(f"   Score final: {final_state.validation.score_conformite}%")
        print(f"   Statut: {'Validée ✓' if final_state.validation.valide else 'Exportée avec réserves ⚠️'}")
        print(f"   Itérations: {final_state.compteur_boucles + 1}")
        print(f"   ⏱️  Temps total: {summary['total_time']}s")
        print("="*60)
        
        return final_state
    
    def visualize(self, output_path: str = "workflow_graph.png"):
        """
        Visualise le graphe de workflow
        Nécessite graphviz installé
        """
        try:
            from langchain_core.runnables.graph import MermaidDrawMethod
            
            # Générer le diagramme Mermaid
            mermaid_png = self.graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API
            )
            
            with open(output_path, 'wb') as f:
                f.write(mermaid_png)
            
            print(f"📊 Graphe de workflow sauvegardé: {output_path}")
        except Exception as e:
            print(f"⚠️ Impossible de générer le graphe: {e}")


def create_orchestrator() -> Orchestrateur:
    """Fonction factory pour créer l'orchestrateur"""
    return Orchestrateur()