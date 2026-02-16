"""
Agent Writer - Producteur Adaptatif & Rédacteur
Génère le contenu de la fiche avec Gemini
"""
import streamlit as st
from google import genai
import time
from typing import Dict, List
import json
from state import GraphState, FicheContent
from config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS
from google.genai import types

# Create a single client instance
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


class AgentWriter:

    """Agent de génération de contenu via Gemini"""

    def __init__(self):
        self.model = GEMINI_MODEL
        self.generation_config = {
            "temperature": GEMINI_TEMPERATURE,
            "max_output_tokens": GEMINI_MAX_TOKENS,
        }

    def _construire_prompt_creation_complete(self, state: GraphState) -> str:
        """Construit le prompt pour une création complète"""
        input_data = state.input_data
        contexte = state.contexte
        referentiel = state.referentiel
        gabarit = self.gabarits[referentiel.gabarit]

        prompt = f"""Tu es un expert pédagogue chargé de créer une fiche de cours professionnelle et complète.

INFORMATIONS OBLIGATOIRES À INCLURE:
- Établissement: {input_data.etablissement}
- Ville: {input_data.ville}
- Année scolaire: {input_data.annee_scolaire}
- Classe: {input_data.classe}
- Professeur: {input_data.nom_professeur}
- Matière: {input_data.matiere}
- Thème/Chapitre: {input_data.theme_chapitre}
- Séquence/Date: {input_data.sequence_ou_date}
- Volume horaire: {input_data.volume_horaire}h

CONTEXTE PÉDAGOGIQUE:
- Cycle: {contexte.cycle}
- Niveau exact: {contexte.niveau_exact}
- Type de cours: {referentiel.gabarit}

OBJECTIFS PÉDAGOGIQUES (À RESPECTER STRICTEMENT):
{chr(10).join(f"- {obj}" for obj in referentiel.objectifs_officiels)}

COMPÉTENCES À DÉVELOPPER:
{chr(10).join(f"- {comp}" for comp in referentiel.competences)}

STRUCTURE ATTENDUE:
{chr(10).join(f"- {section}" for section in gabarit['sections'])}

NOMBRE D'ACTIVITÉS: {gabarit['activites_min']} à {gabarit['activites_max']} activités
TYPE D'ÉVALUATION: {gabarit['evaluation']}

ANCRAGE LOCAL (IMPORTANT):
- Ville: {contexte.ancrage_local['ville']}
- {contexte.ancrage_local['suggestions']}
"""

        if state.necessite_situation_probleme:
            prompt += f"""

SITUATION-PROBLÈME (OBLIGATOIRE pour le Secondaire):
Tu DOIS créer une situation-problème concrète et engageante qui:
1. Part d'un contexte réel lié à {input_data.ville} ou {input_data.etablissement}
2. Pose un défi intellectuel en rapport avec {input_data.theme_chapitre}
3. Mobilise les objectifs pédagogiques définis
4. Permet aux élèves de {input_data.classe} de s'approprier le thème

La situation-problème doit être détaillée (150-250 mots) et intégrée en début de fiche.
"""

        prompt += """
SOURCE DE RÉFÉRENCE:
""" + (f"Document: {referentiel.source_document}" if referentiel.source_document else "Référentiel générique") + """

CONSIGNES DE RÉDACTION:
1. Respecte STRICTEMENT tous les objectifs pédagogiques listés
2. Utilise un langage adapté au niveau {niveau}
3. Intègre des exemples concrets et ancrés localement
4. Structure le cours de manière progressive
5. Propose des activités variées et engageantes
6. Inclus une évaluation pertinente

FORMAT DE SORTIE (JSON):
Réponds UNIQUEMENT avec un objet JSON valide ayant cette structure exacte:
{{
    "titre": "Titre de la fiche",
    "etablissement": "{etablissement}",
    "ville": "{ville}",
    "classe": "{classe}",
    "objectifs": ["objectif 1", "objectif 2", ...],
    "situation_probleme": "Texte de la situation-problème ou null",
    "introduction": "Introduction du cours",
    "developpement": "Contenu principal détaillé",
    "activites": [
        {{"titre": "Activité 1", "description": "...", "duree": "20min"}},
        {{"titre": "Activité 2", "description": "...", "duree": "30min"}}
    ],
    "evaluation": "Description de l'évaluation",
    "conclusion": "Conclusion et ouvertures",
    "references": ["Référence 1", "Référence 2"]
}}

Génère maintenant la fiche complète en JSON:
""".format(
            niveau=contexte.niveau_exact,
            etablissement=input_data.etablissement,
            ville=input_data.ville,
            classe=input_data.classe
        )

        return prompt

    def _construire_prompt_adaptation(self, state: GraphState) -> str:
        """Construit le prompt pour une adaptation de fiche existante"""
        input_data = state.input_data
        contexte = state.contexte
        fiche_existante = state.similarite.contenu_existant

        prompt = f"""Tu es un expert pédagogue chargé d'ADAPTER une fiche de cours existante.

NOUVELLE CONFIGURATION:
- Établissement: {input_data.etablissement}
- Ville: {input_data.ville}
- Année scolaire: {input_data.annee_scolaire}
- Classe: {input_data.classe}
- Professeur: {input_data.nom_professeur}
- Matière: {input_data.matiere}
- Thème: {input_data.theme_chapitre}
- Volume horaire: {input_data.volume_horaire}h

FICHE EXISTANTE À ADAPTER:
{fiche_existante}

CONSIGNES D'ADAPTATION:
1. Conserve la structure et les objectifs principaux
2. ADAPTE les exemples et situations au contexte de {input_data.ville}
3. Actualise les références à l'établissement et au professeur
4. Ajuste la complexité si nécessaire pour le niveau {contexte.niveau_exact}
5. Modifie la situation-problème pour l'ancrer localement à {input_data.ville}

FORMAT DE SORTIE (JSON):
{{
    "titre": "Titre adapté",
    "etablissement": "{input_data.etablissement}",
    "ville": "{input_data.ville}",
    "classe": "{input_data.classe}",
    "objectifs": [...],
    "situation_probleme": "...",
    "introduction": "...",
    "developpement": "...",
    "activites": [...],
    "evaluation": "...",
    "conclusion": "...",
    "references": [...]
}}

Génère la fiche adaptée en JSON:
"""

        return prompt

    def _construire_prompt_correction(self, state: GraphState) -> str:
        """Construit le prompt pour corriger une fiche rejetée"""
        validation = state.validation
        fiche_actuelle = state.fiche

        fiche_json = fiche_actuelle.model_dump_json(indent=2)

        prompt = f"""Tu es un expert pédagogue chargé de CORRIGER une fiche de cours qui a été rejetée.

FICHE ACTUELLE:
{fiche_json}

PROBLÈMES IDENTIFIÉS:
Score de conformité: {validation.score_conformite}%
{chr(10).join(f"- {commentaire}" for commentaire in validation.commentaires)}

ÉLÉMENTS MANQUANTS:
{chr(10).join(f"- {elem}" for elem in validation.elements_manquants)}

CORRECTIONS REQUISES:
{chr(10).join(f"- {correction}" for correction in validation.corrections_requises)}

CONSIGNES DE CORRECTION:
1. Corrige UNIQUEMENT les parties problématiques identifiées
2. Conserve ce qui fonctionne déjà
3. Assure-toi que TOUS les objectifs pédagogiques sont traités
4. Vérifie que la structure est complète

FORMAT DE SORTIE (JSON):
Réponds avec la fiche corrigée au format JSON comme précédemment.

Génère la fiche corrigée en JSON:
"""

        return prompt

    @property
    def gabarits(self):
        return {
            "court": {
                "sections": ["Introduction", "Contenu principal", "Exercice", "Conclusion"],
                "activites_min": 1,
                "activites_max": 2,
                "evaluation": "Exercice d'application"
            },
            "moyen": {
                "sections": ["Introduction", "Cours magistral", "Activités", "Exercices", "Évaluation"],
                "activites_min": 2,
                "activites_max": 4,
                "evaluation": "QCM + Exercice"
            },
            "etendu": {
                "sections": ["Introduction", "Cours détaillé", "Activités pratiques",
                             "Travaux dirigés", "Évaluation formative", "Conclusion"],
                "activites_min": 4,
                "activites_max": 6,
                "evaluation": "Évaluation complète"
            }
        }

    def _parser_json_response(self, response_text: str) -> Dict:
        """Parse la réponse JSON de Gemini"""
        text = response_text.strip()

        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Erreur JSON: {e}")
            print(f"Texte reçu: {text[:500]}")
            return {
                "titre": "Erreur de génération",
                "etablissement": "",
                "ville": "",
                "classe": "",
                "objectifs": [],
                "situation_probleme": None,
                "introduction": "Erreur lors de la génération",
                "developpement": "Le contenu n'a pas pu être généré correctement",
                "activites": [],
                "evaluation": "",
                "conclusion": "",
                "references": []
            }

    def process(self, state: GraphState) -> GraphState:
        """
        Génère ou adapte le contenu de la fiche
        """
        if state.compteur_boucles > 0 and state.validation:
            prompt = self._construire_prompt_correction(state)
            state.historique_corrections.append(f"Itération {state.compteur_boucles}: Correction après rejet")
        elif state.mode_generation == "adaptation":
            prompt = self._construire_prompt_adaptation(state)
        else:
            prompt = self._construire_prompt_creation_complete(state)

        try:
            response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.generation_config["temperature"],
                max_output_tokens=self.generation_config["max_output_tokens"],
                response_mime_type="application/json"
            )   
            )
            
            time.sleep(4)

            raw_text = response.text

            if not raw_text.strip().endswith("}"):
                print("⚠ Response truncated — forcing correction")
                raw_text = raw_text + "}"

            fiche_dict = self._parser_json_response(raw_text)
            fiche = FicheContent(**fiche_dict)
            state.fiche = fiche

        except Exception as e:
            print(f"Erreur lors de la génération: {e}")
            state.fiche = FicheContent(
                titre=f"Fiche de cours - {state.input_data.theme_chapitre}",
                etablissement=state.input_data.etablissement,
                ville=state.input_data.ville,
                classe=state.input_data.classe,
                objectifs=state.referentiel.objectifs_officiels if state.referentiel else [],
                introduction="Erreur lors de la génération du contenu.",
                developpement="Le contenu détaillé n'a pas pu être généré.",
                evaluation="À définir",
                conclusion="À compléter"
            )


        return state


def agent_writer_node(state: GraphState) -> GraphState:
    """Node LangGraph pour l'Agent Writer"""
    agent = AgentWriter()
    return agent.process(state)
