"""Experimental paradigms for post-training evaluation."""

from src.experiments.paradigm_base import (
    TrialConfig,
    TrialSet,
    ConditionData,
    ExperimentResult,
    ParadigmBase,
)
from src.experiments.hidden_state import HiddenStateParadigm
from src.experiments.omission import OmissionParadigm
from src.experiments.ambiguous import AmbiguousParadigm
from src.experiments.cue_local_competitor import CueLocalCompetitorParadigm
from src.experiments.task_relevance import TaskRelevanceParadigm
from src.experiments.surprise_dissociation import SurpriseDissociationParadigm

ALL_PARADIGMS = {
    "hidden_state": HiddenStateParadigm,
    "omission": OmissionParadigm,
    "ambiguous": AmbiguousParadigm,
    "cue_local_competitor": CueLocalCompetitorParadigm,
    "task_relevance": TaskRelevanceParadigm,
    "surprise_dissociation": SurpriseDissociationParadigm,
}
