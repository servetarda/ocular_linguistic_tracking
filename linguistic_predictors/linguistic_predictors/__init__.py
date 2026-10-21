"""linguistic_predictors: word-by-word predictability of German text at five levels.

Lexical, syntactic (part of speech), semantic, phonemic and syllabic
surprisal and entropy, following Heilbron, Armeni, Schoffelen, Hagoort &
de Lange (2022), "A hierarchy of linguistic predictions during natural
language comprehension", PNAS 119(32), e2201968119. The syllabic level is an
extension.

    from linguistic_predictors.components_de import load_components
    from linguistic_predictors.pipeline import compute_predictors
    comps = load_components("/path/to/german_mfa.dict")
    words, meta = compute_predictors("Die Nachricht kam völlig überraschend.", comps)

See docs/methods_report.html for the method and its validation.
"""
from .pipeline import PipelineOptions, compute_predictors

__all__ = ["PipelineOptions", "compute_predictors"]
