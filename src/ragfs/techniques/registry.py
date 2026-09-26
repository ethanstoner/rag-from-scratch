"""Name -> technique class. Drives the CLI and the eval."""
import importlib

# (module, class) in course-lesson order; imported lazily so the torch-backed ones
# cost nothing unless asked for.
TECHNIQUES = {
    "baseline": ("baseline", "Baseline"),
    "hybrid": ("baseline", "Hybrid"),
    "multi_query": ("query_translation", "MultiQuery"),
    "rag_fusion": ("query_translation", "RagFusion"),
    "decomposition": ("query_translation", "Decomposition"),
    "decomposition_individual": ("query_translation", "DecompositionIndividual"),
    "step_back": ("query_translation", "StepBack"),
    "hyde": ("query_translation", "HyDE"),
    "routing_logical": ("routing", "LogicalRouting"),
    "routing_semantic": ("routing", "SemanticRouting"),
    "query_construction": ("routing", "QueryConstruction"),
    "multi_representation": ("indexing", "MultiRepresentation"),
    "raptor": ("indexing", "Raptor"),
    "colbert": ("neural", "ColBERT"),
    "rerank": ("neural", "Rerank"),
    "crag": ("agentic", "CRAG"),
    "self_rag": ("agentic", "SelfRAG"),
    "adaptive": ("agentic", "Adaptive"),
}
NEURAL = {"colbert", "rerank"}  # need torch + model downloads


def get(name: str):
    module, cls = TECHNIQUES[name]
    return getattr(importlib.import_module(f"ragfs.techniques.{module}"), cls)


def names():
    return list(TECHNIQUES)
