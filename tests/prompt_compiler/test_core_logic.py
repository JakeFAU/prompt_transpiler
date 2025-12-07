from prompt_compiler.core_logic import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
    compile_pipeline,
)


def test_core_logic_exports():
    assert CandidatePrompt is not None
    assert OriginalPrompt is not None
    assert ScoringAlgorithm is not None
    assert compile_pipeline is not None
