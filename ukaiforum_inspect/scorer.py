from inspect_ai.scorer import scorer, CORRECT, INCORRECT, Target, Score, Scorer, accuracy, stderr
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy(), stderr()])
def answer_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        if "<answer>" in state.output.completion:
            answer = state.output.completion.split("<answer>")[1].split("</answer>")[0]
            return Score(value=(CORRECT if answer.strip() == target.text.strip() else INCORRECT))
        else:
            return Score(value=INCORRECT)
    return score