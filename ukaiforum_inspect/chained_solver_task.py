from inspect_ai import task, Task
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import match
from tutorial.dataset import decryption_dataset
from tutorial.system_prompt import SYSTEM_PROMPT

@task
def decryption_task_with_chained_solver():
  return Task(
      dataset=decryption_dataset,
      solver=[
          system_message(SYSTEM_PROMPT),
          generate(),
      ],
      scorer=match(),
  )