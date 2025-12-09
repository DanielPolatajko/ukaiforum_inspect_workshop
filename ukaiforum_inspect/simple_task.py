from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.scorer import match
from tutorial.dataset import decryption_dataset

@task
def simple_decryption_task():
  return Task(
      dataset=decryption_dataset,
      solver=generate(),
      scorer=match(),
  )