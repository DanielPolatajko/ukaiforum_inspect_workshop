from inspect_ai import task, Task
from inspect_ai.agent import react
from ukaiforum_inspect.dataset import decryption_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import system_message
from ukaiforum_inspect.tools import str_to_binary, hex_to_binary, binary_decoding, xor_binary
from ukaiforum_inspect.prompts.system import SYSTEM_PROMPT_REACT

@task
def react_task():
  return Task(
      dataset=decryption_dataset,
      solver=[
          system_message(SYSTEM_PROMPT_REACT),
          react(tools=[str_to_binary(), hex_to_binary(), binary_decoding(), xor_binary()]),
      ],
      scorer=match(),
  )