from inspect_ai import task, Task
from tutorial.dataset import decryption_dataset
from inspect_ai.scorer import match
from tutorial.tools import str_to_binary, hex_to_binary, binary_decoding
from tutorial.solver import agentic_solver
from inspect_ai.solver import system_message, use_tools
from tutorial.system_prompt import SYSTEM_PROMPT_TOOLS

@task
def decryption_task_with_tools():
  return Task(
      dataset=decryption_dataset,
      solver=[
          system_message(SYSTEM_PROMPT_TOOLS),
          use_tools(
              str_to_binary(),
              hex_to_binary(),
              binary_decoding(),
          ),
          agentic_solver(),
      ],
      scorer=match(),
  )