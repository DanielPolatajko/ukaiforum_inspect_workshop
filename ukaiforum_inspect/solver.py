from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, execute_tools

@solver
def agentic_solver():
    """
    A minimal agentic solver loop that demonstrates tool use.

    The loop:
    1. Calls the model to generate a response
    2. Checks if the model requested tool calls
    3. If tool calls exist, executes them and adds results to conversation
    4. Repeats until no more tool calls or max iterations reached
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        max_iterations = 10

        model = get_model()

        for _ in range(max_iterations):
            state = await model.generate(state.messages, tools=state.tools)

            if state.output.message.tool_calls:
                tool_messages, output = await execute_tools(
                    state.output.message,
                    state.tools
                )

                state.messages.extend(tool_messages)

                state.output = output
            else:
                break

        return state

    return solve