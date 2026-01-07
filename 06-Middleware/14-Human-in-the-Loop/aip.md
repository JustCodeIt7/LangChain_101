# Middleware | LangChain Reference

[](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain/middleware.md "Edit this page")

Reference docs

This page contains **reference documentation** for Middleware. See [the docs](https://docs.langchain.com/oss/python/langchain/middleware) for conceptual guides, tutorials, and examples on using Middleware.

Middleware classes[¶](https://reference.langchain.com/python/langchain/middleware/#middleware-classes "Copy anchor link to this section for reference")
-------------------------------------------------------------------------------------------------------------------------------------------------------

LangChain provides prebuilt middleware for common agent use cases:

| CLASS | DESCRIPTION |
| --- | --- |
| [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.SummarizationMiddleware) | Automatically summarize conversation history when approaching token limits |
| [`HumanInTheLoopMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.HumanInTheLoopMiddleware) | Pause execution for human approval of tool calls |
| [`ModelCallLimitMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelCallLimitMiddleware) | Limit the number of model calls to prevent excessive costs |
| [`ToolCallLimitMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ToolCallLimitMiddleware) | Control tool execution by limiting call counts |
| [`ModelFallbackMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelFallbackMiddleware) | Automatically fallback to alternative models when primary fails |
| [`PIIMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.PIIMiddleware) | Detect and handle Personally Identifiable Information |
| [`TodoListMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.TodoListMiddleware) | Equip agents with task planning and tracking capabilities |
| [`LLMToolSelectorMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.LLMToolSelectorMiddleware) | Use an LLM to select relevant tools before calling main model |
| [`ToolRetryMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ToolRetryMiddleware) | Automatically retry failed tool calls with exponential backoff |
| [`LLMToolEmulator`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.LLMToolEmulator) | Emulate tool execution using LLM for testing purposes |
| [`ContextEditingMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ContextEditingMiddleware) | Manage conversation context by trimming or clearing tool uses |
| [`ShellToolMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ShellToolMiddleware) | Expose a persistent shell session to agents for command execution |
| [`FilesystemFileSearchMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.FilesystemFileSearchMiddleware) | Provide Glob and Grep search tools over filesystem files |
| [`AgentMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware) | Base middleware class for creating custom middleware |

Decorators[¶](https://reference.langchain.com/python/langchain/middleware/#decorators "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------

Create custom middleware using these decorators:

| DECORATOR | DESCRIPTION |
| --- | --- |
| [`@before_agent`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_agent) | Execute logic before agent execution starts |
| [`@before_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_model) | Execute logic before each model call |
| [`@after_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_model) | Execute logic after each model receives a response |
| [`@after_agent`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_agent) | Execute logic after agent execution completes |
| [`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) | Wrap and intercept model calls |
| [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call) | Wrap and intercept tool calls |
| [`@dynamic_prompt`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt) | Generate dynamic system prompts based on request context |
| [`@hook_config`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.hook_config) | Configure hook behavior (e.g., conditional routing) |

Types and utilities[¶](https://reference.langchain.com/python/langchain/middleware/#types-and-utilities "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------

Core types for building middleware:

| TYPE | DESCRIPTION |
| --- | --- |
| [`AgentState`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentState) | State container for agent execution |
| [`ModelRequest`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest) | Request details passed to model calls |
| [`ModelResponse`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelResponse) | Response details from model calls |
| [`ClearToolUsesEdit`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit) | Utility for clearing tool usage history from context |
| [`InterruptOnConfig`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.InterruptOnConfig) | Configuration for human-in-the-loop interruptions |

[`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.SummarizationMiddleware) types:

| TYPE | DESCRIPTION |
| --- | --- |
| [`ContextSize`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextSize) | Union type |
| [`ContextFraction`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextFraction) | Summarize at fraction of total context |
| [`ContextTokens`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextTokens) | Summarize at token threshold |
| [`ContextMessages`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextMessages) | Summarize at message threshold |

SummarizationMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.SummarizationMiddleware "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
SummarizationMiddleware(
    model: str | BaseChatModel,
    *,
    trigger: ContextSize | list[ContextSize] | None = None,
    keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
    token_counter: TokenCounter = count_tokens_approximately,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
    **deprecated_kwargs: Any,
)
```

Bases: `AgentMiddleware`

Summarizes conversation history when token limits are approached.

This middleware monitors message token counts and automatically summarizes older messages when a threshold is reached, preserving recent messages and maintaining context continuity by ensuring AI/Tool message pairs remain together.

Initialize summarization middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `model` | The language model to use for generating summaries. **TYPE:**`str | BaseChatModel` |
| `trigger` | One or more thresholds that trigger summarization. Provide a single [`ContextSize`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextSize "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code> <span class=\"doc doc-object-name doc-attribute-name\">ContextSize</span> <span class=\"doc doc-labels\"> <small class=\"doc doc-label doc-label-module-attribute\"><code>module-attribute</code></small> </span>") tuple or a list of tuples, in which case summarization runs when any threshold is met. Example ``` # Trigger summarization when 50 messages is reached ("messages", 50) # Trigger summarization when 3000 tokens is reached ("tokens", 3000) # Trigger summarization either when 80% of model's max input tokens # is reached or when 100 messages is reached (whichever comes first) [("fraction", 0.8), ("messages", 100)] ``` See [`ContextSize`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextSize "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code> <span class=\"doc doc-object-name doc-attribute-name\">ContextSize</span> <span class=\"doc doc-labels\"> <small class=\"doc doc-label doc-label-module-attribute\"><code>module-attribute</code></small> </span>") for more details. **TYPE:**`ContextSize | list[ContextSize] | None`**DEFAULT:**`None` |
| `keep` | Context retention policy applied after summarization. Provide a [`ContextSize`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextSize "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code> <span class=\"doc doc-object-name doc-attribute-name\">ContextSize</span> <span class=\"doc doc-labels\"> <small class=\"doc doc-label doc-label-module-attribute\"><code>module-attribute</code></small> </span>") tuple to specify how much history to preserve. Defaults to keeping the most recent `20` messages. Does not support multiple values like `trigger`. Example ``` # Keep the most recent 20 messages ("messages", 20) # Keep the most recent 3000 tokens ("tokens", 3000) # Keep the most recent 30% of the model's max input tokens ("fraction", 0.3) ``` **TYPE:**`ContextSize`**DEFAULT:**`('messages', _DEFAULT_MESSAGES_TO_KEEP)` |
| `token_counter` | Function to count tokens in messages. **TYPE:**`TokenCounter`**DEFAULT:**`count_tokens_approximately` |
| `summary_prompt` | Prompt template for generating summaries. **TYPE:**`str`**DEFAULT:**`DEFAULT_SUMMARY_PROMPT` |
| `trim_tokens_to_summarize` | Maximum tokens to keep when preparing messages for the summarization call. Pass `None` to skip trimming entirely. **TYPE:**`int | None`**DEFAULT:**`_DEFAULT_TRIM_TOKEN_LIMIT` |

HumanInTheLoopMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.HumanInTheLoopMiddleware "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Bases: `AgentMiddleware[StateT, ContextT]`

Human in the loop middleware.

Initialize the human in the loop middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `interrupt_on` | Mapping of tool name to allowed actions. If a tool doesn't have an entry, it's auto-approved by default. * `True` indicates all decisions are allowed: approve, edit, and reject. * `False` indicates that the tool is auto-approved. * `InterruptOnConfig` indicates the specific decisions allowed for this tool. The `InterruptOnConfig` can include a `description` field (`str` or `Callable`) for custom formatting of the interrupt description. **TYPE:**`dict[str, bool | InterruptOnConfig]` |
| `description_prefix` | The prefix to use when constructing action requests. This is used to provide context about the tool call and the action being requested. Not used if a tool has a `description` in its `InterruptOnConfig`. **TYPE:**`str`**DEFAULT:**`'Tool execution requires approval'` |

ModelCallLimitMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelCallLimitMiddleware "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
ModelCallLimitMiddleware(
    *,
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: Literal["end", "error"] = "end",
)
```

Bases: `AgentMiddleware[ModelCallLimitState, Any]`

Tracks model call counts and enforces limits.

This middleware monitors the number of model calls made during agent execution and can terminate the agent when specified limits are reached. It supports both thread-level and run-level call counting with configurable exit behaviors.

Thread-level: The middleware tracks the number of model calls and persists call count across multiple runs (invocations) of the agent.

Run-level: The middleware tracks the number of model calls made during a single run (invocation) of the agent.

Example

```
from langchain.agents.middleware.call_tracking import ModelCallLimitMiddleware
from langchain.agents import create_agent

# Create middleware with limits
call_tracker = ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end")

agent = create_agent("openai:gpt-4o", middleware=[call_tracker])

# Agent will automatically jump to end when limits are exceeded
result = await agent.invoke({"messages": [HumanMessage("Help me with a task")]})
```

Initialize the call tracking middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `thread_limit` | Maximum number of model calls allowed per thread. `None` means no limit. **TYPE:**`int | None`**DEFAULT:**`None` |
| `run_limit` | Maximum number of model calls allowed per run. `None` means no limit. **TYPE:**`int | None`**DEFAULT:**`None` |
| `exit_behavior` | What to do when limits are exceeded. * `'end'`: Jump to the end of the agent execution and inject an artificial AI message indicating that the limit was exceeded. * `'error'`: Raise a `ModelCallLimitExceededError` **TYPE:**`Literal['end', 'error']`**DEFAULT:**`'end'` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If both limits are `None` or if `exit_behavior` is invalid. |

### state_schema`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelCallLimitMiddleware.state_schema "Copy anchor link to this section for reference")

```
state_schema = ModelCallLimitState
```

The schema for state passed to the middleware nodes.

ToolCallLimitMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ToolCallLimitMiddleware "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
ToolCallLimitMiddleware(
    *,
    tool_name: str | None = None,
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: ExitBehavior = "continue",
)
```

Bases: `AgentMiddleware[ToolCallLimitState[ResponseT], ContextT]`, `Generic[ResponseT, ContextT]`

Track tool call counts and enforces limits during agent execution.

This middleware monitors the number of tool calls made and can terminate or restrict execution when limits are exceeded. It supports both thread-level (persistent across runs) and run-level (per invocation) call counting.

Configuration
*   `exit_behavior`: How to handle when limits are exceeded
    *   `'continue'`: Block exceeded tools, let execution continue (default)
    *   `'error'`: Raise an exception
    *   `'end'`: Stop immediately with a `ToolMessage` + AI message for the single tool call that exceeded the limit (raises `NotImplementedError` if there are other pending tool calls (due to parallel tool calling).

Examples:

Continue execution with blocked tools (default)

```
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain.agents import create_agent

# Block exceeded tools but let other tools and model continue
limiter = ToolCallLimitMiddleware(
    thread_limit=20,
    run_limit=10,
    exit_behavior="continue",  # default
)

agent = create_agent("openai:gpt-4o", middleware=[limiter])
```

Stop immediately when limit exceeded

```
# End execution immediately with an AI message
limiter = ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")

agent = create_agent("openai:gpt-4o", middleware=[limiter])
```

Raise exception on limit

```
# Strict limit with exception handling
limiter = ToolCallLimitMiddleware(
    tool_name="search", thread_limit=5, exit_behavior="error"
)

agent = create_agent("openai:gpt-4o", middleware=[limiter])

try:
    result = await agent.invoke({"messages": [HumanMessage("Task")]})
except ToolCallLimitExceededError as e:
    print(f"Search limit exceeded: {e}")
```

Initialize the tool call limit middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tool_name` | Name of the specific tool to limit. If `None`, limits apply to all tools. **TYPE:**`str | None`**DEFAULT:**`None` |
| `thread_limit` | Maximum number of tool calls allowed per thread. `None` means no limit. **TYPE:**`int | None`**DEFAULT:**`None` |
| `run_limit` | Maximum number of tool calls allowed per run. `None` means no limit. **TYPE:**`int | None`**DEFAULT:**`None` |
| `exit_behavior` | How to handle when limits are exceeded. * `'continue'`: Block exceeded tools with error messages, let other tools continue. Model decides when to end. * `'error'`: Raise a `ToolCallLimitExceededError` exception * `'end'`: Stop execution immediately with a `ToolMessage` + AI message for the single tool call that exceeded the limit. Raises `NotImplementedError` if there are multiple parallel tool calls to other tools or multiple pending tool calls. **TYPE:**`ExitBehavior`**DEFAULT:**`'continue'` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If both limits are `None`, if `exit_behavior` is invalid, or if `run_limit` exceeds `thread_limit`. |

### state_schema`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ToolCallLimitMiddleware.state_schema "Copy anchor link to this section for reference")

```
state_schema = ToolCallLimitState
```

The schema for state passed to the middleware nodes.

ModelFallbackMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelFallbackMiddleware "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Bases: `AgentMiddleware`

Automatic fallback to alternative models on errors.

Retries failed model calls with alternative models in sequence until success or all models exhausted. Primary model specified in `create_agent`.

Example

```
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.agents import create_agent

fallback = ModelFallbackMiddleware(
    "openai:gpt-4o-mini",  # Try first on error
    "anthropic:claude-sonnet-4-5-20250929",  # Then this
)

agent = create_agent(
    model="openai:gpt-4o",  # Primary model
    middleware=[fallback],
)

# If primary fails: tries gpt-4o-mini, then claude-sonnet-4-5-20250929
result = await agent.invoke({"messages": [HumanMessage("Hello")]})
```

Initialize model fallback middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `first_model` | First fallback model (string name or instance). **TYPE:**`str | BaseChatModel` |
| `*additional_models` | Additional fallbacks in order. **TYPE:**`str | BaseChatModel`**DEFAULT:**`()` |

PIIMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.PIIMiddleware "Copy anchor link to this section for reference")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
PIIMiddleware(
    pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,
    *,
    strategy: Literal["block", "redact", "mask", "hash"] = "redact",
    detector: Callable[[str], list[PIIMatch]] | str | None = None,
    apply_to_input: bool = True,
    apply_to_output: bool = False,
    apply_to_tool_results: bool = False,
)
```

Bases: `AgentMiddleware`

Detect and handle Personally Identifiable Information (PII) in conversations.

This middleware detects common PII types and applies configurable strategies to handle them. It can detect emails, credit cards, IP addresses, MAC addresses, and URLs in both user input and agent output.

Built-in PII types:

*   `email`: Email addresses
*   `credit_card`: Credit card numbers (validated with Luhn algorithm)
*   `ip`: IP addresses (validated with stdlib)
*   `mac_address`: MAC addresses
*   `url`: URLs (both `http`/`https` and bare URLs)

Strategies:

*   `block`: Raise an exception when PII is detected
*   `redact`: Replace PII with `[REDACTED_TYPE]` placeholders
*   `mask`: Partially mask PII (e.g., `****-****-****-1234` for credit card)
*   `hash`: Replace PII with deterministic hash (e.g., `<email_hash:a1b2c3d4>`)

Strategy Selection Guide:

| Strategy | Preserves Identity? | Best For |
| --- | --- | --- |
| `block` | N/A | Avoid PII completely |
| `redact` | No | General compliance, log sanitization |
| `mask` | No | Human readability, customer service UIs |
| `hash` | Yes (pseudonymous) | Analytics, debugging |

Example

```
from langchain.agents.middleware import PIIMiddleware
from langchain.agents import create_agent

# Redact all emails in user input
agent = create_agent(
    "openai:gpt-5",
    middleware=[
        PIIMiddleware("email", strategy="redact"),
    ],
)

# Use different strategies for different PII types
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        PIIMiddleware("credit_card", strategy="mask"),
        PIIMiddleware("url", strategy="redact"),
        PIIMiddleware("ip", strategy="hash"),
    ],
)

# Custom PII type with regex
agent = create_agent(
    "openai:gpt-5",
    middleware=[
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
    ],
)
```

Initialize the PII detection middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `pii_type` | Type of PII to detect. Can be a built-in type (`email`, `credit_card`, `ip`, `mac_address`, `url`) or a custom type name. **TYPE:**`Literal['email', 'credit_card', 'ip', 'mac_address', 'url'] | str` |
| `strategy` | How to handle detected PII. Options: * `block`: Raise `PIIDetectionError` when PII is detected * `redact`: Replace with `[REDACTED_TYPE]` placeholders * `mask`: Partially mask PII (show last few characters) * `hash`: Replace with deterministic hash (format: `<type_hash:digest>`) **TYPE:**`Literal['block', 'redact', 'mask', 'hash']`**DEFAULT:**`'redact'` |
| `detector` | Custom detector function or regex pattern. * If `Callable`: Function that takes content string and returns list of `PIIMatch` objects * If `str`: Regex pattern to match PII * If `None`: Uses built-in detector for the `pii_type` **TYPE:**`Callable[[str], list[PIIMatch]] | str | None`**DEFAULT:**`None` |
| `apply_to_input` | Whether to check user messages before model call. **TYPE:**`bool`**DEFAULT:**`True` |
| `apply_to_output` | Whether to check AI messages after model call. **TYPE:**`bool`**DEFAULT:**`False` |
| `apply_to_tool_results` | Whether to check tool result messages after tool execution. **TYPE:**`bool`**DEFAULT:**`False` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `pii_type` is not built-in and no detector is provided. |

TodoListMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.TodoListMiddleware "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
TodoListMiddleware(
    *,
    system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
    tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
)
```

Bases: `AgentMiddleware`

Middleware that provides todo list management capabilities to agents.

This middleware adds a `write_todos` tool that allows agents to create and manage structured task lists for complex multi-step operations. It's designed to help agents track progress, organize complex tasks, and provide users with visibility into task completion status.

The middleware automatically injects system prompts that guide the agent on when and how to use the todo functionality effectively.

Example

```
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents import create_agent

agent = create_agent("openai:gpt-4o", middleware=[TodoListMiddleware()])

# Agent now has access to write_todos tool and todo state tracking
result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

print(result["todos"])  # Array of todo items with status tracking
```

Initialize the `TodoListMiddleware` with optional custom prompts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `system_prompt` | Custom system prompt to guide the agent on using the todo tool. **TYPE:**`str`**DEFAULT:**`WRITE_TODOS_SYSTEM_PROMPT` |
| `tool_description` | Custom description for the `write_todos` tool. **TYPE:**`str`**DEFAULT:**`WRITE_TODOS_TOOL_DESCRIPTION` |

### state_schema`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.TodoListMiddleware.state_schema "Copy anchor link to this section for reference")

```
state_schema = PlanningState
```

The schema for state passed to the middleware nodes.

LLMToolSelectorMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.LLMToolSelectorMiddleware "Copy anchor link to this section for reference")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
LLMToolSelectorMiddleware(
    *,
    model: str | BaseChatModel | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tools: int | None = None,
    always_include: list[str] | None = None,
)
```

Bases: `AgentMiddleware`

Uses an LLM to select relevant tools before calling the main model.

When an agent has many tools available, this middleware filters them down to only the most relevant ones for the user's query. This reduces token usage and helps the main model focus on the right tools.

Examples:

Limit to 3 tools

```
from langchain.agents.middleware import LLMToolSelectorMiddleware

middleware = LLMToolSelectorMiddleware(max_tools=3)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[tool1, tool2, tool3, tool4, tool5],
    middleware=[middleware],
)
```

Use a smaller model for selection

```
middleware = LLMToolSelectorMiddleware(model="openai:gpt-4o-mini", max_tools=2)
```

Initialize the tool selector.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `model` | Model to use for selection. If not provided, uses the agent's main model. Can be a model identifier string or `BaseChatModel` instance. **TYPE:**`str | BaseChatModel | None`**DEFAULT:**`None` |
| `system_prompt` | Instructions for the selection model. **TYPE:**`str`**DEFAULT:**`DEFAULT_SYSTEM_PROMPT` |
| `max_tools` | Maximum number of tools to select. If the model selects more, only the first `max_tools` will be used. If not specified, there is no limit. **TYPE:**`int | None`**DEFAULT:**`None` |
| `always_include` | Tool names to always include regardless of selection. These do not count against the `max_tools` limit. **TYPE:**`list[str] | None`**DEFAULT:**`None` |

ToolRetryMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ToolRetryMiddleware "Copy anchor link to this section for reference")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
ToolRetryMiddleware(
    *,
    max_retries: int = 2,
    tools: list[BaseTool | str] | None = None,
    retry_on: RetryOn = (Exception,),
    on_failure: OnFailure = "continue",
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
)
```

Bases: `AgentMiddleware`

Middleware that automatically retries failed tool calls with configurable backoff.

Supports retrying on specific exceptions and exponential backoff.

Examples:

Basic usage with default settings (2 retries, exponential backoff)

```
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(model, tools=[search_tool], middleware=[ToolRetryMiddleware()])
```

Retry specific exceptions only

```
from requests.exceptions import RequestException, Timeout

retry = ToolRetryMiddleware(
    max_retries=4,
    retry_on=(RequestException, Timeout),
    backoff_factor=1.5,
)
```

Custom exception filtering

```
from requests.exceptions import HTTPError

def should_retry(exc: Exception) -> bool:
    # Only retry on 5xx errors
    if isinstance(exc, HTTPError):
        return 500 <= exc.status_code < 600
    return False

retry = ToolRetryMiddleware(
    max_retries=3,
    retry_on=should_retry,
)
```

Apply to specific tools with custom error handling

```
def format_error(exc: Exception) -> str:
    return "Database temporarily unavailable. Please try again later."

retry = ToolRetryMiddleware(
    max_retries=4,
    tools=["search_database"],
    on_failure=format_error,
)
```

Apply to specific tools using `BaseTool` instances

```
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    '''Search the database.'''
    return results

retry = ToolRetryMiddleware(
    max_retries=4,
    tools=[search_database],  # Pass BaseTool instance
)
```

Constant backoff (no exponential growth)

```
retry = ToolRetryMiddleware(
    max_retries=5,
    backoff_factor=0.0,  # No exponential growth
    initial_delay=2.0,  # Always wait 2 seconds
)
```

Raise exception on failure

```
retry = ToolRetryMiddleware(
    max_retries=2,
    on_failure="error",  # Re-raise exception instead of returning message
)
```

Initialize `ToolRetryMiddleware`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `max_retries` | Maximum number of retry attempts after the initial call. Must be `>= 0`. **TYPE:**`int`**DEFAULT:**`2` |
| `tools` | Optional list of tools or tool names to apply retry logic to. Can be a list of `BaseTool` instances or tool name strings. If `None`, applies to all tools. **TYPE:**`list[BaseTool | str] | None`**DEFAULT:**`None` |
| `retry_on` | Either a tuple of exception types to retry on, or a callable that takes an exception and returns `True` if it should be retried. Default is to retry on all exceptions. **TYPE:**`RetryOn`**DEFAULT:**`(Exception,)` |
| `on_failure` | Behavior when all retries are exhausted. Options: * `'continue'`: Return a `ToolMessage` with error details, allowing the LLM to handle the failure and potentially recover. * `'error'`: Re-raise the exception, stopping agent execution. * **Custom callable:** Function that takes the exception and returns a string for the `ToolMessage` content, allowing custom error formatting. **Deprecated values** (for backwards compatibility): * `'return_message'`: Use `'continue'` instead. * `'raise'`: Use `'error'` instead. **TYPE:**`OnFailure`**DEFAULT:**`'continue'` |
| `backoff_factor` | Multiplier for exponential backoff. Each retry waits `initial_delay * (backoff_factor ** retry_number)` seconds. Set to `0.0` for constant delay. **TYPE:**`float`**DEFAULT:**`2.0` |
| `initial_delay` | Initial delay in seconds before first retry. **TYPE:**`float`**DEFAULT:**`1.0` |
| `max_delay` | Maximum delay in seconds between retries. Caps exponential backoff growth. **TYPE:**`float`**DEFAULT:**`60.0` |
| `jitter` | Whether to add random jitter (`±25%`) to delay to avoid thundering herd. **TYPE:**`bool`**DEFAULT:**`True` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `max_retries < 0` or delays are negative. |

LLMToolEmulator[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.LLMToolEmulator "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Bases: `AgentMiddleware`

Emulates specified tools using an LLM instead of executing them.

This middleware allows selective emulation of tools for testing purposes.

By default (when `tools=None`), all tools are emulated. You can specify which tools to emulate by passing a list of tool names or `BaseTool` instances.

Examples:

Emulate all tools (default behavior)

```
from langchain.agents.middleware import LLMToolEmulator

middleware = LLMToolEmulator()

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, get_user_location, calculator],
    middleware=[middleware],
)
```

Emulate specific tools by name

```
middleware = LLMToolEmulator(tools=["get_weather", "get_user_location"])
```

Use a custom model for emulation

```
middleware = LLMToolEmulator(
    tools=["get_weather"], model="anthropic:claude-sonnet-4-5-20250929"
)
```

Emulate specific tools by passing tool instances

```
middleware = LLMToolEmulator(tools=[get_weather, get_user_location])
```

Initialize the tool emulator.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tools` | List of tool names (`str`) or `BaseTool` instances to emulate. If `None`, ALL tools will be emulated. If empty list, no tools will be emulated. **TYPE:**`list[str | BaseTool] | None`**DEFAULT:**`None` |
| `model` | Model to use for emulation. Defaults to `'anthropic:claude-sonnet-4-5-20250929'`. Can be a model identifier string or `BaseChatModel` instance. **TYPE:**`str | BaseChatModel | None`**DEFAULT:**`None` |

ContextEditingMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ContextEditingMiddleware "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
ContextEditingMiddleware(
    *,
    edits: Iterable[ContextEdit] | None = None,
    token_count_method: Literal["approximate", "model"] = "approximate",
)
```

Bases: `AgentMiddleware`

Automatically prune tool results to manage context size.

The middleware applies a sequence of edits when the total input token count exceeds configured thresholds.

Currently the `ClearToolUsesEdit` strategy is supported, aligning with Anthropic's `clear_tool_uses_20250919` behavior [(read more)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).

Initialize an instance of context editing middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `edits` | Sequence of edit strategies to apply. Defaults to a single `ClearToolUsesEdit` mirroring Anthropic defaults. **TYPE:**`Iterable[ContextEdit] | None`**DEFAULT:**`None` |
| `token_count_method` | Whether to use approximate token counting (faster, less accurate) or exact counting implemented by the chat model (potentially slower, more accurate). **TYPE:**`Literal['approximate', 'model']`**DEFAULT:**`'approximate'` |

ShellToolMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ShellToolMiddleware "Copy anchor link to this section for reference")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
ShellToolMiddleware(
    workspace_root: str | Path | None = None,
    *,
    startup_commands: tuple[str, ...] | list[str] | str | None = None,
    shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
    execution_policy: BaseExecutionPolicy | None = None,
    redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None,
    tool_description: str | None = None,
    tool_name: str = SHELL_TOOL_NAME,
    shell_command: Sequence[str] | str | None = None,
    env: Mapping[str, Any] | None = None,
)
```

Bases: `AgentMiddleware[ShellToolState, Any]`

Middleware that registers a persistent shell tool for agents.

The middleware exposes a single long-lived shell session. Use the execution policy to match your deployment's security posture:

*   `HostExecutionPolicy` – full host access; best for trusted environments where the agent already runs inside a container or VM that provides isolation.
*   `CodexSandboxExecutionPolicy` – reuses the Codex CLI sandbox for additional syscall/filesystem restrictions when the CLI is available.
*   `DockerExecutionPolicy` – launches a separate Docker container for each agent run, providing harder isolation, optional read-only root filesystems, and user remapping.

When no policy is provided the middleware defaults to `HostExecutionPolicy`.

Initialize an instance of `ShellToolMiddleware`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `workspace_root` | Base directory for the shell session. If omitted, a temporary directory is created when the agent starts and removed when it ends. **TYPE:**`str | Path | None`**DEFAULT:**`None` |
| `startup_commands` | Optional commands executed sequentially after the session starts. **TYPE:**`tuple[str, ...] | list[str] | str | None`**DEFAULT:**`None` |
| `shutdown_commands` | Optional commands executed before the session shuts down. **TYPE:**`tuple[str, ...] | list[str] | str | None`**DEFAULT:**`None` |
| `execution_policy` | Execution policy controlling timeouts, output limits, and resource configuration. Defaults to `HostExecutionPolicy` for native execution. **TYPE:**`BaseExecutionPolicy | None`**DEFAULT:**`None` |
| `redaction_rules` | Optional redaction rules to sanitize command output before returning it to the model. **TYPE:**`tuple[RedactionRule, ...] | list[RedactionRule] | None`**DEFAULT:**`None` |
| `tool_description` | Optional override for the registered shell tool description. **TYPE:**`str | None`**DEFAULT:**`None` |
| `tool_name` | Name for the registered shell tool. Defaults to `"shell"`. **TYPE:**`str`**DEFAULT:**`SHELL_TOOL_NAME` |
| `shell_command` | Optional shell executable (string) or argument sequence used to launch the persistent session. Defaults to an implementation-defined bash command. **TYPE:**`Sequence[str] | str | None`**DEFAULT:**`None` |
| `env` | Optional environment variables to supply to the shell session. Values are coerced to strings before command execution. If omitted, the session inherits the parent process environment. **TYPE:**`Mapping[str, Any] | None`**DEFAULT:**`None` |

FilesystemFileSearchMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.FilesystemFileSearchMiddleware "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
FilesystemFileSearchMiddleware(
    *, root_path: str, use_ripgrep: bool = True, max_file_size_mb: int = 10
)
```

Bases: `AgentMiddleware`

Provides Glob and Grep search over filesystem files.

This middleware adds two tools that search through local filesystem:

*   Glob: Fast file pattern matching by file path
*   Grep: Fast content search using ripgrep or Python fallback

Example

```
from langchain.agents import create_agent
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
)

agent = create_agent(
    model=model,
    tools=[],  # Add tools as needed
    middleware=[
        FilesystemFileSearchMiddleware(root_path="/workspace"),
    ],
)
```

Initialize the search middleware.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `root_path` | Root directory to search. **TYPE:**`str` |
| `use_ripgrep` | Whether to use `ripgrep` for search. Falls back to Python if `ripgrep` unavailable. **TYPE:**`bool`**DEFAULT:**`True` |
| `max_file_size_mb` | Maximum file size to search in MB. **TYPE:**`int`**DEFAULT:**`10` |

AgentMiddleware[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Bases: `Generic[StateT, ContextT]`

Base middleware class for an agent.

Subclass this and implement any of the defined methods to customize agent behavior between steps in the main agent loop.

before_agent[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_agent "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableWithStateAndRuntime[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
)
```

Decorator used to dynamically create a middleware with the `before_agent` hook.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | The function to be decorated. Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime context **TYPE:**`_CallableWithStateAndRuntime[StateT, ContextT] | None`**DEFAULT:**`None` |
| `state_schema` | Optional custom state schema type. If not provided, uses the default `AgentState` schema. **TYPE:**`type[StateT] | None`**DEFAULT:**`None` |
| `tools` | Optional list of additional tools to register with this middleware. **TYPE:**`list[BaseTool] | None`**DEFAULT:**`None` |
| `can_jump_to` | Optional list of valid jump destinations for conditional edges. Valid values are: `'tools'`, `'model'`, `'end'` **TYPE:**`list[JumpTo] | None`**DEFAULT:**`None` |
| `name` | Optional name for the generated middleware class. If not provided, uses the decorated function's name. **TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]] | AgentMiddleware[StateT, ContextT]` | Either an `AgentMiddleware` instance (if func is provided directly) or a decorator function that can be applied to a function it is wrapping. |

The decorated function should return:

*   `dict[str, Any]` - State updates to merge into the agent state
*   `Command` - A command to control flow (e.g., jump to different node)
*   `None` - No state updates or flow control

Examples:

Basic usage

```
@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"Starting agent with {len(state['messages'])} messages")
```

With conditional jumping

```
@before_agent(can_jump_to=["end"])
def conditional_before_agent(
    state: AgentState, runtime: Runtime
) -> dict[str, Any] | None:
    if some_condition(state):
        return {"jump_to": "end"}
    return None
```

With custom state schema

```
@before_agent(state_schema=MyCustomState)
def custom_before_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
    return {"custom_field": "initialized_value"}
```

Streaming custom events

Use `runtime.stream_writer` to emit custom events during agent execution. Events are received when streaming with `stream_mode="custom"`.

```
from langchain.agents import create_agent
from langchain.agents.middleware import before_agent, AgentState
from langchain.messages import HumanMessage
from langgraph.runtime import Runtime

@before_agent
async def notify_start(state: AgentState, runtime: Runtime) -> None:
    '''Notify user that agent is starting.'''
    runtime.stream_writer(
        {
            "type": "status",
            "message": "Initializing agent session...",
        }
    )
    # Perform prerequisite tasks here
    runtime.stream_writer({"type": "status", "message": "Agent ready!"})

agent = create_agent(
    model="openai:gpt-5.2",
    tools=[...],
    middleware=[notify_start],
)

# Consume with stream_mode="custom" to receive events
async for mode, event in agent.astream(
    {"messages": [HumanMessage("Hello")]},
    stream_mode=["updates", "custom"],
):
    if mode == "custom":
        print(f"Status: {event}")
```

before_model[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_model "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableWithStateAndRuntime[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
)
```

Decorator used to dynamically create a middleware with the `before_model` hook.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | The function to be decorated. Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime context **TYPE:**`_CallableWithStateAndRuntime[StateT, ContextT] | None`**DEFAULT:**`None` |
| `state_schema` | Optional custom state schema type. If not provided, uses the default `AgentState` schema. **TYPE:**`type[StateT] | None`**DEFAULT:**`None` |
| `tools` | Optional list of additional tools to register with this middleware. **TYPE:**`list[BaseTool] | None`**DEFAULT:**`None` |
| `can_jump_to` | Optional list of valid jump destinations for conditional edges. Valid values are: `'tools'`, `'model'`, `'end'` **TYPE:**`list[JumpTo] | None`**DEFAULT:**`None` |
| `name` | Optional name for the generated middleware class. If not provided, uses the decorated function's name. **TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]] | AgentMiddleware[StateT, ContextT]` | Either an `AgentMiddleware` instance (if func is provided directly) or a decorator function that can be applied to a function it is wrapping. |

The decorated function should return:

*   `dict[str, Any]` - State updates to merge into the agent state
*   `Command` - A command to control flow (e.g., jump to different node)
*   `None` - No state updates or flow control

Examples:

Basic usage

```
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    print(f"About to call model with {len(state['messages'])} messages")
```

With conditional jumping

```
@before_model(can_jump_to=["end"])
def conditional_before_model(
    state: AgentState, runtime: Runtime
) -> dict[str, Any] | None:
    if some_condition(state):
        return {"jump_to": "end"}
    return None
```

With custom state schema

```
@before_model(state_schema=MyCustomState)
def custom_before_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
    return {"custom_field": "updated_value"}
```

Streaming custom events before model call

Use `runtime.stream_writer` to emit custom events before each model invocation. Events are received when streaming with `stream_mode="custom"`.

```
@before_model
async def notify_model_call(state: AgentState, runtime: Runtime) -> None:
    '''Notify user before model is called.'''
    runtime.stream_writer(
        {
            "type": "status",
            "message": "Thinking...",
        }
    )
```

after_model[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_model "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableWithStateAndRuntime[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
)
```

Decorator used to dynamically create a middleware with the `after_model` hook.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | The function to be decorated. Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime context **TYPE:**`_CallableWithStateAndRuntime[StateT, ContextT] | None`**DEFAULT:**`None` |
| `state_schema` | Optional custom state schema type. If not provided, uses the default `AgentState` schema. **TYPE:**`type[StateT] | None`**DEFAULT:**`None` |
| `tools` | Optional list of additional tools to register with this middleware. **TYPE:**`list[BaseTool] | None`**DEFAULT:**`None` |
| `can_jump_to` | Optional list of valid jump destinations for conditional edges. Valid values are: `'tools'`, `'model'`, `'end'` **TYPE:**`list[JumpTo] | None`**DEFAULT:**`None` |
| `name` | Optional name for the generated middleware class. If not provided, uses the decorated function's name. **TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]] | AgentMiddleware[StateT, ContextT]` | Either an `AgentMiddleware` instance (if func is provided) or a decorator function that can be applied to a function. |

The decorated function should return:

*   `dict[str, Any]` - State updates to merge into the agent state
*   `Command` - A command to control flow (e.g., jump to different node)
*   `None` - No state updates or flow control

Examples:

Basic usage for logging model responses

```
@after_model
def log_latest_message(state: AgentState, runtime: Runtime) -> None:
    print(state["messages"][-1].content)
```

With custom state schema

```
@after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
def custom_after_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
    return {"custom_field": "updated_after_model"}
```

Streaming custom events after model call

Use `runtime.stream_writer` to emit custom events after model responds. Events are received when streaming with `stream_mode="custom"`.

```
@after_model
async def notify_model_response(state: AgentState, runtime: Runtime) -> None:
    '''Notify user after model has responded.'''
    last_message = state["messages"][-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    runtime.stream_writer(
        {
            "type": "status",
            "message": "Using tools..." if has_tool_calls else "Response ready!",
        }
    )
```

after_agent[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_agent "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableWithStateAndRuntime[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
)
```

Decorator used to dynamically create a middleware with the `after_agent` hook.

Async version is `aafter_agent`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | The function to be decorated. Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime context **TYPE:**`_CallableWithStateAndRuntime[StateT, ContextT] | None`**DEFAULT:**`None` |
| `state_schema` | Optional custom state schema type. If not provided, uses the default `AgentState` schema. **TYPE:**`type[StateT] | None`**DEFAULT:**`None` |
| `tools` | Optional list of additional tools to register with this middleware. **TYPE:**`list[BaseTool] | None`**DEFAULT:**`None` |
| `can_jump_to` | Optional list of valid jump destinations for conditional edges. Valid values are: `'tools'`, `'model'`, `'end'` **TYPE:**`list[JumpTo] | None`**DEFAULT:**`None` |
| `name` | Optional name for the generated middleware class. If not provided, uses the decorated function's name. **TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]] | AgentMiddleware[StateT, ContextT]` | Either an `AgentMiddleware` instance (if func is provided) or a decorator function that can be applied to a function. |

The decorated function should return:

*   `dict[str, Any]` - State updates to merge into the agent state
*   `Command` - A command to control flow (e.g., jump to different node)
*   `None` - No state updates or flow control

Examples:

Basic usage for logging agent completion

```
@after_agent
def log_completion(state: AgentState, runtime: Runtime) -> None:
    print(f"Agent completed with {len(state['messages'])} messages")
```

With custom state schema

```
@after_agent(state_schema=MyCustomState, name="MyAfterAgentMiddleware")
def custom_after_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
    return {"custom_field": "finalized_value"}
```

Streaming custom events on completion

Use `runtime.stream_writer` to emit custom events when agent completes. Events are received when streaming with `stream_mode="custom"`.

```
@after_agent
async def notify_completion(state: AgentState, runtime: Runtime) -> None:
    '''Notify user that agent has completed.'''
    runtime.stream_writer(
        {
            "type": "status",
            "message": "Agent execution complete!",
            "total_messages": len(state["messages"]),
        }
    )
```

wrap_model_call[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
wrap_model_call(
    func: _CallableReturningModelResponse[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningModelResponse[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
)
```

Create middleware with `wrap_model_call` hook from a function.

Converts a function with handler callback into middleware that can intercept model calls, implement retry logic, handle errors, and rewrite responses.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | Function accepting (request, handler) that calls handler(request) to execute the model and returns `ModelResponse` or `AIMessage`. Request contains state and runtime. **TYPE:**`_CallableReturningModelResponse[StateT, ContextT] | None`**DEFAULT:**`None` |
| `state_schema` | Custom state schema. Defaults to `AgentState`. **TYPE:**`type[StateT] | None`**DEFAULT:**`None` |
| `tools` | Additional tools to register with this middleware. **TYPE:**`list[BaseTool] | None`**DEFAULT:**`None` |
| `name` | Middleware class name. Defaults to function name. **TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[_CallableReturningModelResponse[StateT, ContextT]], AgentMiddleware[StateT, ContextT]] | AgentMiddleware[StateT, ContextT]` | `AgentMiddleware` instance if func provided, otherwise a decorator. |

Examples:

Basic retry logic

```
@wrap_model_call
def retry_on_error(request, handler):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return handler(request)
        except Exception:
            if attempt == max_retries - 1:
                raise
```

Model fallback

```
@wrap_model_call
def fallback_model(request, handler):
    # Try primary model
    try:
        return handler(request)
    except Exception:
        pass

    # Try fallback model
    request = request.override(model=fallback_model_instance)
    return handler(request)
```

Rewrite response content (full `ModelResponse`)

```
@wrap_model_call
def uppercase_responses(request, handler):
    response = handler(request)
    ai_msg = response.result[0]
    return ModelResponse(
        result=[AIMessage(content=ai_msg.content.upper())],
        structured_response=response.structured_response,
    )
```

Simple `AIMessage` return (converted automatically)

```
@wrap_model_call
def simple_response(request, handler):
    # AIMessage is automatically converted to ModelResponse
    return AIMessage(content="Simple response")
```

wrap_tool_call[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Create middleware with `wrap_tool_call` hook from a function.

Async version is `awrap_tool_call`.

Converts a function with handler callback into middleware that can intercept tool calls, implement retry logic, monitor execution, and modify responses.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | Function accepting (request, handler) that calls handler(request) to execute the tool and returns final `ToolMessage` or `Command`. Can be sync or async. **TYPE:**`_CallableReturningToolResponse | None`**DEFAULT:**`None` |
| `tools` | Additional tools to register with this middleware. **TYPE:**`list[BaseTool] | None`**DEFAULT:**`None` |
| `name` | Middleware class name. Defaults to function name. **TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[_CallableReturningToolResponse], AgentMiddleware] | AgentMiddleware` | `AgentMiddleware` instance if func provided, otherwise a decorator. |

Examples:

Retry logic

```
@wrap_tool_call
def retry_on_error(request, handler):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return handler(request)
        except Exception:
            if attempt == max_retries - 1:
                raise
```

Async retry logic

```
@wrap_tool_call
async def async_retry(request, handler):
    for attempt in range(3):
        try:
            return await handler(request)
        except Exception:
            if attempt == 2:
                raise
```

Modify request

```
@wrap_tool_call
def modify_args(request, handler):
    modified_call = {
        **request.tool_call,
        "args": {
            **request.tool_call["args"],
            "value": request.tool_call["args"]["value"] * 2,
        },
    }
    request = request.override(tool_call=modified_call)
    return handler(request)
```

Short-circuit with cached result

```
@wrap_tool_call
def with_cache(request, handler):
    if cached := get_cache(request):
        return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
    result = handler(request)
    save_cache(request, result)
    return result
```

dynamic_prompt[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
dynamic_prompt(
    func: _CallableReturningSystemMessage[StateT, ContextT] | None = None,
) -> (
    Callable[
        [_CallableReturningSystemMessage[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
)
```

Decorator used to dynamically generate system prompts for the model.

This is a convenience decorator that creates middleware using `wrap_model_call` specifically for dynamic prompt generation. The decorated function should return a string that will be set as the system prompt for the model request.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | The function to be decorated. Must accept: `request: ModelRequest` - Model request (contains state and runtime) **TYPE:**`_CallableReturningSystemMessage[StateT, ContextT] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[_CallableReturningSystemMessage[StateT, ContextT]], AgentMiddleware[StateT, ContextT]] | AgentMiddleware[StateT, ContextT]` | Either an `AgentMiddleware` instance (if func is provided) or a decorator function that can be applied to a function. |

The decorated function should return
*   `str` – The system prompt string to use for the model request
*   `SystemMessage` – A complete system message to use for the model request

Examples:

Basic usage with dynamic content:

```
@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.get("user_name", "User")
    return f"You are a helpful assistant helping {user_name}."
```

Using state to customize the prompt:

```
@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    msg_count = len(request.state["messages"])
    if msg_count > 10:
        return "You are in a long conversation. Be concise."
    return "You are a helpful assistant."
```

Using with agent:

```
agent = create_agent(model, middleware=[my_prompt])
```

hook_config[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.hook_config "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
hook_config(
    *, can_jump_to: list[JumpTo] | None = None
) -> Callable[[CallableT], CallableT]
```

Decorator to configure hook behavior in middleware methods.

Use this decorator on `before_model` or `after_model` methods in middleware classes to configure their behavior. Currently supports specifying which destinations they can jump to, which establishes conditional edges in the agent graph.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `can_jump_to` | Optional list of valid jump destinations. Can be: * `'tools'`: Jump to the tools node * `'model'`: Jump back to the model node * `'end'`: Jump to the end of the graph **TYPE:**`list[JumpTo] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Callable[[CallableT], CallableT]` | Decorator function that marks the method with configuration metadata. |

Examples:

Using decorator on a class method

```
class MyMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end", "model"])
    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        if some_condition(state):
            return {"jump_to": "end"}
        return None
```

Alternative: Use the `can_jump_to` parameter in `before_model`/`after_model` decorators:

```
@before_model(can_jump_to=["end"])
def conditional_middleware(state: AgentState) -> dict[str, Any] | None:
    if should_exit(state):
        return {"jump_to": "end"}
    return None
```

ModelRequest`dataclass`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest "Copy anchor link to this section for reference")
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
ModelRequest(
    *,
    model: BaseChatModel,
    messages: list[AnyMessage],
    system_message: SystemMessage | None = None,
    system_prompt: str | None = None,
    tool_choice: Any | None = None,
    tools: list[BaseTool | dict] | None = None,
    response_format: ResponseFormat | None = None,
    state: AgentState | None = None,
    runtime: Runtime[ContextT] | None = None,
    model_settings: dict[str, Any] | None = None,
)
```

Model request information for the agent.

Initialize ModelRequest with backward compatibility for system_prompt.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `model` | The chat model to use. **TYPE:**`BaseChatModel` |
| `messages` | List of messages (excluding system prompt). **TYPE:**`list[AnyMessage]` |
| `tool_choice` | Tool choice configuration. **TYPE:**`Any | None`**DEFAULT:**`None` |
| `tools` | List of available tools. **TYPE:**`list[BaseTool | dict] | None`**DEFAULT:**`None` |
| `response_format` | Response format specification. **TYPE:**`ResponseFormat | None`**DEFAULT:**`None` |
| `state` | Agent state. **TYPE:**`AgentState | None`**DEFAULT:**`None` |
| `runtime` | Runtime context. **TYPE:**`Runtime[ContextT] | None`**DEFAULT:**`None` |
| `model_settings` | Additional model settings. **TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `system_message` | System message instance (preferred). **TYPE:**`SystemMessage | None`**DEFAULT:**`None` |
| `system_prompt` | System prompt string (deprecated, converted to SystemMessage). **TYPE:**`str | None`**DEFAULT:**`None` |

| METHOD | DESCRIPTION |
| --- | --- |
| `__setattr__` | Set an attribute with a deprecation warning. |
| `override` | Replace the request with a new request with the given overrides. |

### system_prompt`property`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest.system_prompt "Copy anchor link to this section for reference")

```
system_prompt: str | None
```

Get system prompt text from system_message.

| RETURNS | DESCRIPTION |
| --- | --- |
| `str | None` | The content of the system message if present, otherwise `None`. |

### __setattr__ [¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest.__setattr__ "Copy anchor link to this section for reference")

```
__setattr__(name: str, value: Any) -> None
```

Set an attribute with a deprecation warning.

Direct attribute assignment on `ModelRequest` is deprecated. Use the `override()` method instead to create a new request with modified attributes.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | Attribute name. **TYPE:**`str` |
| `value` | Attribute value. **TYPE:**`Any` |

### override[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest.override "Copy anchor link to this section for reference")

Replace the request with a new request with the given overrides.

Returns a new `ModelRequest` instance with the specified attributes replaced.

This follows an immutable pattern, leaving the original request unchanged.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**overrides` | Keyword arguments for attributes to override. Supported keys: * `model`: `BaseChatModel` instance * `system_prompt`: deprecated, use `system_message` instead * `system_message`: `SystemMessage` instance * `messages`: `list` of messages * `tool_choice`: Tool choice configuration * `tools`: `list` of available tools * `response_format`: Response format specification * `model_settings`: Additional model settings **TYPE:**`Unpack[_ModelRequestOverrides]`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `ModelRequest` | New `ModelRequest` instance with specified overrides applied. |

Examples:

Create a new request with different model

```
new_request = request.override(model=different_model)
```

Override system message (preferred)

```
from langchain_core.messages import SystemMessage

new_request = request.override(
    system_message=SystemMessage(content="New instructions")
)
```

Override multiple attributes

```
new_request = request.override(
    model=ChatOpenAI(model="gpt-4o"),
    system_message=SystemMessage(content="New instructions"),
)
```

ModelResponse`dataclass`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelResponse "Copy anchor link to this section for reference")
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Response from model execution including messages and optional structured output.

The result will usually contain a single `AIMessage`, but may include an additional `ToolMessage` if the model used a tool for structured output.

### result`instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelResponse.result "Copy anchor link to this section for reference")

List of messages from model execution.

### structured_response`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelResponse.structured_response "Copy anchor link to this section for reference")

```
structured_response: Any = None
```

Parsed structured output if `response_format` was specified, `None` otherwise.

ClearToolUsesEdit`dataclass`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit "Copy anchor link to this section for reference")
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
ClearToolUsesEdit(
    trigger: int = 100000,
    clear_at_least: int = 0,
    keep: int = 3,
    clear_tool_inputs: bool = False,
    exclude_tools: Sequence[str] = (),
    placeholder: str = DEFAULT_TOOL_PLACEHOLDER,
)
```

Bases: `ContextEdit`

Configuration for clearing tool outputs when token limits are exceeded.

| METHOD | DESCRIPTION |
| --- | --- |
| `apply` | Apply the clear-tool-uses strategy. |

### trigger`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit.trigger "Copy anchor link to this section for reference")

Token count that triggers the edit.

### clear_at_least`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit.clear_at_least "Copy anchor link to this section for reference")

Minimum number of tokens to reclaim when the edit runs.

### keep`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit.keep "Copy anchor link to this section for reference")

Number of most recent tool results that must be preserved.

### clear_tool_inputs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit.clear_tool_inputs "Copy anchor link to this section for reference")

```
clear_tool_inputs: bool = False
```

Whether to clear the originating tool call parameters on the AI message.

### exclude_tools`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit.exclude_tools "Copy anchor link to this section for reference")

List of tool names to exclude from clearing.

### placeholder`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit.placeholder "Copy anchor link to this section for reference")

```
placeholder: str = DEFAULT_TOOL_PLACEHOLDER
```

Placeholder text inserted for cleared tool outputs.

### apply[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit.apply "Copy anchor link to this section for reference")

```
apply(messages: list[AnyMessage], *, count_tokens: TokenCounter) -> None
```

Apply the clear-tool-uses strategy.

InterruptOnConfig[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.InterruptOnConfig "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Bases: `TypedDict`

Configuration for an action requiring human in the loop.

This is the configuration format used in the `HumanInTheLoopMiddleware.__init__` method.

### allowed_decisions`instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.InterruptOnConfig.allowed_decisions "Copy anchor link to this section for reference")

```
allowed_decisions: list[DecisionType]
```

The decisions that are allowed for this action.

### description`instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.InterruptOnConfig.description "Copy anchor link to this section for reference")

The description attached to the request for human input.

Can be either:

*   A static string describing the approval request
*   A callable that dynamically generates the description based on agent state, runtime, and tool call information

Example

```
# Static string description
config = ToolConfig(
    allowed_decisions=["approve", "reject"],
    description="Please review this tool execution"
)

# Dynamic callable description
def format_tool_description(
    tool_call: ToolCall,
    state: AgentState,
    runtime: Runtime[ContextT]
) -> str:
    import json
    return (
        f"Tool: {tool_call['name']}\n"
        f"Arguments:\n{json.dumps(tool_call['args'], indent=2)}"
    )

config = InterruptOnConfig(
    allowed_decisions=["approve", "edit", "reject"],
    description=format_tool_description
)
```

### args_schema`instance-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.InterruptOnConfig.args_schema "Copy anchor link to this section for reference")

JSON schema for the args associated with the action, if edits are allowed.

ContextSize`module-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextSize "Copy anchor link to this section for reference")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Union type for context size specifications.

Can be either:

*   [`ContextFraction`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextFraction "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code>            <span class=\"doc doc-object-name doc-attribute-name\">ContextFraction</span>

  <span class=\"doc doc-labels\">
      <small class=\"doc doc-label doc-label-module-attribute\"><code>module-attribute</code></small>
  </span>"): A fraction of the model's maximum input tokens.
*   [`ContextTokens`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextTokens "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code>            <span class=\"doc doc-object-name doc-attribute-name\">ContextTokens</span>

  <span class=\"doc doc-labels\">
      <small class=\"doc doc-label doc-label-module-attribute\"><code>module-attribute</code></small>
  </span>"): An absolute number of tokens.
*   [`ContextMessages`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextMessages "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code>            <span class=\"doc doc-object-name doc-attribute-name\">ContextMessages</span>

  <span class=\"doc doc-labels\">
      <small class=\"doc doc-label doc-label-module-attribute\"><code>module-attribute</code></small>
  </span>"): An absolute number of messages.

Depending on use with `trigger` or `keep` parameters, this type indicates either when to trigger summarization or how much context to retain.

Example

```
# ContextFraction
context_size: ContextSize = ("fraction", 0.5)

# ContextTokens
context_size: ContextSize = ("tokens", 3000)

# ContextMessages
context_size: ContextSize = ("messages", 50)
```

ContextFraction`module-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextFraction "Copy anchor link to this section for reference")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Fraction of model's maximum input tokens.

Example
To specify 50% of the model's max input tokens:

```
("fraction", 0.5)
```

ContextTokens`module-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextTokens "Copy anchor link to this section for reference")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Absolute number of tokens.

Example
To specify 3000 tokens:

```
("tokens", 3000)
```

ContextMessages`module-attribute`[¶](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.summarization.ContextMessages "Copy anchor link to this section for reference")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Absolute number of messages.

Example
To specify 50 messages:

```
("messages", 50)
```