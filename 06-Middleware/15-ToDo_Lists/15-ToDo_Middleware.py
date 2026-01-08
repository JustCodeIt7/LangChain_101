"""
# LangChain Todo Middleware Examples

This file contains two examples of custom middleware for LangChain todo lists:
1. A simple logging middleware that logs all requests and responses
2. A validation middleware that validates input and output
"""

from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig
import logging
from datetime import datetime

# For LangChain v0.1+ compatibility
try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:
    # Fallback for newer LangChain versions
    class AgentMiddleware:
        """Base middleware class"""
        async def adispatch(self, next_ai, agent, agent_input):
            return await next_ai.adispatch(agent, agent_input)

        async def aprocess_agent_action(self, action, color=None):
            return action

        async def aprocess_agent_finish(self, response):
            return response

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoggingTodoMiddleware(AgentMiddleware):
    """
    A simple middleware that logs all requests and responses for todo operations.

    This middleware demonstrates how to intercept and log agent actions and finishes
    related to todo list management, providing visibility into the agent's task planning.

    Example usage:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware.todo import TodoListMiddleware

        agent = create_agent(
            model="gpt-4o",
            tools=[read_file, write_file],
            middleware=[
                TodoListMiddleware(),
                LoggingTodoMiddleware()  # Add logging middleware
            ],
        )
        ```
    """

    async def adispatch(
        self,
        next_ai,
        agent,
        agent_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dispatch method that logs the input and output of the agent.

        Args:
            next_ai: The next middleware or agent in the chain
            agent: The agent instance
            agent_input: The input to the agent

        Returns:
            The agent's output with logging information added
        """
        # Log the incoming request
        logger.info("=== TODO AGENT REQUEST START ===")
        logger.info(f"Input: {agent_input}")
        logger.info(f"Agent tools: {[tool.name for tool in agent.tools]}")

        # Call the next middleware/agent
        result = await next_ai.adispatch(agent, agent_input)

        # Log the outgoing response
        logger.info("=== TODO AGENT RESPONSE ===")
        logger.info(f"Response keys: {list(result.keys())}")
        if "todos" in result:
            logger.info(f"Todos: {result['todos']}")
        if "output" in result:
            logger.info(f"Output: {result['output']}")

        logger.info("=== TODO AGENT REQUEST END ===")

        return result

    async def aprocess_agent_action(
        self,
        action: Any,
        color: Optional[str] = None,
    ) -> Any:
        """
        Process agent actions and log todo-related operations.

        Args:
            action: The agent action to process
            color: Optional color for output formatting

        Returns:
            The processed action
        """
        logger.info(f"Agent action: {action.tool}")
        logger.info(f"Action input: {action.tool_input}")

        # Check if this is a todo-related action
        if "todo" in action.tool.lower() or "write_todos" in action.tool.lower():
            logger.info("📝 Detected todo-related action")
            logger.info(f"Todo operation: {action.tool_input}")

        return await super().aprocess_agent_action(action, color)

    async def aprocess_agent_finish(
        self,
        response: Any,
    ) -> Any:
        """
        Process agent completion and log final todo state.

        Args:
            response: The agent finish response

        Returns:
            The processed response
        """
        logger.info("Agent finished execution")
        logger.info(f"Return values: {response.return_values}")

        # Log todos if present in return values
        if "todos" in response.return_values:
            todos = response.return_values["todos"]
            logger.info(f"Final todo list ({len(todos)} items):")
            for i, todo in enumerate(todos, 1):
                status = todo.get("status", "unknown")
                description = todo.get("description", "No description")
                logger.info(f"  {i}. [{status}] {description}")

        return await super().aprocess_agent_finish(response)

class ValidationTodoMiddleware(AgentMiddleware):
    """
    A middleware that validates the input and output of todo operations.

    This middleware ensures that:
    - Todo items have required fields
    - Todo status is valid
    - Todo descriptions are not empty
    - Output contains properly formatted todo list

    Example usage:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware.todo import TodoListMiddleware

        agent = create_agent(
            model="gpt-4o",
            tools=[read_file, write_file],
            middleware=[
                TodoListMiddleware(),
                ValidationTodoMiddleware()  # Add validation middleware
            ],
        )
        ```
    """

    def __init__(self):
        """Initialize the validation middleware."""
        super().__init__()
        self.valid_statuses = {"pending", "in_progress", "completed", "cancelled"}

    async def adispatch(
        self,
        next_ai,
        agent,
        agent_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dispatch method that validates input and output.

        Args:
            next_ai: The next middleware or agent in the chain
            agent: The agent instance
            agent_input: The input to the agent

        Returns:
            The validated agent output

        Raises:
            ValueError: If validation fails
        """
        # Validate input if it contains todo-related content
        self._validate_input(agent_input)

        # Call the next middleware/agent
        result = await next_ai.adispatch(agent, agent_input)

        # Validate output
        self._validate_output(result)

        return result

    def _validate_input(self, agent_input: Dict[str, Any]) -> None:
        """
        Validate the input to the agent.

        Args:
            agent_input: The input to validate

        Raises:
            ValueError: If input validation fails
        """
        # Check if input contains todo-related content
        if "messages" in agent_input:
            messages = agent_input["messages"]
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    # Basic validation - check for empty content
                    if not content.strip():
                        raise ValueError("Todo input cannot be empty")

    def _validate_output(self, output: Dict[str, Any]) -> None:
        """
        Validate the output from the agent.

        Args:
            output: The output to validate

        Raises:
            ValueError: If output validation fails
        """
        if "todos" in output:
            todos = output["todos"]
            if not isinstance(todos, list):
                raise ValueError("Todos must be a list")

            for i, todo in enumerate(todos):
                if not isinstance(todo, dict):
                    raise ValueError(f"Todo item {i} must be a dictionary")

                # Validate required fields
                if "description" not in todo:
                    raise ValueError(f"Todo item {i} missing 'description' field")
                if not todo["description"].strip():
                    raise ValueError(f"Todo item {i} description cannot be empty")

                # Validate status if present
                if "status" in todo:
                    status = todo["status"]
                    if status not in self.valid_statuses:
                        raise ValueError(
                            f"Todo item {i} has invalid status '{status}'. "
                            f"Valid statuses: {self.valid_statuses}"
                        )

                # Validate optional fields
                if "priority" in todo:
                    priority = todo["priority"]
                    if priority not in {1, 2, 3, 4, 5}:
                        raise ValueError(
                            f"Todo item {i} has invalid priority '{priority}'. "
                            "Priority must be between 1 and 5"
                        )

    async def aprocess_agent_action(
        self,
        action: Any,
        color: Optional[str] = None,
    ) -> Any:
        """
        Process agent actions and validate todo operations.

        Args:
            action: The agent action to process
            color: Optional color for output formatting

        Returns:
            The processed action
        """
        # Validate action input if it's a todo operation
        if hasattr(action, 'tool') and ("todo" in action.tool.lower() or "write_todos" in action.tool.lower()):
            if not action.tool_input:
                raise ValueError("Todo action input cannot be empty")

            # Parse the todo input and validate structure
            todo_input = action.tool_input
            if isinstance(todo_input, dict):
                if "todos" in todo_input:
                    for i, todo in enumerate(todo_input["todos"]):
                        if not isinstance(todo, dict):
                            raise ValueError(f"Todo item {i} must be a dictionary")
                        if "description" not in todo:
                            raise ValueError(f"Todo item {i} missing 'description'")

        return await super().aprocess_agent_action(action, color)

    async def aprocess_agent_finish(
        self,
        response: Any,
    ) -> Any:
        """
        Process agent completion and validate final todo state.

        Args:
            response: The agent finish response

        Returns:
            The processed response
        """
        # Validate the final todo list
        if hasattr(response, 'return_values') and "todos" in response.return_values:
            todos = response.return_values["todos"]
            if not isinstance(todos, list):
                raise ValueError("Todos must be a list")

            # Check for empty todo list
            if len(todos) == 0:
                logger.warning("Warning: Todo list is empty")

            # Validate each todo item
            for i, todo in enumerate(todos):
                if not isinstance(todo, dict):
                    raise ValueError(f"Todo item {i} must be a dictionary")

                # Ensure description exists and is not empty
                description = todo.get("description", "")
                if not description.strip():
                    raise ValueError(f"Todo item {i} description cannot be empty")

                # Set default status if not provided
                if "status" not in todo:
                    todo["status"] = "pending"
                    logger.info(f"Set default status 'pending' for todo item {i}")

                # Ensure status is valid
                status = todo["status"]
                if status not in self.valid_statuses:
                    raise ValueError(
                        f"Todo item {i} has invalid status '{status}'. "
                        f"Valid statuses: {self.valid_statuses}"
                    )

        return await super().aprocess_agent_finish(response)

# Example usage
if __name__ == "__main__":
    print("Example middleware classes created successfully!")
    print("")
    print("Usage instructions:")
    print("1. Import the middleware classes: LoggingTodoMiddleware, ValidationTodoMiddleware")
    print("2. Add them to your agent's middleware list along with TodoListMiddleware")
    print("3. The middleware will automatically log and validate todo operations")
    print("")
    print("Example:")
    print("""
    from langchain.agents import create_agent
    from langchain.agents.middleware.todo import TodoListMiddleware

    agent = create_agent(
        model="gpt-4o",
        tools=[read_file, write_file],
        middleware=[
            TodoListMiddleware(),
            LoggingTodoMiddleware(),  # Logs all requests/responses
            ValidationTodoMiddleware(),  # Validates input/output
        ],
    )
    """)