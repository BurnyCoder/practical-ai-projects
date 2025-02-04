# Import required libraries
import asyncio
from dataclasses import dataclass
from typing import List

# Import core AutoGen components for agent functionality
from autogen_core import (
    AgentId,  # Used to identify agents
    MessageContext,  # Context for message handling
    RoutedAgent,  # Base class for routed agents
    SingleThreadedAgentRuntime,  # Runtime for single-threaded agent execution
    message_handler,  # Decorator for message handlers
)

# Import model-related components
from autogen_core.models import (
    ChatCompletionClient,  # Base class for chat completion clients
    LLMMessage,  # Base class for LLM messages
    SystemMessage,  # System message type
    UserMessage,  # User message type
)

# Import tool-related components
from autogen_core.tool_agent import ToolAgent, tool_agent_caller_loop
from autogen_core.tools import FunctionTool, Tool, ToolSchema
from autogen_ext.models.openai import OpenAIChatCompletionClient

import random

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Example tool function to get stock price
async def get_stock_price(ticker: str, date: Annotated[str, "Date in YYYY/MM/DD"]) -> float:
    # Returns a random stock price for demonstration purposes.
    return random.uniform(10, 200)

# Create a function tool for stock price lookup
stock_price_tool = FunctionTool(get_stock_price, description="Get the stock price.")

# Simple message dataclass
@dataclass
class Message:
    content: str

# Custom agent class that can use tools
class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[ToolSchema], tool_agent_type: str) -> None:
        # Initialize the agent with a description
        super().__init__("An agent with tools")
        # Set up system messages for the agent
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        # Store the model client for LLM interactions
        self._model_client = model_client
        # Store tool schemas
        self._tool_schema = tool_schema
        # Create tool agent ID
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages including system and user messages
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]
        
        # Run the tool caller loop to handle any tool calls
        messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=session,
            tool_schema=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        
        # Extract and return the final response
        assert isinstance(messages[-1].content, str)
        return Message(content=messages[-1].content)
    
async def create_runtime():
    # Create a single-threaded runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Create list of available tools
    tools: List[Tool] = [FunctionTool(get_stock_price, description="Get the stock price.")]
    
    # Register the tool executor agent
    await ToolAgent.register(runtime, "tool_executor_agent", lambda: ToolAgent("tool executor agent", tools))
    
    # Register the tool use agent with OpenAI model
    await ToolUseAgent.register(
        runtime,
        "tool_use_agent",
        lambda: ToolUseAgent(
            OpenAIChatCompletionClient(model="gpt-4o-mini"), [tool.schema for tool in tools], "tool_executor_agent"
        ),
    )
    return runtime

async def run(runtime):
    # Start the runtime
    runtime.start()
    
    # Create agent ID for the tool use agent
    tool_use_agent = AgentId("tool_use_agent", "default")
    
    # Send a test message and get response
    response = await runtime.send_message(Message("What is the stock price of NVDA on 2024/06/01?"), tool_use_agent)
    print(response.content)
    
    # Clean up by stopping the runtime
    await runtime.stop()

# Main execution block
if __name__ == "__main__":
    # Create and run the runtime
    runtime = asyncio.run(create_runtime())
    asyncio.run(run(runtime))