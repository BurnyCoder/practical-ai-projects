from datetime import datetime
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
import os
from rag import RAG

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM = os.getenv("OPENAI_LLM")
#OPENAI_LLM = "gpt-4o"

#our_system_prompt = os.getenv("SYSTEM_PROMPT")
our_system_prompt = f"""
Answer the following questions as best you can. You are an AI assistant called Exobrain.
Respond to the human as helpfully and accurately as possible. 
"""

precoded_end_of_system_prompt = """

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}


## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in the one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
```

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
"""

react_system_header_str = our_system_prompt + precoded_end_of_system_prompt


class ExocortexAgent:    
    def __init__(self):        
        self.rag = RAG()

        react_system_prompt = PromptTemplate(react_system_header_str)
        
        tools = [
            FunctionTool.from_defaults(
                name="AddContent", 
                fn=self.rag.add_to_index, 
                description="The AddContent tool enables you to incorporate new information into your database. It takes a URL as input, extracts the content from the specified webpage, and stores it in the database."
            ),
            FunctionTool.from_defaults(
                name="EmptyDatabase",
                fn=self.rag.empty_db,
                description="The EmptyDatabase tool allows you to remove all content from your database. It requires no input. Exercise caution when using this tool, as it will irreversibly delete all stored information."
            ),
            FunctionTool.from_defaults(
                name="FillDatabase",
                fn=self.rag.fill_db,
                description="The FillDatabase tool enables you to populate your database with pre-defined test data. It doesn't need any input. This tool is beneficial for initializing the database with a set of known documents for testing or demonstration purposes."
            ),
            FunctionTool.from_defaults(
                name="RetrieveFromDatabase", 
                fn=self.rag.query, 
                description="""The RetrieveFromDatabase tool allows you to fetch content from your database. It searches the database using the provided input and returns the most relevant content along with a synthesized LLM reply. Note that this tool may sometimes provide content regardless of its relevance. It's your task to identify and present the most pertinent information to the user."""
            )
        ]

        llm = OpenAI(model=OPENAI_LLM)
        self.agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

        self.agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

        self.agent.reset()

    def run(self, input=""):
        try:
            response_gen = str(self.agent.chat(input))
        except Exception as e:
            response_gen = f"An error occurred: {str(e)}"
        return response_gen

