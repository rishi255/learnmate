import json
from operator import add
from pathlib import Path
from typing import Annotated, List, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool  # Import the tool decorator

# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from llm import get_llm
from paths import CONFIG_FILE_PATH, PROMPT_CONFIG_FILE_PATH
from prompt_builder import build_prompt_from_config
from paths import CONFIG_FILE_PATH, PROMPT_CONFIG_FILE_PATH, OUTPUTS_DIR
from pydantic import BaseModel, Field
from pyjokes import get_joke
from typing_extensions import TypedDict
from utils import load_config

# ===================
# Define Structured Output Classes for Agents
# ===================


class ResearchedSubtopicOutput(BaseModel):
    """Represents a subtopic within a research topic."""

    name: str = Field(
        description="The title of the subtopic to be covered in the wiki."
    )  # title of the subtopic
    description: str = Field(
        description="A brief description of the subtopic to be covered in the wiki."
    )  # description of the subtopic


class InitialResearchOutput(BaseModel):
    """Output from initial research phase"""

    title: str = Field(description="The title for the wiki page")
    topic: str = Field(description="Main topic being researched")
    category: str = Field(
        description="Domain/category of the topic (e.g. Science/Finance/Software)"
    )
    summary: str = Field(description="Brief overview of the topic")
    key_concepts: List[str] = Field(
        default_factory=list, description="Core concepts identified in initial research"
    )
    search_urls: List[str] = Field(
        default_factory=list, description="URLs of sources found in initial search"
    )


class DeepResearchOutput(BaseModel):
    """Output from deep research phase, building upon initial research"""

    subtopics: List[ResearchedSubtopicOutput] = Field(
        default_factory=list,
        description="Detailed subtopics discovered during deep research",
    )
    visual_suggestions: List[str] = Field(
        default_factory=list,
        description="Description of concepts/relationships that would benefit from visual representation. Each string describes what the visual should explain or represent. A visual can be a Mermaid diagram, table, or image.",
    )


class PlannedVisualOutput(BaseModel):
    """Represents the plan of a visual element that is part of a Wiki Section."""

    visual_type: Literal["mermaid_diagram", "table", "image"] = Field(
        description="The type of visual element."
    )
    description: str = Field(
        description="A detailed description of the visual element that explains its purpose and content."
    )
    # generation_prompt: str = Field(
    #     description="This should clearly outline the steps or components of the visual element, which can be used to generate the visual using Mermaid code.",
    # )


# class VisualWithCodeOutput(VisualWithoutCodeOutput):
#     """Represents a visual element that is part of a Wiki Subtopic. This also includes the code to generate the visual."""

#     code: str = Field(
#         description="The Mermaid code to generate the visual element.",
#         default=None,
#     )  # code to generate the visual (if applicable)


class PlannedSectionOutput(BaseModel):
    """
    Represents a section (at any level) within a wiki page.
    A section title could be a heading, subheading, or any other markdown formatting.
    """

    title: str = Field(description="The title of the section in markdown format.")
    description: str = Field(
        description="A detailed description of the suggested content to be present in the section."
    )
    visuals: List[PlannedVisualOutput] = Field(
        default_factory=list,
        description="List of planned Visuals related to the section.",
    )


class PlannerOutput(BaseModel):
    """Output from the planner agent that creates a structured plan for the wiki content with sections and visual suggestions, based on the deep research done by the research agent."""

    title: str = Field(description="The title of the wiki page")
    sections: List[PlannedSectionOutput] = Field(
        default_factory=list,
        description="List of sections to be included in the wiki page, each with its own title, description, and visuals.",
    )


# ===================
# Define State
# ===================


class AgentState(TypedDict):
    """State for the agentic wiki creation process."""

    user_input: str | None  # Initial user input text
    initial_research: dict | None  # Output from initial research phase
    deep_research: dict | None  # Output from deep research phase
    structure_plan: dict | None  # Output from planner agent
    user_preferences: dict | None  # User preferences for the topic
    retry_count: int | None
    quit: bool | None  # Whether to exit the bot


# ===================
# Utilities
# ===================


def get_cleaned_topic_name(topic: str) -> str:
    """Cleans the topic name by removing unwanted characters.

    Args:
        topic: The raw topic name

    Returns:
        str: Cleaned topic name suitable for file paths
    """
    return "".join(
        char for char in topic if char.isalnum() or char in ("_", " ") or char.isspace()
    ).replace(" ", "_")


def get_topic_directory_name(topic: str, create: bool = True) -> str:
    """Returns (and optionally creates) the directory path for storing topic outputs.

    Args:
        topic: The topic name
        create: Whether to create the directory if it doesn't exist

    Returns:
        str: Absolute path to the topic directory
    """
    topic_name = get_cleaned_topic_name(topic)
    topic_dir = Path(OUTPUTS_DIR) / topic_name

    if create:
        topic_dir.mkdir(parents=True, exist_ok=True)

    return str(topic_dir)


# ===================
# Define Nodes
# ===================


def get_user_input_node(state: AgentState) -> dict:
    """Gets user input for the topic of the wiki."""
    print("Welcome to the Wiki Creator Bot! Let's create a wiki together.")
    user_input = input(
        "Enter the topic you want to create a wiki for (or type 'exit' to quit): "
    ).strip()

    return {"user_input": user_input}


def route_choice(
    state: AgentState,
) -> Literal["get_user_input_node", "research_agent_node", "exit_bot_node"]:
    """Router function to determine whether to get more input, exit, or proceed.

    Returns:
        get_user_input_node: If no input provided, ask for input again
        exit_bot_node: If user types 'exit'
        research_agent_node: Otherwise proceed with research
    """
    if not state["user_input"]:
        print("❌ No input provided. Please enter a topic.")
        return "get_user_input_node"
    elif state["user_input"].lower() == "exit":
        return "exit_bot_node"
    else:
        return "research_agent_node"


def exit_bot_node(state: AgentState) -> dict:
    print("\n" + "🚪" + "=" * 58 + "🚪")
    print("    GOODBYE!")
    print("=" * 60)
    return {"quit": True}


def verify_mermaid_code_node(state: AgentState) -> dict:
    """Verifies that all Mermaid code in the wiki sections is valid.

    Args:
        state: Current agent state containing wiki with sections

    Returns:
        dict: Updated state with verification results
    """
    # TODO: Implement actual Mermaid code verification logic
    return NotImplementedError("Mermaid code verification is not implemented yet.")


# ========== Prompt Config ==========

prompt_cfg: dict = load_config(PROMPT_CONFIG_FILE_PATH)
app_cfg: dict = load_config(CONFIG_FILE_PATH)

# ========== Tool Definitions ==========


def get_web_search_tool() -> TavilySearch:
    """Returns a configured web search tool instance.

    Returns:
        TavilySearch: Configured search tool with specified parameters
    """
    try:
        return TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_images=True,
        )
    except Exception as e:
        print(f"❌ Error initializing web search tool: {e}")
        raise


def get_tools():
    return [get_web_search_tool()]


# ========== Agent Nodes ==========


def research_agent_node(state: AgentState) -> dict:
    """Research agent that conducts topic research in two phases.

    First performs basic topic search for overview and key concepts,
    then searches each key concept for detailed understanding.
    Returns initial and deep research results in structured format.
    """
    research_llm = get_llm(app_cfg["research_model_llm"], temperature=0)
    search_tool = get_web_search_tool()

    # Initial Research Phase - Perform initial search
    initial_search_results = search_tool.invoke(state["user_input"])

    initial_config = prompt_cfg["initial_research_agent_cfg"]
    initial_prompt = build_prompt_from_config(
        initial_config,
        app_config=app_cfg,
        input_data=repr(initial_search_results),
    )

    initial_response: InitialResearchOutput = research_llm.with_structured_output(
        InitialResearchOutput
    ).invoke(initial_prompt)

    # Deep Research Phase - Search for each key concept
    concept_search_results = {}
    for concept in initial_response.key_concepts:
        concept_search = search_tool.invoke(f"{state['user_input']} {concept}")
        concept_search_results[concept] = concept_search

    deep_config = prompt_cfg["deep_research_agent_cfg"]
    deep_prompt = build_prompt_from_config(
        deep_config,
        app_config=app_cfg,
        input_data=repr(
            {
                "initial_research": initial_response.model_dump(),
                "concept_searches": concept_search_results,
            }
        ),
    )

    deep_response: DeepResearchOutput = research_llm.with_structured_output(
        DeepResearchOutput
    ).invoke(deep_prompt)

    return {
        "initial_research": initial_response.model_dump(),
        "deep_research": deep_response.model_dump(),
    }


def planner_agent_node(state: AgentState) -> dict:
    """Planner agent - creates a structured plan for the wiki content with sections and visual suggestions, based on the deep research done by the research agent."""
    config = prompt_cfg["planner_agent_cfg"]
    planner_llm = get_llm(app_cfg["planner_model_llm"], temperature=0)

    planner_prompt = build_prompt_from_config(
        config,
        app_config=app_cfg,
        input_data=repr(state["deep_research"]),
    )

    planner_response = planner_llm.with_structured_output(PlannerOutput).invoke(
        planner_prompt
    )

    return {
        "structure_plan": planner_response.model_dump(),
    }


# TODO: Implement Content Writer Agent
def content_writer_agent_node(state: AgentState) -> dict:
    """Content writer agent - generates the actual wiki content for each section based on the planner's structure."""
    pass


# TODO: Implement Design Coder Agent
def design_coder_agent_node(state: AgentState) -> dict:
    """Design coder agent - creates Mermaid diagrams for the wiki based on the planner's visual suggestions."""
    pass


# ========== Graph Assembly ==========


def build_graph() -> CompiledStateGraph:
    """Builds the LangGraph graph."""

    builder = StateGraph(AgentState)

    builder.add_node("get_user_input_node", get_user_input_node)
    builder.add_node("research_agent_node", research_agent_node)
    builder.add_node("exit_bot_node", exit_bot_node)
    # TODO: These nodes will be implemented in next phases
    builder.add_node("planner_agent_node", planner_agent_node)
    # builder.add_node(
    #     "content_writer_agent_node",
    #     content_writer_agent_node
    # )
    # builder.add_node(
    #     "design_coder_agent_node",
    #     design_coder_agent_node
    # )

    builder.set_entry_point("get_user_input_node")

    # decides whether to return to input or proceed to research
    builder.add_conditional_edges("get_user_input_node", route_choice)
    # TODO: Update edges when implementing other agents
    builder.add_edge("research_agent_node", "planner_agent_node")
    # builder.add_edge("planner_agent_node", "content_writer_agent_node")
    # builder.add_edge("content_writer_agent_node", "design_coder_agent_node")
    builder.add_edge("research_agent_node", "exit_bot_node")
    builder.add_edge("exit_bot_node", END)

    # Add state checkpointing
    checkpointer = InMemorySaver()
    in_memory_store = InMemoryStore()
    return builder.compile(checkpointer=checkpointer, store=in_memory_store)


# ========== Entry Point ==========


def main():
    print("\n📃 Starting wiki creation pipeline...")
    graph = build_graph()

    # Get image representation of the graph
    print("\n📊 Generating graph...")
    graph.get_graph().draw_mermaid_png(output_file_path="wiki_creation_graph.png")
    print("\t💹 Graph image saved as 'wiki_creation_graph.png'.")

    print("\n🚀 Starting the wiki creation process...")

    # Initialize state with None values for all fields
    initial_state: AgentState = {
        "user_input": None,
        "initial_research": None,
        "deep_research": None,
        "user_preferences": None,
        "retry_count": None,
        "quit": None,
    }

    # Set up config with recursion limit
    config = {"configurable": {"thread_id": "1", "recursion_limit": 200}}
    final_state: dict = graph.invoke(initial_state, config=config)
    print("\t✅ Done.")

    # Get directory name from user input (which is our topic)
    topic = (
        get_cleaned_topic_name(final_state["user_input"])
        if final_state.get("user_input")
        else "default"
    )
    topic_dir = get_topic_directory_name(topic, create=True)

    # Save state in the topic directory
    output_file = str(Path(topic_dir) / "final_wiki_state.json")

    # Save final state as JSON with proper formatting
    with open(output_file, "w") as f:
        json.dump(final_state, f, indent=4)

    print(f"\n📄 Final state saved in topic directory: '{output_file}'.")

    # if "wiki" in final_state:
    print("Printing Final State:")
    print(final_state)


if __name__ == "__main__":
    main()
