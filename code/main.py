import json
import os
from collections import OrderedDict
from operator import add
from pathlib import Path
from typing import Annotated, List, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool  # Import the tool decorator

# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from llm import get_llm
from paths import CONFIG_FILE_PATH, OUTPUTS_DIR, PROMPT_CONFIG_FILE_PATH
from prompt_builder import build_prompt_from_config
from pydantic import BaseModel, Field
from pyjokes import get_joke
from typing_extensions import TypedDict
from utils import load_config

# ===================
# Define Structured Output Classes for Agents
# ===================

EXIT_COMMAND = "exit"  # Command to exit the bot


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


class WikiSectionOutput(BaseModel):
    """Represents a section of the generated wiki, excluding visuals."""

    title: str = Field(description="Title of the section")
    content: str = Field(
        description="Full markdown content for this section (text only, no visuals)"
    )


class WikiOutput(BaseModel):
    """Structured output for the full wiki (text only, no visuals)."""

    title: str = Field(description="Title of the wiki page")
    sections: List[WikiSectionOutput] = Field(
        default_factory=list,
        description="Textual sections for the wiki, one per section in the plan, in order.",
    )


# Mapping state keys to node names and their expected output types.
STATE_KEY_MAP = OrderedDict(
    {
        "user_input": ("get_user_input_node", str),
        "initial_research": ("research_agent_node", InitialResearchOutput),
        "deep_research": ("research_agent_node", DeepResearchOutput),
        "structure_plan": ("planner_agent_node", PlannerOutput),
        "wiki_content": ("content_writer_agent_node", WikiOutput),
    }
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
    # Check if user_input is already populated
    if state.get("user_input"):
        print("✅ User input is already populated. Skipping input node.")
        return {}

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
    elif state["user_input"].lower() == EXIT_COMMAND:
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
    # Check if initial_research is already populated
    if state.get("initial_research"):
        print("✅ Initial research is already populated. Skipping research node.")
        return {}

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
    # Check if structure_plan is already populated
    if state.get("structure_plan"):
        print("✅ Structure plan is already populated. Skipping planner node.")
        return {}

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


def content_writer_agent_node(state: AgentState) -> dict:
    """Content writer agent - generates the full wiki content for all sections in a single LLM call with structured output, excluding any visual content."""

    if state.get("wiki_content"):
        print("✅ Wiki content is already populated. Skipping content writer node.")
        return {}

    content_llm = get_llm(app_cfg["content_writer_model_llm"], temperature=0.3)
    structure_plan: PlannerOutput = state["structure_plan"]

    # Build a single prompt for the entire plan (all sections)
    full_plan_prompt = build_prompt_from_config(
        prompt_cfg["content_writer_agent_cfg"],
        app_config=app_cfg,
        input_data=repr(structure_plan),
    )

    # Single LLM call to generate all textual content with strongly-typed structured output
    wiki_output: WikiOutput = content_llm.with_structured_output(WikiOutput).invoke(
        full_plan_prompt
    )

    # Convert structured output to markdown (text only)
    wiki_content = f"# {wiki_output.title}\n\n"
    for section in wiki_output.sections:
        wiki_content += f"{section.title}\n{section.content}\n\n"

    topic = get_cleaned_topic_name(state["user_input"])
    topic_dir = get_topic_directory_name(topic, create=True)
    output_file = str(Path(topic_dir) / "generated_wiki_content.md")

    with open(output_file, "w") as f:
        f.write(wiki_content)

    print(f"📄 Wiki content generated and saved to: {output_file}")
    return {"wiki_content": wiki_content}


# TODO: Implement Design Coder Agent
def design_coder_agent_node(state: AgentState) -> dict:
    """Design coder agent - creates Mermaid diagrams for the wiki based on the planner's visual suggestions."""
    pass


# ========== Graph Assembly ==========


def register_nodes(builder: StateGraph) -> None:
    """Registers all nodes in the graph."""
    for node_name in set([x[0] for x in STATE_KEY_MAP.values()]):
        builder.add_node(node_name, globals()[node_name])
    builder.add_node("exit_bot_node", exit_bot_node)


def define_edges(builder: StateGraph) -> None:
    """Defines the edges of the graph."""
    builder.set_entry_point("get_user_input_node")
    builder.add_conditional_edges("get_user_input_node", route_choice)
    builder.add_edge("research_agent_node", "planner_agent_node")
    builder.add_edge("planner_agent_node", "content_writer_agent_node")
    # builder.add_edge("content_writer_agent_node", "design_coder_agent_node")
    builder.add_edge("content_writer_agent_node", "exit_bot_node")
    builder.add_edge("exit_bot_node", END)


def compile_graph(builder: StateGraph) -> CompiledStateGraph:
    """Compiles the graph with checkpointing and store."""
    checkpointer = InMemorySaver()
    in_memory_store = InMemoryStore()
    return builder.compile(checkpointer=checkpointer, store=in_memory_store)


def build_graph() -> CompiledStateGraph:
    """Builds the LangGraph graph."""
    builder = StateGraph(AgentState)
    register_nodes(builder)
    define_edges(builder)
    return compile_graph(builder)


# ========== Entry Point ==========


def main(
    file_name_to_save_state: str = "saved_state.json",
    starting_state_file_path: str = None,
) -> None:
    """Main function to run the wiki creation pipeline.

    Args:
        file_name_to_save_state (str, optional): Name of the file to save the final state.
            This will be saved in the topic directory. Defaults to "saved_state.json".
        starting_state_file_path (str, optional): Path to a JSON file containing the
            starting state. If provided, the pipeline will continue from that state.
            If not provided, the pipeline will start from scratch. Defaults to None.
    """
    print("\n📃 Starting wiki creation pipeline...")
    graph = build_graph()

    # Get image representation of the graph
    print("\n📊 Generating graph...")
    graph_png_save_path = str(Path(OUTPUTS_DIR) / "wiki_creation_graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=graph_png_save_path)
    print(f"\t💹 Graph image saved as '{graph_png_save_path}'.")

    print("\n🚀 Starting the wiki creation process...")

    starting_state: AgentState = {
        "user_input": None,
        "initial_research": None,
        "deep_research": None,
        "structure_plan": None,
        "user_preferences": None,
        "retry_count": None,
        "quit": None,
    }

    if starting_state_file_path:
        if not Path(starting_state_file_path).exists():
            print(
                f"❗ State file '{starting_state_file_path}' does not exist. Starting with an empty state."
            )
        else:
            print(f"📂 Loading state from '{starting_state_file_path}'...")
            with open(starting_state_file_path, "r") as f:
                loaded_state = json.load(f)
                # Merge loaded state with initial_state to ensure all keys are present
                starting_state.update(loaded_state)
            print("\t✅ State loaded successfully.")

    config = {"configurable": {"thread_id": "1", "recursion_limit": 200}}
    final_state: dict = graph.invoke(starting_state, config=config)

    if final_state["user_input"] == EXIT_COMMAND:
        return

    print("\t✅ Done.")

    # Get directory name from user input (which is our topic)
    topic = (
        get_cleaned_topic_name(final_state["user_input"])
        if final_state.get("user_input")
        else "default"
    )
    topic_dir = get_topic_directory_name(topic, create=True)

    # Save state in the topic directory
    output_file = str(Path(topic_dir) / file_name_to_save_state)

    # Save final state as JSON with proper formatting
    with open(output_file, "w") as f:
        json.dump(final_state, f, indent=4)

    print(f"\n📄 Final state saved in topic directory: '{output_file}'.")

    print("Printing Final State:")
    print(final_state)


if __name__ == "__main__":
    main(
        file_name_to_save_state="saved_wiki_state.json",
        # starting_state_file_path=os.path.join(
        #     OUTPUTS_DIR, "Retrieval_Augmented_Generation", "saved_wiki_state.json"
        # ),  # Set to a path like os.path.join(OUTPUTS_DIR, "Your_Topic", "saved_wiki_state.json") to load
    )
