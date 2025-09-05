import json
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from backend.llm import get_llm
from backend.mermaid_lib import (
    call_mermaid_api,
    render_mermaid_svg_bytes,
    svg_bytes_to_data_uri,
)
from backend.paths import CONFIG_FILE_PATH, OUTPUTS_DIR, PROMPT_CONFIG_FILE_PATH
from backend.prompt_builder import build_prompt_from_config
from backend.utils import load_config, save_state_checkpoint

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


class CombinedResearchOutput(BaseModel):
    """Combined output from both initial and deep research phases."""

    initial_research: InitialResearchOutput = Field(
        description="Output from the initial research phase."
    )
    deep_research: DeepResearchOutput = Field(
        description="Output from the deep research phase."
    )


class PlannedVisualOutput(BaseModel):
    """Represents the plan of a visual element that is part of a Wiki Section."""

    visual_type: Literal["table", "image", "mermaid_diagram"] = Field(
        description="The type of visual element. This could be a table, image, or mermaid diagram."
    )
    description: str = Field(
        description="A detailed description of the visual element that explains its purpose and content."
    )


class PlannedSectionOutput(BaseModel):
    """
    Represents a section (at any level) within a wiki page.
    A section title could be a heading, subheading, or any other markdown formatting.
    """

    title: str = Field(
        description="The title of the section in markdown format. This should always be an H2 starting with 2 hashes (##). "
    )
    description: str = Field(
        description="A detailed description of the suggested content to be present in the section."
    )
    visuals: List[PlannedVisualOutput] = Field(
        default_factory=list,
        description="List of planned Visuals related to the section.",
    )


class PlannerOutput(BaseModel):
    """Output from the planner agent that creates a structured plan for the wiki content with sections and visual suggestions, based on the research done by the research agent."""

    title: str = Field(
        description="The title of the wiki page. This should always be an H1 starting with 1 hash (#)."
    )
    sections: List[PlannedSectionOutput] = Field(
        default_factory=list,
        description="List of sections to be included in the wiki page, each with its own title, description, and visuals.",
    )


class WikiSectionOutput(BaseModel):
    """Represents a section of the generated wiki, excluding visuals."""

    title: str = Field(
        description="Title of the section. NOTE that this should ALWAYS be the same as the corresponding section in the plan."
    )
    content: str = Field(
        description="Full markdown content for this section (text only). This should NOT include any visual elements AT ALL like images, diagrams or tables."
    )


class WikiOutput(BaseModel):
    """Structured output for the full wiki (text only, no visuals)."""

    title: str = Field(
        description="Title of the wiki page. NOTE that this should ALWAYS be the same as the title in the plan."
    )
    sections: List[WikiSectionOutput] = Field(
        default_factory=list,
        description="Textual sections for the wiki, one per section in the plan, in order. NOTE that sections should ALWAYS match the corresponding sections in the planned structure.",
    )


class CodedVisualOutput(BaseModel):
    """Represents a visual element that is part of a Wiki Section. This includes the code to generate the visual."""

    title: str = Field(description="Title of the visual element.")
    visual_type: Literal["mermaid_diagram", "table", "image"] = Field(
        description="The type of visual element."
    )
    code: str = Field(
        description="""Code to generate the visual element (if applicable). This is required if visual_type is 'table' or 'mermaid_diagram'. This could be markdown formatted data for tables or Mermaid code for diagrams.
        The code should NOT be in a code block, just plain text. The formatting will be handled when rendering the final wiki based on the visual type.
        For tables, ensure the code is valid markdown table syntax.
        For mermaid diagrams, ensure the code is valid Mermaid syntax and use double quotes for all labels to avoid issues with syntax and parsing due to special characters.
        """,
        default=None,
    )
    image_url: str = Field(
        description="URL for the image visual (if applicable). This is required if visual_type is 'image'.",
        default=None,
    )


class CodedVisualSectionOutput(BaseModel):
    """
    Represents a section of the generated wiki that contains visual elements.
    """

    # title: str = Field(
    #     description="Title of the section. NOTE that this should ALWAYS be the same as the corresponding section in the wiki text."
    # )
    visuals: List[CodedVisualOutput] = Field(
        default_factory=list,
        description="List of visual elements related to the section.",
    )


class DesignCoderOutput(BaseModel):
    """Structured output for the design coding process."""

    # title: str = Field(
    #     description="The title of the wiki page. NOTE that this should ALWAYS be the same as the title in the plan."
    # )
    sections: List[CodedVisualSectionOutput] = Field(
        default_factory=list,
        description="List of sections to be included in the wiki page, each with its own title and visuals. NOTE that sections should ALWAYS match the corresponding sections in the planned structure.",
    )


# Mapping state keys to node names and their expected output types.
STATE_KEY_MAP = OrderedDict(
    {
        "combined_research": ("research_agent_node", CombinedResearchOutput),
        "structure_plan": ("planner_agent_node", PlannerOutput),
        "wiki_content": ("content_writer_agent_node", WikiOutput),
        "design_code": ("design_coder_agent_node", DesignCoderOutput),
        "final_wiki_path": ("merge_wiki_and_visuals_node", str),
    }
)

# ===================
# Define State
# ===================


class AgentState(TypedDict):
    """State for the agentic wiki creation process."""

    user_input: str | None  # Initial user input text
    combined_research: dict | None  # Output from research agent
    structure_plan: dict | None  # Output from planner agent
    wiki_content: dict | None  # Generated wiki content (text only)
    design_code: dict | None  # Generated design code (e.g. Mermaid diagrams and tables)
    final_wiki_path: (
        str | None
    )  # Path to the final merged wiki file after merging text and visuals
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


def exit_bot_node(state: AgentState) -> dict:
    print("\n" + "🚪" + "=" * 58 + "🚪")
    print("    GOODBYE!")
    print("=" * 60)
    return {"quit": True}


# ========== Prompt Config ==========

prompt_cfg: dict = load_config(PROMPT_CONFIG_FILE_PATH)
app_cfg: dict = load_config(CONFIG_FILE_PATH)

# ========== Tool Definitions ==========


@tool
def validate_mermaid_syntax(mermaid_code: str) -> bool:
    """
    Validate Mermaid syntax by attempting to render it via the mermaid.ink API.

    Args:
        mermaid_code (str): The Mermaid diagram syntax as a string.

    Returns:
        bool: True if the syntax is valid (or if API call failed), False otherwise.
        str: Error message if invalid or if API call failed, None otherwise.
    """
    resp, error = call_mermaid_api(mermaid_code)
    print("response in validation: ", resp)
    if resp is None:
        print("Error calling Mermaid API:", error)
        return False, error
    if resp is not None and resp.status_code == 400:
        print("Mermaid validation error:", error)
        return False, error
    return True


def get_tools():
    return [
        validate_mermaid_syntax,
        TavilySearch(
            topic="general",
            search_depth="advanced",
            include_images=True,
        ),
    ]


# ========== Agent Nodes ==========


def research_agent_node(state: AgentState) -> dict:
    """Research agent that conducts topic research in two phases.

    First performs basic topic search for overview and key concepts,
    then searches each key concept for detailed understanding.
    Returns combined research results in structured format.
    """

    # Check if combined_research is already populated
    if state.get("combined_research"):
        print("✅ Combined research is already populated. Skipping research node.")
        return {}

    research_llm = get_llm(app_cfg["research_model_llm"], temperature=0)

    initial_config = prompt_cfg["research_agent_cfg"]
    initial_prompt = build_prompt_from_config(
        initial_config,
        app_config=app_cfg,
        input_data=repr(state["user_input"]),
    )

    research_agent: CompiledStateGraph = create_react_agent(
        name="ResearchAgent",
        model=research_llm,
        tools=[
            TavilySearch(
                topic="general",
                search_depth="advanced",
                include_images=True,
            )
        ],
        debug=True,
        response_format=CombinedResearchOutput,
        prompt=initial_prompt,
    )

    agent_response = research_agent.invoke(
        {
            "input": f'Now perform research for the given user input: {state["user_input"]}'
        }
    )
    i = 0
    for event in agent_response:
        message = event
        print(f"Agent message #{i}: {message}")
        i += 1

    structured_response: CombinedResearchOutput = agent_response["structured_response"]

    result = {"combined_research": structured_response.model_dump()}

    # Save checkpoint
    topic = get_cleaned_topic_name(state["user_input"])
    topic_dir = get_topic_directory_name(topic, create=True)
    save_state_checkpoint({**state, **result}, topic_dir)
    print("🔄 Research agent node completed - State checkpoint saved")

    return result


def planner_agent_node(state: AgentState) -> dict:
    """Planner agent - creates a structured plan for the wiki content with sections and visual suggestions, based on the research done by the research agent."""
    # Check if structure_plan is already populated
    if state.get("structure_plan"):
        print("✅ Structure plan is already populated. Skipping planner node.")
        return {}

    config = prompt_cfg["planner_agent_cfg"]
    planner_llm = get_llm(app_cfg["planner_model_llm"], temperature=0)

    planner_prompt = build_prompt_from_config(
        config,
        app_config=app_cfg,
        input_data=repr(state["combined_research"]),
    )

    planner_response = planner_llm.with_structured_output(PlannerOutput).invoke(
        planner_prompt
    )

    result = {
        "structure_plan": planner_response.model_dump(),
    }

    # Save checkpoint
    topic = get_cleaned_topic_name(state["user_input"])
    topic_dir = get_topic_directory_name(topic, create=True)
    save_state_checkpoint({**state, **result}, topic_dir)
    print("🔄 Planner agent node completed - State checkpoint saved")

    return result


def content_writer_agent_node(state: AgentState) -> dict:
    """Content writer agent - generates the full wiki content for all sections in a single LLM call with structured output, excluding any visual content."""

    if state.get("wiki_content"):
        print("✅ Wiki content is already populated. Skipping content writer node.")
        return {}

    content_llm = get_llm(app_cfg["content_writer_model_llm"], temperature=0)
    structure_plan: PlannerOutput = deepcopy(state["structure_plan"])

    # strip the visual planning so that the content writer does not get confused
    for section in structure_plan["sections"]:
        del section["visuals"]

    # Build a single prompt for the entire plan (all sections)
    full_plan_prompt = build_prompt_from_config(
        prompt_cfg["content_writer_agent_cfg"],
        app_config=app_cfg,
        input_data=repr(structure_plan),
    )

    content_writer_agent: CompiledStateGraph = create_react_agent(
        name="ContentWriterAgent",
        model=content_llm,
        tools=[
            TavilySearch(
                topic="general",
                search_depth="advanced",
            )
        ],
        debug=True,
        response_format=WikiOutput,
        prompt=full_plan_prompt,
    )

    agent_response = content_writer_agent.invoke(
        {
            "input": f"Now write the full wiki content (text only, no visuals) using the structure plan given above."
        }
    )
    i = 0
    for event in agent_response:
        message = event
        print(f"Agent message #{i}: {message}")
        i += 1

    structured_response: WikiOutput = agent_response["structured_response"]

    result = {"wiki_content": structured_response.model_dump()}

    # Save checkpoint
    topic = get_cleaned_topic_name(state["user_input"])
    topic_dir = get_topic_directory_name(topic, create=True)
    save_state_checkpoint({**state, **result}, topic_dir)
    print("🔄 Content writer node completed - State checkpoint saved")

    return result


def design_coder_agent_node(state: AgentState) -> dict:
    """Design coder agent - generates Mermaid diagrams for the wiki based on the planner's visual suggestions."""

    if state.get("design_code"):
        print("✅ Visual content already generated. Skipping design coder agent node.")
        return {}

    design_llm = get_llm(app_cfg["design_coder_model_llm"], temperature=0)
    structure_plan: PlannerOutput = state["structure_plan"]
    design_coder_result: DesignCoderOutput = deepcopy(
        structure_plan
    )  # we will fill in the visuals in this structure

    # Build prompt for diagrams generation from planner's specification
    visual_prompt = build_prompt_from_config(
        prompt_cfg["design_coder_agent_cfg"],
        app_config=app_cfg,
        # input_data=repr(structure_plan),
    )

    design_coder_agent: CompiledStateGraph = create_react_agent(
        name="DesignCoderAgent",
        model=design_llm,
        tools=[
            TavilySearch(
                topic="general",
                search_depth="advanced",
                include_images=True,
            )
        ],
        debug=True,
        response_format=CodedVisualSectionOutput,
        prompt=visual_prompt,
    )

    # instead of calling agent for entire wiki at once, call it for each section
    # this way, we can ensure that the number of sections and section titles remain exactly the same as the input plan
    # and we can just stitch the visuals into the sections ourselves
    for section in design_coder_result["sections"]:
        if not section.get("visuals"):
            continue
        print(f"\n🎨 Generating visuals for section: {section['title']}")
        section_agent_response = design_coder_agent.invoke(
            {
                "input": f"Generate visuals for the given wiki section. What follows below is the planned visual specification for the section. Ensure Mermaid diagrams use double quotes for all labels.\n\n{repr(section)}"
            }
        )
        section_structured_response: CodedVisualSectionOutput = section_agent_response[
            "structured_response"
        ]

        # stitch the generated visuals into the planned section
        section["visuals"] = section_structured_response.model_dump()["visuals"]

    # agent_response = design_coder_agent.invoke(
    #     {
    #         "input": f"Now write the code for all the visuals (diagrams, tables, images) using the structure plan given above. Ensure Mermaid diagrams use double quotes for all labels."
    #     }
    # )

    # i = 0
    # for event in agent_response:
    #     message = event
    #     print(f"Agent message #{i}: {message}")
    #     i += 1
    # structured_response: DesignCoderOutput = agent_response["structured_response"]

    print(f"Design Coder Result: {design_coder_result}")
    result = {"design_code": design_coder_result.model_dump()}

    # Save checkpoint
    topic = get_cleaned_topic_name(state["user_input"])
    topic_dir = get_topic_directory_name(topic, create=True)
    save_state_checkpoint({**state, **result}, topic_dir)
    print("🔄 Design coder node completed - State checkpoint saved")

    return result


def merge_wiki_and_visuals_node(state: AgentState) -> dict:
    """Merges the wiki content and visual code into a single markdown file."""
    if not state.get("wiki_content") or not state.get("design_code"):
        print("❌ Cannot merge wiki and visuals. Missing content.")
        return {}

    if state.get("final_wiki_path"):
        print("✅ Final wiki with visuals already created. Skipping merge node.")
        return {}

    topic = get_cleaned_topic_name(state["user_input"])
    topic_dir = get_topic_directory_name(topic, create=True)
    output_file = str(Path(topic_dir) / "complete_wiki.md")

    wiki_output = state["wiki_content"]
    design_code = state["design_code"]

    assert len(wiki_output["sections"]) == len(design_code["sections"]), (
        "Number of sections in wiki content and design code do not match."
        f" Wiki sections: {len(wiki_output['sections'])}, Design sections: {len(design_code['sections'])}"
    )

    assert all(
        wiki_output["sections"][i]["title"] == design_code["sections"][i]["title"]
        for i in range(len(wiki_output["sections"]))
    ), (
        "Section titles in wiki content and design code do not match."
        f"Wiki section titles: {[section['title'] for section in wiki_output['sections']]}, "
        f"Design section titles: {[section['title'] for section in design_code['sections']]}"
    )

    # Simple merge logic: Append visuals at the end of the wiki content

    merged_content = f"# {wiki_output['title']}\n\n"
    for visual_section, content_section in zip(
        design_code["sections"], wiki_output["sections"]
    ):
        merged_content += (
            f"{content_section['title']}\n{content_section['content']}\n\n"
        )
        for visual in visual_section["visuals"]:
            if visual["visual_type"] == "image":
                if not visual.get("image_url"):
                    print(
                        f"❗ Image URL '{visual.get('image_url')}' does not exist. Skipping image."
                    )
                    continue
                merged_content += f"![{visual['title']}]({visual['image_url']})\n\n"
            else:
                if not visual.get("code"):
                    print(
                        f"❗ No code provided for {visual['visual_type']} titled '{visual['title']}'. Skipping."
                    )
                    continue
                if visual["visual_type"] == "mermaid_diagram":
                    print("\tRendering Mermaid diagram...")
                    svg_bytes, err = render_mermaid_svg_bytes(visual["code"])
                    if err:
                        base_url = app_cfg.get("mermaid_api_base_url")
                        print(
                            f"❗ Failed to render Mermaid SVG using base '{base_url}': {err}. Embedding code block instead."
                        )
                        merged_content += f"```mermaid\n{visual['code']}\n```\n\n"
                    else:
                        data_uri = svg_bytes_to_data_uri(svg_bytes)
                        alt = visual.get("title") or "Diagram"
                        merged_content += f"![{alt}]({data_uri})\n\n"
                else:  # table
                    merged_content += f"{visual['code']}\n\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(merged_content)

    result = {"final_wiki_path": output_file}

    # Save checkpoint
    topic_dir = get_topic_directory_name(state["user_input"])
    save_state_checkpoint({**state, **result}, topic_dir)
    print("🔄 Merge node completed - State checkpoint saved")
    print(f"📄 Complete wiki with visuals saved to: {output_file}")

    return result


# ========== Graph Assembly ==========


def register_nodes(builder: StateGraph) -> None:
    """Registers all nodes in the graph."""
    for node_name in set([x[0] for x in STATE_KEY_MAP.values()]):
        builder.add_node(node_name, globals()[node_name])


def define_edges(builder: StateGraph) -> None:
    """Defines the edges of the graph."""
    builder.set_entry_point("research_agent_node")  # Start directly with research
    builder.add_edge("research_agent_node", "planner_agent_node")

    # Parallelize content writing and design coding
    builder.add_edge("planner_agent_node", "content_writer_agent_node")
    builder.add_edge("planner_agent_node", "design_coder_agent_node")

    # Merge results and end
    builder.add_edge("content_writer_agent_node", "merge_wiki_and_visuals_node")
    builder.add_edge("design_coder_agent_node", "merge_wiki_and_visuals_node")
    builder.add_edge("merge_wiki_and_visuals_node", END)


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
    user_input: str,
    file_name_to_save_state: str = "saved_wiki_state.json",
    starting_state_file_path: Optional[str] = None,
) -> str:
    """Main function to run the wiki creation pipeline. The provided topic will be cleaned
    (removing special characters and replacing spaces with underscores) and used to create
    a directory for storing outputs.

    Args:
        topic: The topic to create a wiki for. Will be cleaned for filesystem compatibility.
        file_name_to_save_state: Name of the file to save the final state
        starting_state_file_path: Path to a JSON file containing the starting state

    Returns:
        str: Path to the generated wiki file
    """
    print("\n📃 Starting wiki creation pipeline...")
    graph = build_graph()

    # Get image representation of the graph
    print("\n📊 Generating graph visualization...")
    graph_png_save_path = str(Path(OUTPUTS_DIR) / "wiki_creation_graph.png")
    graph.get_graph().draw_mermaid_png(
        output_file_path=graph_png_save_path,
        # draw_method=MermaidDrawMethod.PYPPETEER,
        # max_retries=5,
        # retry_delay=2.0,
    )
    print(f"\t💹 Graph image saved as '{graph_png_save_path}'.")

    starting_state: AgentState = {
        "user_input": user_input,  # Initialize with exact user input, not cleaned to give the research agent the exact input
        "combined_research": None,
        "structure_plan": None,
        "wiki_content": None,
        "design_code": None,
        "final_wiki_path": None,
        "user_preferences": None,
        "retry_count": None,
        "quit": None,
    }

    if not starting_state_file_path:
        print(f"🛠️ No state file path provided. Generating wiki from scratch.")
    else:
        print(f"📂 Loading state from '{starting_state_file_path}'...")

        # If state file was requested but doesn't exist, return early
        if not Path(starting_state_file_path).exists():
            print(
                f"❗ State file '{starting_state_file_path}' was passed but it does not exist."
            )
            return None

        with open(starting_state_file_path, "r") as f:
            loaded_state = json.load(f)
            # Merge loaded state with initial_state to ensure all keys are present
            starting_state.update(loaded_state)
        print("\t✅ State loaded successfully.")

        # load topic from state
        user_input = starting_state.get("user_input")

    if user_input and user_input.lower() == EXIT_COMMAND:
        return  # exit early if the topic is the exit command

    print(f"\n🚀 Starting the wiki creation process for user input: '{user_input}'...")

    # Clean topic name and create directory
    cleaned_topic = get_cleaned_topic_name(user_input)
    topic_dir = get_topic_directory_name(cleaned_topic, create=True)

    config = {"configurable": {"thread_id": "1", "recursion_limit": 200}}
    # final_state: dict = graph.invoke(starting_state, config=config)

    # stream the graph output to console
    final_state = None
    i = 0
    for state in graph.stream(
        input=starting_state, stream_mode="values", config=config
    ):
        final_state = state
        # print(state["quit"])
        print(f"\n--- State #{i} ---")
        print("-" * 50)
        i += 1

    # final_state = final_state[-1] if final_state else starting_state

    # # Directory was already created at the start, just use it
    state_file = Path(topic_dir) / file_name_to_save_state
    with open(state_file, "w") as f:
        json.dump(final_state, f, indent=4)

    print(f"\n📄 Final state saved in topic directory: '{state_file}'")

    # Return the path to the generated wiki file
    return final_state.get("final_wiki_path")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a wiki on any topic with automatic visuals."
    )
    parser.add_argument(
        "topic",
        type=str,
        help="The topic to create a wiki for",
    )
    parser.add_argument(
        "--state",
        type=str,
        help="Path to a saved state file to resume from",
        required=False,
    )
    args = parser.parse_args()

    wiki_path = main(
        user_input=args.topic,
        file_name_to_save_state="saved_wiki_state.json",
        starting_state_file_path=args.state,
    )
    print(f"\n📄 Generated wiki file: {wiki_path}")
