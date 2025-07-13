import json
from operator import add
from typing import Annotated, List, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool  # Import the tool decorator

# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from llm import get_llm
from paths import CONFIG_FILE_PATH, PROMPT_CONFIG_FILE_PATH
from prompt_builder import build_prompt_from_config
from pydantic import BaseModel, Field
from pyjokes import get_joke
from typing_extensions import TypedDict
from utils import load_config

# ===================
# Define Structured Output Classes for Agents
# ===================


class SubtopicOutput(BaseModel):
    """Represents a subtopic within a research topic."""

    name: str = Field(
        description="The title of the subtopic to be covered in the wiki."
    )  # title of the subtopic
    description: str = Field(
        description="A brief description of the subtopic to be covered in the wiki."
    )  # description of the subtopic

    # def __repr__(self):
    #     """Print a subtopic with nice formatting."""
    #     return f"'{self.name}': {self.description}"


class VisualWithoutCodeOutput(BaseModel):
    """Represents a visual element that is part of a Wiki Subtopic."""

    visual_type: Literal["flowchart"] = Field(
        description="The type of visual element (image / flowchart)."
    )
    description: str = Field(description="A brief description of the visual element.")
    design_code_generation_prompt: str = Field(
        description="This should clearly outline the steps or components of the visual element, which can be used to generate the visual using Mermaid code.",
    )


class VisualWithCodeOutput(VisualWithoutCodeOutput):
    """Represents a visual element that is part of a Wiki Subtopic. This also includes the code to generate the visual."""

    code: str = Field(
        description="The Mermaid code to generate the visual element.",
        default=None,
    )  # code to generate the visual (if applicable)


class SectionInitialOutput(BaseModel):
    """
    Represents a section (at any level) within a wiki page.
    A section title could be a heading, subheading, or any other markdown formatting.
    """

    title: str = Field(description="The title of the section in markdown format.")
    description: str = Field(
        description="A brief description of the suggested content to be present in the section."
    )
    visuals: List[VisualWithoutCodeOutput] = Field(
        default_factory=list,
        description="List of Visuals related to the section.",
    )  # list of visuals related to the section

    def __repr__(self):
        """Print a section with nice formatting."""
        return f"'{self.title}': {self.description}. Visuals = {self.visuals})"


class SectionWithVisualCodeOutput(SectionInitialOutput):
    """
    Represents a section (at any level) within a wiki page.
    A section title could be a heading, subheading, or any other markdown formatting.
    This is a more complete version of SectionInitialOutput that includes visuals with code.
    """

    visuals: List[VisualWithCodeOutput] = Field(
        default_factory=list,
        description="List of Visuals related to the section. Includes Mermaid code to generate the visual.",
    )  # list of visuals related to the section

    def __repr__(self):
        return super().__repr__()


class WikiOutput(BaseModel):
    """
    Represents the output from the Research Agent.
    """

    # Wiki information
    title: str = Field(description="The title of the wiki page.")
    topic: str = Field(description="The topic of the wiki.")
    category: str = Field(
        description="Category of the wiki. This is the domain the topic belongs to."
    )
    summary: str = Field(
        description="A very short summary about the topic.",
    )
    subtopics: List[SubtopicOutput] = Field(
        default_factory=list,
        description="List of important subtopics to be covered in the wiki based on research.",
    )

    def __repr__(self):
        """Print a wiki with nice formatting."""
        x = "=" * 60
        x += f"\nWiki: {self.title}"
        x += f"\nTopic: {self.topic}"
        x += f"\nCategory: {self.category}"
        x += f"\nSummary: {self.summary}"
        x += f"\nSubtopics: " + "\n\t- ".join(
            [repr(subtopic) for subtopic in self.subtopics]
        )
        return x


class StructuredWikiOutput(WikiOutput):
    """
    Represents the output from the Structure Agent.
    This is a structured version of the WikiOutput with sections populated.
    """

    # remove subtopics field from StructuredWikiOutput
    subtopics: None = None
    # Inherit all fields from WikiOutput
    sections: List[SectionInitialOutput] = Field(
        default_factory=list,
        description="List of sections in the wiki page. Each section contains a title, description, and visuals.",
    )

    def __repr__(self):
        """Print a wiki with nice formatting."""
        x = "=" * 60
        x += f"\nWiki: {self.title}"
        x += f"\nTopic: {self.topic}"
        x += f"\nCategory: {self.category}"
        x += f"\nSummary: {self.summary}"
        x += f"\nSections: " + "\n\t- ".join(
            [repr(section) for section in self.sections]
        )
        return x


class FinalSectionListOutput(BaseModel):
    sections: List[SectionWithVisualCodeOutput] = Field(
        default_factory=list,
        description="List of sections in the wiki page.",
    )


# ===================
# Define State
# ===================


# class WikiState(TypedDict):
#     """
#     Represents the evolving state of the wiki creators + reviewer for a particular wiki.
#     """

#     # Wiki information
#     topic: str  # The topic of the wiki page.
#     title: str
#     text: str
#     category: str
#     subtopics: List[
#         SubtopicOutput
#     ]  # List of important subtopics to be covered in the wiki based on research
#     sections: List[SectionOutput] = []  # List of sections in the wiki page

#     def __repr__(self):
#         """Print a markdown wiki with nice formatting."""
#         x = "=" * 60
#         x += f"\nWiki: {self.title}"
#         x += f"\nTopic: {self.topic}"
#         x += f"\nCategory: {self.category}"
#         x += f"\nSubtopics: " + "\n\t".join(self.subtopics)
#         x += f"\nContent: {self.text}"
#         x += f"\n{self.text}\n"
#         x += "=" * 60
#         return x


class AgentState(TypedDict):
    """State for the agentic wiki creation process."""

    wiki: WikiOutput | StructuredWikiOutput  # The Wiki being created

    user_input: str  # Initial user input text
    search_results: List[dict] = []  # Tavily Search results returned from search tool

    user_preferences: dict = {}  # User preferences for the topic
    approved: bool = False  # Whether wiki approved by the validator AI agent
    retry_count: int = 0
    quit: bool = False  # Whether to exit the bot

    def __repr__(self):
        """Return a string representation of the AgentState."""
        x = ""
        x += f"\nWiki: {self.wiki.title}"
        x += f"\nCategory: {self.wiki.category}"
        x += f"\nUser Input: {self.user_input}"
        x += f"\nApproved: {self.approved}"
        x += f"\nRetry Count: {self.retry_count}"
        # x += f"\nSearch Results: {len(self.search_results)} results"
        return x


# ===================
# Utilities
# ===================


# ===================
# Define Nodes
# ===================


def get_user_input(state: AgentState) -> dict:
    """Gets user input for the topic of the wiki."""
    print("Welcome to the Wiki Creator Bot! Let's create a wiki together.")
    user_input = input(
        "Enter the topic you want to create a wiki for (or type 'exit' to quit): "
    ).strip()

    return {"user_input": user_input}


def route_choice(
    state: AgentState,
) -> Literal["get_user_input", "search_web_for_topic", "exit_bot"]:
    """
    Router function to determine whether to exit.
    """
    if not state["user_input"]:
        print("❌ No input provided. Please enter a topic.")
        return "get_user_input"
    elif state["user_input"].lower() == "exit":
        return "exit_bot"
    else:
        return "search_web_for_topic"


def search_web_for_topic(state: AgentState) -> dict:
    """Searches the web for the topic provided by the user."""
    results = get_web_search_tool().invoke(state["user_input"])
    return {"search_results": results}


def exit_bot(state: AgentState) -> dict:
    print("\n" + "🚪" + "=" * 58 + "🚪")
    print("    GOODBYE!")
    print("=" * 60)
    return {"quit": True}


def verify_mermaid_code(state: AgentState) -> dict:
    return NotImplementedError("Mermaid code verification is not implemented yet.")
    # try:
    #     # This will attempt to parse and potentially render the diagram
    #     Mermaid("graph TD\nA-->B")
    #     print("Mermaid code is valid.")
    # except Exception as e:
    #     print(f"Mermaid code is invalid: {e}")


# ========== Prompt Config ==========

prompt_cfg: dict = load_config(PROMPT_CONFIG_FILE_PATH)
app_cfg: dict = load_config(CONFIG_FILE_PATH)

# ========== Tool Definitions ==========


def get_web_search_tool() -> TavilySearch:
    return TavilySearch(
        max_results=5,
        search_depth="advanced",
        # include_answer=True,
        # include_raw_content=True,
        include_images=True,
    )


def get_tools():
    return [get_web_search_tool()]


# ========== Agent Node Factories ==========


def make_research_extractor_agent(research_llm: BaseChatModel):

    def extract_sub_topics_node(state: AgentState) -> dict:
        """Defines the necessary sub topics and visuals for the wiki page."""
        config = prompt_cfg["research_agent_cfg"]

        prompt_str = build_prompt_from_config(
            config,
            app_config=app_cfg,
            input_data=repr(state["search_results"]),
        )

        # TODO: use user preferences.

        response: WikiOutput = research_llm.with_structured_output(WikiOutput).invoke(
            prompt_str
        )

        # Prevent the "sections" key from being populated
        # response.sections = []  # Clear or exclude sections
        return {"wiki": response}

    return extract_sub_topics_node


def make_structure_agent(structure_llm: BaseChatModel):

    def create_sections_layout_node(state: AgentState) -> dict:
        """Defines the necessary sub topics and visuals for the wiki page."""
        config = prompt_cfg["structure_agent_layout_cfg"]

        prompt_str = build_prompt_from_config(
            config, app_config=app_cfg, input_data=repr(state["wiki"])
        )

        # TODO: use user preferences.

        response: StructuredWikiOutput = structure_llm.with_structured_output(
            StructuredWikiOutput
        ).invoke(prompt_str)
        # newstate = state["wiki"].model_copy()  # WikiOutput object
        # newstate.sections = response.sections  # modify the sections field
        return {"wiki": response}

    return create_sections_layout_node


def make_design_code_agent(design_code_llm: BaseChatModel):

    def create_design_code_node(state: AgentState) -> dict:
        """Defines the necessary sub topics and visuals for the wiki page."""
        config = prompt_cfg["design_code_agent_cfg"]

        prompt_str = build_prompt_from_config(
            config,
            app_config=app_cfg,
            input_data=repr(state["wiki"]),
        )

        # TODO: use user preferences.

        response: FinalSectionListOutput = design_code_llm.with_structured_output(
            FinalSectionListOutput
        ).invoke(prompt_str)
        newstate = state["wiki"].model_copy()  # WikiOutput object
        newstate.sections = response.sections  # modify the sections field
        return {"wiki": newstate}

    return create_design_code_node


# ========== Graph Assembly ==========


def build_graph(
    research_model_name: str,
    structure_model_name: str,
    design_code_model_name: str,
    research_model_temp: float = 0,
    structure_model_temp: float = 0,
    design_code_model_temp: float = 0,
) -> CompiledStateGraph:

    research_llm = get_llm(research_model_name, research_model_temp)
    structure_llm = get_llm(structure_model_name, structure_model_temp)
    design_code_llm = get_llm(design_code_model_name, design_code_model_temp)

    builder = StateGraph(AgentState)

    builder.add_node("get_user_input", get_user_input)
    builder.add_node("search_web_for_topic", search_web_for_topic)
    builder.add_node("exit_bot", exit_bot)
    builder.add_node(
        "researcher",
        make_research_extractor_agent(research_llm),
        # metadata={"label": "Research Agent"},
    )
    builder.add_node(
        "structurer",
        make_structure_agent(structure_llm),
        # metadata={"label": "Structure Agent"},
    )
    builder.add_node(
        "design_coder",
        make_design_code_agent(design_code_llm),
        # metadata={"label": "Design & Code Agent"},
    )
    builder.add_node(
        "verify_mermaid_code",
        verify_mermaid_code,
        metadata={"Note": "Not Implemented Yet"},
    )

    builder.set_entry_point("get_user_input")

    # decides whether to return to input or proceed to search
    builder.add_conditional_edges("get_user_input", route_choice)

    # Multi agent flow edges
    builder.add_edge("search_web_for_topic", "researcher")
    builder.add_edge("researcher", "structurer")
    builder.add_edge("structurer", "design_coder")
    builder.add_edge("design_coder", "exit_bot")
    builder.add_edge("exit_bot", END)

    return builder.compile()


# ========== Entry Point ==========


def main():
    print("\n📃 Starting joke bot with writer–critic LLM loop...")
    graph = build_graph(
        research_model_name=app_cfg["research_model_llm"],
        structure_model_name=app_cfg["structure_model_llm"],
        design_code_model_name=app_cfg["design_code_model_llm"],
    )

    # Get image representation of the graph
    print("\n📊 Generating graph...")
    graph.get_graph().draw_mermaid_png(output_file_path="wiki_creation_graph.png")
    print("\t💹 Graph image saved as 'wiki_creation_graph.png'.")

    print("\n🚀 Starting the wiki creation process...")

    final_state: dict = graph.invoke(AgentState(), config={"recursion_limit": 200})
    print("\t✅ Done.")

    # Convert the final_state dictionary to a JSON string
    final_state_json = json.dumps(final_state, indent=4, default=lambda o: o.__dict__)
    # Save final state json to a file
    with open("final_wiki_state.json", "w") as f:
        f.write(final_state_json)

    print("\n📄 Final state saved as 'final_wiki_state.json'.")

    if "wiki" in final_state:
        print("Printing Final Wiki:")
        print(final_state)
    else:
        print("No wiki was created.")


if __name__ == "__main__":
    main()
