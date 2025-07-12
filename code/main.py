from operator import add
from typing import Annotated, List, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from llm import get_llm
from paths import PROMPT_CONFIG_FILE_PATH
from prompt_builder import build_prompt_from_config
from pydantic import BaseModel
from pyjokes import get_joke
from utils import load_config

# ===================
# Define State
# ===================


class Wiki(BaseModel):
    topic: str  # topic of the wiki page
    title: str  # title of the wiki page
    text: str  # markdown text
    category: str  # category of the topic (what domain the topic belongs to)
    user_preferences: (
        dict  # user preferences for the topic (e.g., language, tone, etc.)
    )


class WikiState(BaseModel):
    """
    Represents the evolving state of the wiki creators + reviewer for a particular wiki.
    """

    wiki: Wiki = None  # Current wiki page
    user_input: str = ""  # User input text
    user_preferences: dict = {}  # User preferences for the topic
    approved: bool = False  # Whether approved by the validator AI agent
    retry_count: int = 0
    quit: bool = False  # Whether to exit the bot


# ===================
# Utilities
# ===================


def print_wiki(wiki: Wiki):
    """Print a markdown wiki with nice formatting."""
    print(f"\nWiki: {wiki.title}\n")
    print(f"\nTopic: {wiki.topic}\n")
    print(f"\nCategory: {wiki.category}\n")
    print(f"\nContent: {wiki.text}\n")
    print(f"\n{wiki.text}\n")
    print("=" * 60)


# ===================
# Define Structured Output Classes for Agents
# ===================
# TODO This part next.


# ===================
# Define Nodes
# ===================


def research_topic(state: WikiState) -> dict:
    """Gathers key concepts and visuals needed using WebSearch."""
    return NotImplementedError("This agent is not implemented yet.")


# ========== Prompt Config ==========

prompt_cfg = load_config(PROMPT_CONFIG_FILE_PATH)

# ========== Writer–Critic Node Factories ==========


def make_writer_node(writer_llm):
    def writer_node(state: AgenticJokeState) -> dict:
        config = prompt_cfg["joke_writer_cfg"]
        prompt = build_prompt_from_config(config, input_data="", app_config=None)
        prompt += f"\\n\\nThe category is: {state.category}"
        response = writer_llm.invoke(prompt)
        return {"latest_joke": response.content}

    return writer_node


def make_critic_node(critic_llm):
    def critic_node(state: AgenticJokeState) -> dict:
        config = prompt_cfg["joke_critic_cfg"]
        prompt = build_prompt_from_config(
            config, input_data=state.latest_joke, app_config=None
        )
        decision = critic_llm.invoke(prompt).content.strip().lower()
        approved = "yes" in decision
        return {"approved": approved, "retry_count": state.retry_count + 1}

    return critic_node


def show_final_joke(state: AgenticJokeState) -> dict:
    joke = Joke(text=state.latest_joke, category=state.category)
    print_joke(joke)
    return {"jokes": [joke], "retry_count": 0, "approved": False, "latest_joke": ""}


def writer_critic_router(state: AgenticJokeState) -> str:
    if state.approved or state.retry_count >= 5:
        return "show_final_joke"
    return "writer"


def update_category(state: AgenticJokeState) -> dict:
    categories = ["dad developer", "chuck norris developer", "general"]
    emoji_map = {
        "knock-knock": "🚪",
        "dad developer": "👨‍💻",
        "chuck norris developer": "🥋",
        "general": "🎯",
    }

    print("📂" + "=" * 58 + "📂")
    print("    CATEGORY SELECTION")
    print("=" * 60)

    for i, cat in enumerate(categories):
        emoji = emoji_map.get(cat, "📂")
        print(f"    {i}. {emoji} {cat.upper()}")

    print("=" * 60)

    try:
        selection = int(input("    Enter category number: ").strip())
        if 0 <= selection < len(categories):
            selected_category = categories[selection]
            print(f"    ✅ Category changed to: {selected_category.upper()}")
            return {"category": selected_category}
        else:
            print("    ❌ Invalid choice. Keeping current category.")
            return {}
    except ValueError:
        print("    ❌ Please enter a valid number. Keeping current category.")
        return {}


# ========== Graph Assembly ==========


def build_joke_graph(
    writer_model: str = "gpt-4o-mini",
    critic_model: str = "gpt-4o-mini",
    writer_temp: float = 0.95,
    critic_temp: float = 0.1,
) -> CompiledStateGraph:

    writer_llm = get_llm(writer_model, writer_temp)
    critic_llm = get_llm(critic_model, critic_temp)

    builder = StateGraph(AgenticJokeState)

    builder.add_node("show_menu", show_menu)
    builder.add_node("update_category", update_category)
    builder.add_node("exit_bot", exit_bot)
    builder.add_node("writer", make_writer_node(writer_llm))
    builder.add_node("critic", make_critic_node(critic_llm))
    builder.add_node("show_final_joke", show_final_joke)

    builder.set_entry_point("show_menu")

    builder.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "fetch_joke": "writer",
            "update_category": "update_category",
            "exit_bot": "exit_bot",
        },
    )

    builder.add_edge("update_category", "show_menu")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        writer_critic_router,
        {"writer": "writer", "show_final_joke": "show_final_joke"},
    )
    builder.add_edge("show_final_joke", "show_menu")
    builder.add_edge("exit_bot", END)

    return builder.compile()


# ========== Entry Point ==========


def main():
    print("\n🎭 Starting joke bot with writer–critic LLM loop...")
    graph = build_joke_graph(writer_temp=0.8, critic_temp=0.1)
    final_state = graph.invoke(
        AgenticJokeState(category="dad developer"), config={"recursion_limit": 200}
    )
    print("\n✅ Done. Final Joke Count:", len(final_state["jokes"]))


if __name__ == "__main__":
    main()
