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


class Joke(BaseModel):
    text: str
    category: str


class JokeState(BaseModel):
    """
    Represents the evolving state of the joke bot.
    """

    jokes: Annotated[List[Joke], add] = []  # Using built-in add operator
    jokes_choice: Literal["n", "c", "q"] = "n"  # next, change, quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False


# ========== Extended State ==========


class AgenticJokeState(JokeState):
    latest_joke: str = ""
    approved: bool = False
    retry_count: int = 0


# ===================
# Utilities
# ===================


def get_user_input(prompt: str) -> str:
    return input(prompt).strip().lower()


def print_joke(joke: Joke):
    """Print a joke with nice formatting."""
    # print(f"\n📂 CATEGORY: {joke.category.upper()}\n")
    print(f"\n😂 {joke.text}\n")
    print("=" * 60)


def print_menu_header(category: str, total_jokes: int):
    """Print a compact menu header."""
    print(f"🎭 Menu | Category: {category.upper()} | Jokes: {total_jokes}")
    print("-" * 50)


# ===================
# Define Nodes
# ===================


def show_menu(state: JokeState) -> dict:
    print_menu_header(state.category, len(state.jokes))
    print("Pick an option:")
    user_input = get_user_input(
        "[n] 🎭 Next Joke  [c] 📂 Change Category  [q] 🚪 Quit\nUser Input: "
    )
    while user_input not in ["n", "c", "q"]:
        print("❌ Invalid input. Please try again.")
        user_input = get_user_input(
            "[n] 🎭 Next Joke  [c] 📂 Change Category  [q] 🚪 Quit\n    User Input: "
        )
    return {"jokes_choice": user_input}


def exit_bot(state: JokeState) -> dict:
    print("\n" + "🚪" + "=" * 58 + "🚪")
    print("    GOODBYE!")
    print("=" * 60)
    return {"quit": True}


def route_choice(state: JokeState) -> str:
    """
    Router function to determine the next node based on user choice.
    Keys must match the target node names.
    """
    if state.jokes_choice == "n":
        return "fetch_joke"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "q":
        return "exit_bot"
    else:
        return "exit_bot"


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
