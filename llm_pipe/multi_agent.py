import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_pipe.multi_agent_tools import (
    query_openai,
    get_answer_apple,
    get_answer_nvidia,
    get_answer_amazon,
    get_answer_tesla,
    youtube_url,
    web_search,
)

from config import app_config
from llm_pipe.sql_agent import process_query_sql


# ============================================================
# VALID TOOLS
# ============================================================

VALID_TOOLS = (
    "get_answer_apple",
    "get_answer_nvidia",
    "get_answer_amazon",
    "get_answer_tesla",
    "youtube_url",
    "query_csv",
    "web_search",
)


# ============================================================
# LANGCHAIN CHAT HISTORY NORMALIZATION
# ============================================================


def _message_content(message: BaseMessage) -> str:
    """Return plain text for LangChain message content."""
    content = message.content
    return content if isinstance(content, str) else str(content)


def normalize_chat_history(history=None) -> list[BaseMessage]:
    """
    Convert Gradio dictionary/tuple history and LangChain messages into a
    consistent LangChain message list for routing and prompt construction.
    """
    normalized: list[BaseMessage] = []

    for entry in history or []:
        if isinstance(entry, BaseMessage):
            normalized.append(entry)
            continue

        if isinstance(entry, dict):
            role = str(entry.get("role", "")).lower()
            content = entry.get("content")
            if content:
                if role == "user":
                    normalized.append(HumanMessage(content=str(content)))
                elif role == "assistant":
                    normalized.append(AIMessage(content=str(content)))
            continue

        try:
            user_message, assistant_message = entry
        except (TypeError, ValueError):
            continue

        if user_message:
            normalized.append(HumanMessage(content=str(user_message)))
        if assistant_message:
            normalized.append(AIMessage(content=str(assistant_message)))

    return normalized


def format_history_context(history=None, max_messages: int = 8) -> str:
    """Build a bounded, readable history section for the routing prompt."""
    messages = normalize_chat_history(history)[-max_messages:]
    if not messages:
        return "RECENT CONVERSATION:\n(none)\n"

    lines = ["RECENT CONVERSATION:"]
    for message in messages:
        role = "USER" if isinstance(message, HumanMessage) else "ASSISTANT"
        content = _message_content(message).strip()
        if content:
            lines.append(f"{role}: {content[:600]}")

    return "\n".join(lines) + "\n"


# ============================================================
# ROUTER
# ============================================================

def llm_route_question(
    question: str,
    chat_history: list = None,
    has_csv: bool = False,
) -> str:
    """
    Decide which tool should answer the question.

    Priority:

    1. Explicit video request -> YouTube
    2. PDF company question -> corresponding PDF
    3. Follow-up to a PDF question -> same PDF
    4. Current/external/general information -> web
    """

    # Deterministic routing guards run before the classifier. This makes
    # URL/CSV intent reliable even if a small local model formats its answer badly.
    lowered = question.lower().strip()
    youtube_url_match = re.search(
        r"https?://[^\s]*(?:youtube\.com|youtu\.be)[^\s]*",
        question,
        flags=re.I,
    )
    if youtube_url_match:
        return "youtube_url"

    # An uploaded CSV must take precedence over vague keyword/LLM routing.
    # This makes arbitrary natural-language data questions such as
    # "give me the insights" route to the SQL tool reliably.
    if has_csv:
        return "query_csv"

    csv_intent = any(term in lowered for term in (
        "csv", "spreadsheet", "dataset", "column", "row",
        "average", "sum", "count", "maximum", "minimum",
        "group by", "top ", "sort ", "filter ", "show me the data",
    ))

    company_tool_guards = {
        "apple": "get_answer_apple",
        "nvidia": "get_answer_nvidia",
        "amazon": "get_answer_amazon",
        "tesla": "get_answer_tesla",
    }
    for company, tool_name in company_tool_guards.items():
        if re.search(rf"\b{company}\b", lowered):
            return tool_name

    history_context = format_history_context(chat_history, max_messages=8)


    input_context = (
        "ATTACHED INPUTS:\n"
        f"- CSV file: {'YES' if has_csv else 'NO'}\n"
        f"- YouTube URL: {'YES' if youtube_url_match else 'NO'}\n"
    )

    routing_prompt = f"""
You are a STRICT routing classifier for a financial
document-question-answering application.

{input_context}

ROUTING PRIORITY:
1. A YouTube URL wins unless the user explicitly asks for a finance PDF or CSV action.
2. If a CSV is attached and the question is about its rows, columns, values, totals,
   averages, sorting, filtering, grouping, or comparisons, route to query_csv.
3. An explicitly named PDF company or a clear follow-up about the four reports routes to that PDF.
4. Current/external/news questions with no stronger PDF, CSV, or YouTube intent route to web_search.

Your ONLY job is to select exactly ONE tool.

AVAILABLE TOOLS:

get_answer_apple
    Use ONLY for questions that should be answered from
    the Apple PDF/10-K.

get_answer_nvidia
    Use ONLY for questions that should be answered from
    the NVIDIA PDF/10-K.

get_answer_amazon
    Use ONLY for questions that should be answered from
    the Amazon PDF/10-K.

get_answer_tesla
    Use ONLY for questions that should be answered from
    the Tesla PDF/10-K.

youtube_url
    Use ONLY when the user explicitly asks for a video,
    YouTube video, tutorial video, demonstration, or
    something to watch.

query_csv
    Use when a CSV file is attached and the user asks to
    analyze, query, filter, aggregate, compare, or summarize the data.

web_search
    Use for current information, news, live information,
    external information, or questions outside the four PDF documents
    and without a stronger attached-file or URL intent.

============================================================
CRITICAL PDF ROUTING RULE
============================================================

If the question can reasonably be answered from one of
the four PDF documents, ROUTE TO THAT PDF.

DO NOT route a PDF question to web_search simply because
the internet could also answer it.

PDF ROUTING HAS PRIORITY OVER WEB SEARCH.

============================================================
FOLLOW-UP QUESTIONS
============================================================

Use conversation history.

If the previous conversation was about one of the four
PDF companies and the current question is a follow-up,
KEEP USING THE SAME PDF.

Examples:

Previous:
"What was Apple's revenue?"

Current:
"What about 2021?"

-> get_answer_apple


Previous:
"Tell me about Tesla's operating expenses."

Current:
"What was the percentage change?"

-> get_answer_tesla


Previous:
"How did NVIDIA's data center revenue perform?"

Current:
"What were the main reasons?"

-> get_answer_nvidia


Previous:
"What was Amazon's operating income?"

Current:
"Which year was higher?"

-> get_answer_amazon

============================================================
COMPANY ROUTING
============================================================

Apple -> get_answer_apple

NVIDIA -> get_answer_nvidia

Amazon -> get_answer_amazon

Tesla -> get_answer_tesla

============================================================
CSV ROUTING
============================================================
If a CSV file is attached, prefer query_csv when the user asks to
query, analyze, filter, compare, aggregate, calculate, count, sort,
visualize, summarize, or inspect rows and columns in the attached data.
Do not route the same question to a PDF solely because it names a company.
An attached CSV is the strongest explicit data intent.

Examples:
"What is the average sales value?"
-> query_csv

"Show the top 5 rows by revenue"
-> query_csv

============================================================
WEB ROUTING
============================================================
Use web_search when:
- the user explicitly asks for latest/current information
- the user asks about current stock price
- the user asks about current news
- the user asks about something outside the PDFs
- the question is unrelated to the four PDF companies
- no relevant PDF company can be established

Examples:

"What is Apple's stock price today?"
-> web_search

"What happened in NVIDIA news today?"
-> web_search

"What is the weather in Delhi?"
-> web_search

============================================================
YOUTUBE ROUTING
============================================================
Route to youtube_url when:

- the message contains a youtube.com or youtu.be URL
- the user asks to watch, find, search, analyze, summarize, or answer a video
- the message contains a YouTube URL and a question; analyze the transcript
- the message contains only a YouTube URL; summarize it as the fallback behavior

Examples:

"https://youtube.com/watch?v=... What did the speaker say about revenue?"
-> youtube_url

"https://youtu.be/..."
-> youtube_url

"Find YouTube tutorials on financial analysis"
-> youtube_url
============================================================
IMPORTANT
============================================================

DO NOT answer the question.

DO NOT explain your decision.

Return ONLY the tool name.

{history_context}

CURRENT QUESTION:
{question}

TOOL:
"""


    try:

        response = query_openai(
            routing_prompt,
            temperature=0.0,
            max_tokens=20,
        )

        response = (
            response
            .strip()
            .lower()
            .replace("`", "")
        )

        print(
            f"[ROUTER LLM RESPONSE] {response}"
        )


        # ====================================================
        # EXACT MATCH
        # ====================================================

        if response in VALID_TOOLS:

            return response


        # ====================================================
        # KEYWORD FALLBACK
        # ====================================================

        if "youtube" in response or "youtu.be" in response:

            return "youtube_url"

        if "video" in response:

            return "youtube_url"

        if has_csv and ("query_csv" in response or "csv" in response or "sql" in response):

            return "query_csv"

        if "apple" in response:

            return "get_answer_apple"

        if "nvidia" in response:

            return "get_answer_nvidia"

        if "amazon" in response:

            return "get_answer_amazon"

        if "tesla" in response:

            return "get_answer_tesla"


        # ====================================================
        # IMPORTANT FALLBACK
        # ====================================================

        print(
            "[WARNING] Router returned an unclear response."
        )

        print(
            "[WARNING] Falling back to document routing."
        )

        if has_csv and csv_intent:
            return "query_csv"
        return _infer_company_from_history(
            chat_history
        ) or "web_search"


    except Exception as exc:

        print(
            f"[ERROR] LLM routing failed: {exc}"
        )

        if has_csv and csv_intent:
            return "query_csv"
        return (
            _infer_company_from_history(
                chat_history
            )
            or "web_search"
        )


# ============================================================
# HISTORY-BASED COMPANY RESOLUTION
# ============================================================

def _infer_company_from_history(
    chat_history: list = None,
) -> str | None:
    """
    Resolve the most recently discussed PDF company
    from conversation history.

    Returns a tool name.
    """

    if not chat_history:

        return None

    company_tools = {
        "apple": "get_answer_apple",
        "nvidia": "get_answer_nvidia",
        "amazon": "get_answer_amazon",
        "tesla": "get_answer_tesla",
    }


    # Search newest messages first, using normalized LangChain messages.
    for message in reversed(normalize_chat_history(chat_history)):
        content = _message_content(message).lower()

        for company, tool_name in company_tools.items():
            if re.search(rf"\b{company}\b", content):
                return tool_name

    return None


# ============================================================
# BASE LLM
# ============================================================

class BaseLLM:

    def __init__(
        self,
        message: str,
        history,
        csv_file=None,
    ):

        self.user_input = message
        self.history = history or []
        self.csv_file = csv_file


        self.function_map = {

            "get_answer_apple":
                get_answer_apple,

            "get_answer_nvidia":
                get_answer_nvidia,

            "get_answer_amazon":
                get_answer_amazon,

            "get_answer_tesla":
                get_answer_tesla,

            "youtube_url":
                youtube_url,

            "web_search":
                web_search,
        }

        self.function_map["query_csv"] = self._query_csv


    def _query_csv(self, question: str) -> str:
        if self.csv_file is None:
            return "A CSV file was not attached. Please upload one in the CSV input area."

        status, generated_sql, result = process_query_sql(
            self.csv_file,
            question,
            app_config.OLLAMA_MODEL,
            True,
        )

        if result is not None and hasattr(result, "empty") and not result.empty:
            table = result.to_string(index=False)
        elif result is not None:
            table = "The query returned no rows."
        else:
            table = "The query did not return tabular data."

        return (
            f"{status}\n\n"
            f"Generated SQL:\n```sql\n{generated_sql}\n```\n\n"
            f"Result:\n{table}"
        )


    def generate_answer(self):

        try:

            selected_function = llm_route_question(
                self.user_input,
                self.history,
                has_csv=self.csv_file is not None,
            )


            # =================================================
            # DEBUGGING
            # =================================================

            print("\n" + "=" * 60)

            print(
                f"[ROUTING] Question: "
                f"{self.user_input}"
            )

            print(
                f"[ROUTING] History: "
                f"{len(self.history)} messages"
            )

            print(
                f"[ROUTING] Selected Tool: "
                f"{selected_function}"
            )

            print("=" * 60)


            # =================================================
            # EXECUTE TOOL
            # =================================================

            if selected_function in self.function_map:

                func = self.function_map[
                    selected_function
                ]

                print(
                    f"[EXECUTION] Calling "
                    f"{selected_function}..."
                )


                if selected_function == "query_csv":
                    answer = func(self.user_input)
                else:
                    answer = func.invoke(
                        self.user_input
                    )


                print(
                    f"[EXECUTION] "
                    f"{selected_function} completed."
                )

                print(
                    f"[RESPONSE] Length: "
                    f"{len(answer)} characters"
                )


                return {
                    "output": answer,
                    "intermediate_steps":
                        f"LLM routed to "
                        f"{selected_function}",
                }


            # =================================================
            # SAFETY FALLBACK
            # =================================================

            print(
                "[FALLBACK] No valid tool selected."
            )

            response = query_openai(
                self.user_input
            )

            return {
                "output": response,
                "intermediate_steps":
                    "Direct OpenAI response",
            }


        except Exception as exc:

            print(
                f"[ERROR] Exception in "
                f"generate_answer: {exc}"
            )

            return {
                "output":
                    "I encountered an error while "
                    "processing your question: "
                    f"{exc}",

                "intermediate_steps":
                    f"Error: {exc}",
            }


# ============================================================
# GRADIO INVOKE
# ============================================================

def invoke(
    message,
    history,
    csv_file=None,
):

    try:

        print(
            f"\n[INVOKE] New request: "
            f"{message}"
        )


        history = history or []


        # =====================================================
        # NORMALIZE HISTORY
        # =====================================================

        formatted_history = normalize_chat_history(history)


        # =====================================================
        # CREATE AGENT
        # =====================================================

        agent = BaseLLM(
            message=message,
            history=formatted_history,
            csv_file=csv_file,
        )


        # =====================================================
        # GENERATE
        # =====================================================

        output = agent.generate_answer()

        main_output = str(
            output["output"]
        )

        intermediate_steps = str(
            output["intermediate_steps"]
        )


        # =====================================================
        # UPDATE HISTORY
        # =====================================================

        updated_history = [
            *history,

            {
                "role": "user",
                "content": message,
            },

            {
                "role": "assistant",
                "content": main_output,
            },
        ]


        print(
            "[INVOKE] Response generated successfully."
        )


        return (
            updated_history,
            intermediate_steps,
        )


    except Exception as exc:

        print(
            f"[INVOKE ERROR] {exc}"
        )

        error_msg = (
            "Sorry, I encountered an error: "
            f"{exc}"
        )


        history = history or []


        updated_history = [
            *history,

            {
                "role": "user",
                "content": message,
            },

            {
                "role": "assistant",
                "content": error_msg,
            },
        ]


        return (
            updated_history,
            f"Error in invoke function: {exc}",
        )