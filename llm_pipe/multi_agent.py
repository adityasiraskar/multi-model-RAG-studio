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


# ============================================================
# VALID TOOLS
# ============================================================

VALID_TOOLS = (
    "get_answer_apple",
    "get_answer_nvidia",
    "get_answer_amazon",
    "get_answer_tesla",
    "youtube_url",
    "web_search",
)


# ============================================================
# ROUTER
# ============================================================

def llm_route_question(
    question: str,
    chat_history: list = None,
) -> str:
    """
    Decide which tool should answer the question.

    Priority:

    1. Explicit video request -> YouTube
    2. PDF company question -> corresponding PDF
    3. Follow-up to a PDF question -> same PDF
    4. Current/external/general information -> web
    """

    history_context = ""

    if chat_history:

        recent_history = chat_history[-8:]

        history_context = (
            "RECENT CONVERSATION:\n"
        )

        for entry in recent_history:

            role = entry.get("role", "")
            content = entry.get("content", "")

            if not content:
                continue

            history_context += (
                f"{role.upper()}: "
                f"{content[:600]}\n"
            )

    else:

        history_context = (
            "RECENT CONVERSATION:\n"
            "(none)\n"
        )


    routing_prompt = f"""
You are a STRICT routing classifier for a financial
document-question-answering application.

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

web_search
    Use for current information, news, live information,
    external information, or questions outside the
    four PDF documents.

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

If the user explicitly asks for a video:

"Show me a video about Apple's products"
-> youtube_url

"Find a YouTube tutorial"
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

        if "youtube" in response:

            return "youtube_url"

        if "video" in response:

            return "youtube_url"

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

        return _infer_company_from_history(
            chat_history
        ) or "web_search"


    except Exception as exc:

        print(
            f"[ERROR] LLM routing failed: {exc}"
        )

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


    # Search newest messages first
    for entry in reversed(chat_history):

        content = str(
            entry.get("content", "")
        ).lower()

        for company, tool_name in company_tools.items():

            if company in content:

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
    ):

        self.user_input = message
        self.history = history or []


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


    def generate_answer(self):

        try:

            selected_function = llm_route_question(
                self.user_input,
                self.history,
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

        formatted_history = []


        for entry in history:

            # New Gradio message format
            if isinstance(entry, dict):

                role = entry.get("role")
                content = entry.get("content")

                if (
                    role in {"user", "assistant"}
                    and content
                ):

                    formatted_history.append(
                        {
                            "role": role,
                            "content": str(content),
                        }
                    )

                continue


            # Old tuple format
            try:

                user_msg, assistant_msg = entry

            except (
                TypeError,
                ValueError,
            ):

                continue


            if user_msg:

                formatted_history.append(
                    {
                        "role": "user",
                        "content": str(user_msg),
                    }
                )


            if assistant_msg:

                formatted_history.append(
                    {
                        "role": "assistant",
                        "content": str(assistant_msg),
                    }
                )


        # =====================================================
        # CREATE AGENT
        # =====================================================

        agent = BaseLLM(
            message=message,
            history=formatted_history,
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