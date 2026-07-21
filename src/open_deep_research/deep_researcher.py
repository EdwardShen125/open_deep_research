"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import re
from typing import Literal, Optional

from open_deep_research import create_configurable_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

from open_deep_research.evidence_units import EvidenceUnit, extract_numbers, EntityRef  # noqa: E402
from open_deep_research.eu_extractor import extract_from_search_results  # noqa: E402
from open_deep_research.cited_report import (  # noqa: E402
    CITED_REPORT_PROMPT, parse_cited_report, render_eu_pool,
)
from open_deep_research.verifier import verify  # noqa: E402
from open_deep_research.report_data import ReportDataObject, enforce_page_level  # noqa: E402

# Initialize a configurable model that we will use throughout the agent.
# This wrapper routes ``minimax:`` model names to our ChatMiniMax class and
# everything else to the standard chat model dispatcher. See llm.py for the
# unified LLM entry point used by all v2 code paths.
configurable_model = create_configurable_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings.
    # Plan v2 fix: tool_choice="any" forces the MiniMax-M3 model to actually
    # emit a tool_call instead of pure prose. Without this, MiniMax-M3 often
    # skips the tool layer and answers directly, which makes supervisor_tools
    # short-circuit to END via the no_tool_calls branch — leaving the
    # researcher subgraph never invoked and the EU pool empty.
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools, tool_choice="any")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        # Phase 1.5 / gap #7 — defensive read.
        # LLM occasionally omits the `reflection` field (esp. with long Chinese
        # prompts); the previous `tool_call["args"]["reflection"]` raised
        # KeyError and crashed the whole supervisor subgraph. Substitute a
        # safe placeholder and let LangGraph continue.
        args = tool_call.get("args") or {}
        reflection_content = args.get("reflection") or "(empty reflection)"
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

            # Phase 1.5 / gap #7 (defense) — fall back to a topic-shaped
            # string when the model omits the `research_topic` field. This
            # path is reached from long-prompt supervisor output where the
            # MiniMax model can drop required tool args.
            def _topic_for(tool_call):
                args = tool_call.get("args") or {}
                t = args.get("research_topic")
                if t:
                    return t
                # Last-resort: synthesize from user-provided research brief
                # in state, rather than silently drop the unit. This is
                # surfaced via the tool_message back to the supervisor so
                # the loop recovers on the next iteration.
                return (state.get("research_brief") if isinstance(state, dict) else None) \
                    or "(missing research_topic — fallback)"

            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=_topic_for(tool_call))
                    ],
                    "research_topic": _topic_for(tool_call)
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]    

            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes + evidence_units from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

            # Plan v2 — forward the EU pool collected upstream by
            # researcher_tools. Each researcher subgraph returns its own
            # EU list; supervisor aggregates across the parallel units so
            # final_report_generation sees a single unified pool.
            eus_concat: list = []
            for observation in tool_results:
                for eu in observation.get("evidence_units") or []:
                    # Avoid re-emitting identical EUs across iterations.
                    if isinstance(eu, dict):
                        key = eu.get("content_hash") or eu.get("text")
                    else:
                        key = getattr(eu, "content_hash", None) or getattr(eu, "text", None)
                    if key and any(
                        (isinstance(x, dict) and (x.get("content_hash") or x.get("text")) == key)
                        or (not isinstance(x, dict) and (getattr(x, "content_hash", None) or getattr(x, "text", None)) == key)
                        for x in eus_concat
                    ):
                        continue
                    eus_concat.append(eu)
            if eus_concat:
                update_payload["evidence_units"] = eus_concat
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # ---- Phase 2.2 EU extraction (Plan v2) ----
    # Parse each search-tool observation into structured EvidenceUnits and
    # accumulate them into `state["evidence_units"]`. The downstream
    # `compress_research` and `final_report_generation` will read from this
    # list instead of relying on the LLM-compressed `notes` only.
    # We deliberately do NOT raise on extraction failure — the researcher
    # loop continues and the LLM tool-call flow is unaffected.
    new_eus: list[EvidenceUnit] = []
    for observation, tool_call in zip(observations, tool_calls):
        try:
            if tool_call["name"] not in (
                "tavily_search", "web_search", "tavily_search_async",
            ):
                continue
            # The tavily_search tool returns a formatted string with
            # "SOURCE i: <title>" / "URL: <url>" / "SUMMARY: <text>" markers.
            new_eus.extend(_parse_tavily_observation(
                observation,
                run_id=str(state.get("research_topic", "")) or None,
            ))
        except Exception:
            continue
    if new_eus:
        # De-dup against existing state content_hash + the new ones.
        existing = list(state.get("evidence_units") or [])
        from open_deep_research.evidence_units import dedup_eus
        new_eus = dedup_eus(existing + new_eus)
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        update = {"researcher_messages": tool_outputs}
        if new_eus:
            update["evidence_units"] = new_eus
        return Command(
            goto="compress_research",
            update=update,
        )

    # Continue research loop with tool results
    update = {"researcher_messages": tool_outputs}
    if new_eus:
        update["evidence_units"] = new_eus
    return Command(
        goto="researcher",
        update=update,
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            # Phase 2.3 — forward EUs (collected upstream by researcher_tools)
            # so downstream supervisor / writer have structured citations,
            # not just the LLM-compressed prose.
            result = {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
            }
            eus = list(state.get("evidence_units") or [])
            if eus:
                result["evidence_units"] = eus
            return result

        except Exception as e:
            synthesis_attempts += 1

            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # For other errors, continue retrying
            continue

    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    result = {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }
    eus = list(state.get("evidence_units") or [])
    if eus:
        result["evidence_units"] = eus
    return result

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


def _is_transient_writer_error(e: Exception) -> bool:
    """True if `e` is a recoverable network/timeout/rate-limit error.

    Used by `final_report_generation` to decide between immediate failure
    vs. exponential-backoff retry. MiniMax-M3 + langchain-anthropic stack
    intermittently hits these under sustained load — they're not bugs in
    the graph, they're provider transient errors that benefit from retry.

    Recognised classes:
      - httpx.ReadTimeout / ConnectTimeout / WriteTimeout / PoolTimeout
      - langchain_anthropic errors with status 429 / 500 / 502 / 503 / 504
      - any exception whose message contains "timeout" / "rate limit" /
        "429" / "503" / "502" / "connection"
    """
    # Type-based detection
    type_name = type(e).__name__
    transient_types = {
        "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
        "Timeout", "ConnectionError", "RemoteProtocolError",
        "RateLimitError", "ServiceUnavailableError",
    }
    if type_name in transient_types:
        return True
    # MRO walk for httpx.Timeout / TimeoutException variants
    for cls in type(e).__mro__:
        if cls.__name__ in transient_types:
            return True
    # Message-based fallback for wrapped exceptions
    msg = (str(e) or "").lower()
    transient_markers = (
        "timeout", "timed out", "rate limit", "rate_limit",
        "429", "502", "503", "504", "connection", "temporarily unavailable",
    )
    return any(m in msg for m in transient_markers)


def _eu_attr(eu, name, default=None):
    """Pull an attribute off an EU, supporting both dataclass and dict."""
    if isinstance(eu, dict):
        return eu.get(name, default)
    return getattr(eu, name, default)


def _render_eu_digest(eu_pool, model_name: str) -> str:
    """Render a last-resort markdown digest from the EU pool.

    Used when `final_report_generation` exhausts retries on transient
    errors. The output is NOT a polished report — it's a raw evidence
    table that preserves the research data and shows the user what was
    actually found. The writer LLM can later rewrite it into prose.

    Output structure:
      - Top section: stats (count, domains, numeric anchors)
      - Per-domain grouped sections with claim + source + numbers
      - Footer: note about writer failure
    """
    if not eu_pool:
        return ""

    # Stats
    urls = set()
    domains = set()
    nums_total = 0
    for e in eu_pool:
        u = _eu_attr(e, "source_url")
        if u:
            urls.add(u)
            try:
                domains.add("/".join(u.split("/")[:3]))
            except Exception:
                pass
        ns = _eu_attr(e, "numbers") or []
        nums_total += len(ns) if isinstance(ns, list) else 0

    # Group EUs by domain
    by_domain: dict[str, list] = {}
    for e in eu_pool:
        u = _eu_attr(e, "source_url") or "(no-url)"
        try:
            d = "/".join(u.split("/")[:3])
        except Exception:
            d = "(unknown)"
        by_domain.setdefault(d, []).append(e)

    lines: list[str] = []
    title = f"# Raw Evidence Digest — {model_name or 'unknown model'}"
    lines.append(title)
    lines.append("")
    lines.append("> ⚠️ The writer LLM failed after retries; this digest preserves the "
                 "extracted evidence units. A polished report can be regenerated later "
                 "once the writer provider is healthy again.")
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    lines.append(f"- **Evidence units:** {len(eu_pool)}")
    lines.append(f"- **Unique URLs:** {len(urls)}")
    lines.append(f"- **Unique domains:** {len(domains)}")
    lines.append(f"- **Numeric anchors:** {nums_total}")
    lines.append("")

    # Per-domain sections, sorted by EU count desc, capped to top 20
    # domains to keep digest readable. Each section lists up to 50 EUs.
    sorted_domains = sorted(by_domain.items(), key=lambda kv: -len(kv[1]))[:20]
    for domain, eus in sorted_domains:
        lines.append(f"## {domain} ({len(eus)} EUs)")
        lines.append("")
        for eu in eus[:50]:
            eid = _eu_attr(eu, "id", "")
            claim = _eu_attr(eu, "claim", "") or "(no claim)"
            conf = _eu_attr(eu, "confidence", 0.0)
            source_url = _eu_attr(eu, "source_url", "")
            source_title = _eu_attr(eu, "source_title", "")
            numbers = _eu_attr(eu, "numbers") or []
            entities = _eu_attr(eu, "entities") or []

            lines.append(f"- **[{eid}]** {claim}")
            if conf:
                lines.append(f"  - confidence: `{conf:.2f}`")
            if source_title:
                lines.append(f"  - title: {source_title}")
            if source_url:
                lines.append(f"  - source: {source_url}")
            if entities:
                ent_names = sorted({str(_eu_attr(en, 'name') if isinstance(en, dict)
                                     else getattr(en, 'name', str(en)))
                                    for en in entities})
                lines.append(f"  - entities: {', '.join(ent_names)}")
            if numbers:
                num_strs = []
                for n in numbers:
                    txt = n.get("text") if isinstance(n, dict) else getattr(n, "text", "")
                    unit = n.get("unit") if isinstance(n, dict) else getattr(n, "unit", "")
                    if txt:
                        num_strs.append(f"{txt} {unit}".strip())
                if num_strs:
                    lines.append(f"  - numbers: {', '.join(num_strs[:8])}")
            lines.append("")
        if len(eus) > 50:
            lines.append(f"  _(showing 50 of {len(eus)} EUs for this domain)_")
            lines.append("")

    if len(by_domain) > 20:
        lines.append(f"## Other domains ({len(by_domain) - 20} more)")
        lines.append("")
        for d, eus in sorted(by_domain.items(), key=lambda kv: -len(kv[1]))[20:]:
            lines.append(f"- {d}: {len(eus)} EUs")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.

    Phase 2.3 / Phase 3a / Phase 3b integration:
      1. Reads structured `evidence_units` from state (Phase 2 accumulator).
      2. Constructs a chain-of-citation prompt with the EU pool.
      3. Calls LLM with a JSON-only output format.
      4. Parses the response into a `CitedReport` (Phase 2.3 parser).
      5. Runs `verifier.verify()` (Phase 3a rules 1/2/3/C) on the report.
      6. Renders to markdown and writes `final_report`.

    Failure modes are logged but the function still produces a report —
    Plan v2 evidence data is preserved in state even if the writer LLM
    misbehaves (parser warnings + verifier issues are surfaced).
    """

    # Step 1: Extract research findings and prepare state cleanup.
    cleared_state = {"notes": {"type": "override", "value": []}}
    notes = state.get("notes", [])
    findings = "\n".join(notes)
    eu_pool = list(state.get("evidence_units") or [])

    # Step 2: Configure the final report generation model.
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

    # Step 3: Attempt report generation with token limit retry logic.
    # Each attempt rebuilds the prompt with the EU pool so the LLM can
    # produce a JSON-typed chain-of-citation response.
    max_retries = 3
    current_retry = 0
    findings_token_limit: Optional[int] = None

    cited_report_dict = None
    verification_dict = None
    url_issues: list = []

    while current_retry <= max_retries:
        try:
            # Build the v2 prompt — EU pool is rendered as JSON, the rest
            # of the prompt is a strict JSON-only citation schema.
            eu_block = render_eu_pool(eu_pool) if eu_pool else "(no evidence units extracted)"
            cited_prompt = CITED_REPORT_PROMPT.format(eu_pool_block=eu_block)
            fallback_prompt_args = {
                "research_brief": state.get("research_brief", ""),
                "messages": get_buffer_string(state.get("messages", [])),
                "findings": findings or eu_block,
                "date": get_today_str(),
            }
            writer_prompt_text = cited_prompt + "\n\n" + (
                "Research brief: " + fallback_prompt_args["research_brief"] +
                "\nMessages: " + fallback_prompt_args["messages"] +
                "\nDate: " + fallback_prompt_args["date"]
            )

            # Call the writer LLM.
            final_report_msg = await configurable_model.with_config(
                writer_model_config
            ).ainvoke([HumanMessage(content=writer_prompt_text)])

            raw_response = str(final_report_msg.content)
            cited, parse_warns = parse_cited_report(raw_response)
            cited_report_dict = cited.to_dict() if cited else None

            # Rehydrate EUs: state may store them either as EvidenceUnit
            # objects (early pipeline stages) OR as dicts (cross-run
            # serialization from supervisor). Normalize before passing
            # to the verifier / Rule 4 audit.
            normalized_eus: list[EvidenceUnit] = []
            for raw in eu_pool or []:
                if isinstance(raw, EvidenceUnit):
                    normalized_eus.append(raw)
                elif isinstance(raw, dict):
                    try:
                        normalized_eus.append(EvidenceUnit.from_dict(raw))
                    except Exception:
                        continue

            # Run verifier + Rule 4 if we got a parseable cited report.
            if cited and normalized_eus:
                try:
                    v = verify(cited, normalized_eus)
                    verification_dict = v.to_dict()
                except Exception:
                    verification_dict = None

                # Rule 4 audit on the rendered report.
                try:
                    rdo = ReportDataObject(title=cited.title or "Report")
                    # Build minimal RDO rows for the audit.
                    from open_deep_research.report_data import DataRow, ReportSection
                    for sec in cited.sections:
                        rsec = rdo.add_section(heading=sec.heading)
                        for c in sec.claims:
                            source_url = ""
                            for eu in normalized_eus:
                                if eu.id in c.eu_ids:
                                    source_url = eu.source_url
                                    break
                            rsec.add_row(DataRow(
                                key=c.text[:32],
                                label=c.text[:40],
                                category="claim",
                                values={"claim": c.text},
                                source_url=source_url,
                                eu_ids=list(c.eu_ids),
                                confidence=c.confidence,
                                prose_template=c.text,
                                table_columns=["claim"],
                            ))
                    url_issues = [
                        u.to_dict() for u in enforce_page_level(rdo)
                    ]
                except Exception:
                    url_issues = []


            # Render to markdown — this *is* the final report.
            # Only switch to the structured markdown renderer when the
            # writer actually produced a parseable JSON with at least
            # one section; otherwise fall through to the legacy path
            # which faithfully reproduces the LLM prose.
            if cited and cited_report_dict and cited_report_dict.get("sections"):
                final_report_md = cited.to_markdown()
            else:
                # Fallback to legacy prose report (v1 path).
                final_report_md = str(final_report_msg.content)

            # Attach verifier notes so the failure modes are still in the
            # human-visible report (markdown comment block).
            if (verification_dict
                    and verification_dict.get("by_severity", {}).get("critical", 0) > 0):
                cs = verification_dict["by_severity"].get("critical", 0)
                final_report_md += (
                    f"\n\n<!-- verifier_warning: {cs} critical issue(s) flagged. "
                    f"Inspect state['verification'] for details. -->\n"
                )

            update = {
                "final_report": final_report_md,
                "messages": [final_report_msg],
                **cleared_state,
            }
            if cited_report_dict is not None:
                update["cited_report"] = cited_report_dict
            if verification_dict is not None:
                update["verification"] = verification_dict
            update["url_compliance"] = url_issues
            return update

        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation.
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                if current_retry == 1:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state,
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                findings = findings[:findings_token_limit] if findings_token_limit else findings
                continue
            # Handle transient network/timeout errors with exponential
            # backoff. MiniMax-M3 + langchain-anthropic occasionally
            # hangs or hits httpx.ReadTimeout under sustained load —
            # those are recoverable, unlike token-limit errors.
            elif _is_transient_writer_error(e):
                current_retry += 1
                if current_retry > max_retries:
                    # Surface the last error so callers can diagnose
                    _tb_err = type(e).__name__
                    _msg = str(e)[:400] if str(e) else "(empty error msg)"
                    # Last-ditch fallback: if we have EUs, render them
                    # as a "raw evidence digest" so the user gets the
                    # research data even when the writer LLM is down.
                    fallback_md = _render_eu_digest(eu_pool, configurable.final_report_model)
                    err_summary = f"Error generating final report after {max_retries} retries: [{_tb_err}] {_msg}"
                    return {
                        "final_report": (fallback_md + "\n\n---\n\n" + err_summary) if fallback_md else err_summary,
                        "messages": [AIMessage(content="Report generation failed after retries; rendered EU digest fallback")],
                        **cleared_state,
                    }
                import asyncio as _asyncio
                _backoff = min(2 ** current_retry, 16)
                print(f"[writer] transient error ({type(e).__name__}: {str(e)[:120]}); "
                      f"retry {current_retry}/{max_retries} in {_backoff}s")
                await _asyncio.sleep(_backoff)
                continue
            else:
                # Log full traceback so we can diagnose future failures
                import traceback as _tb
                _tb.print_exc()
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state,
                }

    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state,
    }


# =============================================================================
# Phase 2 helpers: Tavily observation → EvidenceUnit list
# =============================================================================

_TAVILY_OBS_URL_RE = re.compile(r"URL:\s*(\S+)", re.IGNORECASE)
_TAVILY_OBS_TITLE_RE = re.compile(r"SOURCE\s+\d+:\s*([^\n]+)\s*\n\s*URL:", re.IGNORECASE)
_TAVILY_OBS_SUMMARY_RE = re.compile(r"SUMMARY:\s*(.+?)(?:\n-{3,}|\Z)", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Phase 2.5 — Tavily observation noise filters
# ---------------------------------------------------------------------------
# Tavily occasionally returns source pages from completely unrelated domains
# (social media, entertainment, paywalled aggregators) and/or pages whose raw
# content is full of markdown image tokens / HTML fragments. We drop those
# BEFORE handing chunks to the EU extractor so the citation pool stays
# focused on actual research material.

# Domains that Tavily historically returns as "noise" relative to funding /
# product / company research. Matched on suffix (so `m.facebook.com/x` is
# caught by `facebook.com`). When new noise domains are observed, add them
# here — DO NOT hardcode single URLs.
_TAVILY_NOISE_DOMAIN_SUFFIXES = (
    # Social media (mostly user-generated, not authoritative)
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",  # often returns login walls + share widgets
    "reddit.com",
    "tiktok.com",
    # Entertainment / pop culture / movie blogs that bled into our queries
    "worldofreel.com",
    "throughthesilverscreen.com",
    # Marketing aggregators with shallow / fabricated company profiles
    "topstartups.io",  # returns domain-only URLs and partial data
)

# Patterns inside the raw chunk content that indicate the page wasn't
# properly summarized by Tavily (i.e. we got raw markdown/HTML rather than
# the `<summary>...</summary>` block). When matched we drop the chunk.
_NOISE_CONTENT_PATTERNS = (
    re.compile(r"!\[[^\]]*\]\([^)]+\)"),         # markdown image tokens
    re.compile(r"<img\s", re.IGNORECASE),       # raw <img> tags
    re.compile(r"<svg", re.IGNORECASE),         # inline SVG placeholders
    re.compile(r"\bdata:image/[a-z]+;base64,"),  # embedded data URIs
)

# Minimum usable content length (chars) for a source chunk. Below this we
# treat the chunk as too sparse to extract real claims from. 200 chars is
# roughly one informative sentence.
_MIN_CHUNK_CONTENT_CHARS = 200


def _host_of(url: str) -> str:
    """Extract host (no port, no scheme) from URL. Empty string on failure."""
    if not isinstance(url, str):
        return ""
    m = re.match(r"https?://([^/?#]+)", url, re.IGNORECASE)
    return (m.group(1) if m else "").lower()


def _is_noise_domain(url: str) -> bool:
    host = _host_of(url)
    if not host:
        return False
    for suf in _TAVILY_NOISE_DOMAIN_SUFFIXES:
        if host == suf or host.endswith("." + suf):
            return True
    return False


def _chunk_is_low_quality(chunk_text: str) -> bool:
    """Heuristic: chunk is mostly markdown/HTML noise rather than prose."""
    if not chunk_text or len(chunk_text.strip()) < _MIN_CHUNK_CONTENT_CHARS:
        return True
    noise_hits = sum(
        1 for pat in _NOISE_CONTENT_PATTERNS if pat.search(chunk_text)
    )
    # If 2+ different noise patterns appear, the chunk is dominated by
    # layout/asset markup — drop it.
    return noise_hits >= 2


def _filter_tavily_chunks(raws: list[dict]) -> list[dict]:
    """Drop chunks whose URL is on the noise list or whose content is junk.

    Logging-only: surfaces counts so we can tune the blacklist over time.
    """
    if not raws:
        return raws
    dropped_domain = 0
    dropped_lowq = 0
    kept: list[dict] = []
    for r in raws:
        url = r.get("url") or ""
        content = r.get("content") or ""
        if _is_noise_domain(url):
            dropped_domain += 1
            continue
        if _chunk_is_low_quality(content):
            dropped_lowq += 1
            continue
        kept.append(r)
    if dropped_domain or dropped_lowq:
        # stderr so it shows up in the langgraph dev server log without
        # requiring an import cycle through the logging module.
        print(
            f"[tavily_filter] dropped {dropped_domain} noise-domain chunks "
            f"+ {dropped_lowq} low-quality chunks; kept {len(kept)}",
            file=__import__("sys").stderr,
        )
    return kept


def _parse_tavily_observation(
    observation: str,
    *,
    run_id: Optional[str] = None,
) -> list[EvidenceUnit]:
    """Convert a Tavily tool output (formatted string) into EU list.

    Tavily tool returns text shaped like:
        SOURCE 1: <title>
        URL: <url>
        SUMMARY:
        <multi-line summary body>

        ---------------------------------------------------------------

    Pipeline:
      1. Split by the '-----------------' separator into per-source chunks
      2. Extract (title, url, summary) triples
      3. **Phase 2.5**: drop noise-domain + low-quality chunks
      4. Hand survivors to the deterministic EU extractor, which handles:
         - sentence splitting
         - numeric anchor mining
         - entity mining (CI vendor lexicon)
         - per-sentence EU with verbatim quote preserved

    Returns [] if the observation is empty or unparseable (never raises).
    """
    if not isinstance(observation, str) or not observation.strip():
        return []
    # The tool string uses '------------------' as section separator.
    chunks = re.split(r"\n-{3,}\n", observation)
    if not chunks:
        return []
    raws: list[dict] = []
    for chunk in chunks:
        url_m = _TAVILY_OBS_URL_RE.search(chunk)
        if not url_m:
            continue
        url = url_m.group(1).strip()
        title_m = _TAVILY_OBS_TITLE_RE.search(chunk)
        title = title_m.group(1).strip() if title_m else None
        sum_m = _TAVILY_OBS_SUMMARY_RE.search(chunk)
        summary = sum_m.group(1).strip() if sum_m else ""
        raws.append({
            "url": url,
            "title": title,
            "content": summary,
            "score": None,
            "provider": "tavily",
        })
    if not raws:
        # Fallback: treat the whole observation as one chunk so we don't
        # silently drop pages that have been heavily formatted by downstream
        # summarization.
        first_url = _TAVILY_OBS_URL_RE.search(observation)
        if first_url:
            raws = [{
                "url": first_url.group(1).strip(),
                "title": None,
                "content": observation,
                "score": None,
                "provider": "tavily",
            }]
    # Phase 2.5 — drop noise before extraction.
    raws = _filter_tavily_chunks(raws)
    return extract_from_search_results(raws, run_id=run_id)


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()