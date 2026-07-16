"""Chat model for MiniMax (Anthropic-compatible endpoint).

Wraps MiniMax's Anthropic-compatible API at https://api.minimaxi.com/anthropic.
Uses httpx directly instead of the official anthropic SDK because MiniMax
expects the raw x-api-key header, which the official SDK obscures behind its
own auth flow.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr


def _safe_loads(s: str) -> Any:
    """Best-effort JSON parse; returns the raw string on failure."""
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


def _to_anthropic_messages(messages: List[BaseMessage]) -> Tuple[List[dict], List[str]]:
    """Convert LangChain messages to Anthropic API format.

    Returns (anthropic_messages, system_prompts).
    """
    system_prompts: List[str] = []
    anthropic_msgs: List[dict] = []

    for m in messages:
        if isinstance(m, SystemMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            system_prompts.append(content)
        elif isinstance(m, HumanMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            anthropic_msgs.append({"role": "user", "content": content})
        elif isinstance(m, AIMessage):
            blocks: List[dict] = []
            if m.content:
                text = m.content if isinstance(m.content, str) else str(m.content)
                blocks.append({"type": "text", "text": text})
            for tc in (m.tool_calls or []):
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc.get("args", {}),
                })
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            anthropic_msgs.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolMessage):
            anthropic_msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content if isinstance(m.content, str) else str(m.content),
                }],
            })
        else:
            # Fallback: treat as user
            content = m.content if isinstance(m.content, str) else str(m.content)
            anthropic_msgs.append({"role": "user", "content": content})

    return anthropic_msgs, system_prompts


def _parse_anthropic_response(data: dict) -> AIMessage:
    """Parse Anthropic API response into a LangChain AIMessage."""
    import json as _json
    blocks = data.get("content", [])
    text_parts: List[str] = []
    tool_calls: List[dict] = []
    openai_style_tool_calls: List[dict] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "name": block["name"],
                "args": block.get("input", {}),
            })
            # Also emit OpenAI-style tool_call so that LangChain's
            # PydanticToolsParser / JsonOutputToolsParser can read it.
            openai_style_tool_calls.append({
                "id": block["id"],
                "function": {
                    "name": block["name"],
                    "arguments": _json.dumps(block.get("input", {}), ensure_ascii=False),
                },
                "type": "function",
            })

    content = "".join(text_parts) if text_parts else ""
    usage = data.get("usage", {})
    usage_metadata = UsageMetadata(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    )

    additional_kwargs: dict = {}
    if openai_style_tool_calls:
        additional_kwargs["tool_calls"] = openai_style_tool_calls

    msg = AIMessage(
        content=content,
        usage_metadata=usage_metadata,
        additional_kwargs=additional_kwargs,
    )
    if tool_calls:
        msg.tool_calls = tool_calls
    msg.response_metadata = {
        "model": data.get("model"),
        "stop_reason": data.get("stop_reason"),
        "usage": usage,
        "request_id": data.get("id"),
    }
    return msg


class ChatMiniMax(BaseChatModel):
    """Chat model for MiniMax's Anthropic-compatible endpoint.

    Example:
        llm = ChatMiniMax(model="MiniMax-M3")
        response = llm.invoke([HumanMessage(content="hi")])
    """

    model: str = "MiniMax-M3"
    api_key: Optional[str] = Field(default=None)
    api_base: str = "https://api.minimaxi.com/anthropic"
    anthropic_version: str = "2023-06-01"
    max_tokens: int = 4096
    temperature: Optional[float] = None
    timeout: float = 120.0

    # Internal client cache
    _client: Optional[httpx.Client] = PrivateAttr(default=None)
    _async_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "minimax-anthropic"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        key = os.getenv("MINIMAX_API_KEY")
        if key:
            return key
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            return key
        raise ValueError(
            "No API key found for MiniMax. Set api_key, MINIMAX_API_KEY, "
            "or ANTHROPIC_API_KEY."
        )

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    # ------------------------------------------------------------------
    # Structured output: use function_calling (tool_use) when supported,
    # otherwise fall back to a markdown-aware JsonOutputParser. MiniMax's
    # Anthropic-compatible endpoint sometimes returns plain ```json```
    # markdown instead of a tool_use block even with tool_choice=any, so we
    # chain the model with a robust parser instead of relying on tool calls.
    # ------------------------------------------------------------------
    def with_structured_output(
        self,
        schema,
        *,
        include_raw: bool = False,
        method: str = "json_schema",  # default to json_schema for stability
        **kwargs,
    ):
        """Return a Runnable that produces output matching ``schema``.

        Implements the LangChain standard interface. We always use the
        "json_schema" path (system-prompt + markdown-JSON parser) because the
        function_calling path on MiniMax is flaky.
        """
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import Runnable, RunnablePassthrough
        from langchain_core.utils.function_calling import convert_to_openai_tool
        from operator import itemgetter

        # Build a description of the schema to put in the system prompt
        if isinstance(schema, type):
            # Pydantic class
            try:
                spec = convert_to_openai_tool(schema)
                # Use the JSON schema for "parameters" (which the model fills in)
                params_json = json.dumps(
                    spec["function"].get("parameters", {}), ensure_ascii=False
                )
                schema_description = (
                    "You MUST respond with a single JSON object that matches this schema:\n"
                    f"{params_json}\n\n"
                    "Rules:\n"
                    '- Wrap the JSON in ```json ... ``` code fences OR use the OpenAI-style wrapper {"type":"function","function":{"name":"<any>","parameters":{...}}}.\n'
                    "- Do not write any text outside the JSON.\n"
                    "- Populate all required fields with values that answer the user's request."
                )
                target_cls = schema
            except Exception:
                schema_description = f"Return ONLY valid JSON matching schema: {schema.__name__}"
                target_cls = None
        elif isinstance(schema, dict):
            schema_description = (
                "You MUST respond with a single JSON object that matches this schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n\n"
                "Rules:\n"
                "- Wrap the JSON in ```json ... ``` code fences OR use the OpenAI-style wrapper.\n"
                "- Do not write any text outside the JSON."
            )
            target_cls = None
        else:
            schema_description = "Return ONLY valid JSON."
            target_cls = None

        # Bind a small system prompt that tells the model to output JSON
        class _StructuredRunnable(Runnable):
            def __init__(self, model, sys_prompt, parser, target_cls, include_raw):
                self._model = model
                self._sys = sys_prompt
                self._parser = parser
                self._target_cls = target_cls
                self._include_raw = include_raw

            def invoke(self, input, config=None, **kw):
                from langchain_core.messages import SystemMessage
                msgs = self._inject_system(input)
                raw = self._model.invoke(msgs, config=config, **kw)
                if self._include_raw:
                    parsed = self._safe_parse(raw)
                    return {"raw": raw, "parsed": parsed, "parsing_error": None}
                return self._safe_parse(raw)

            async def ainvoke(self, input, config=None, **kw):
                from langchain_core.messages import SystemMessage
                msgs = self._inject_system(input)
                raw = await self._model.ainvoke(msgs, config=config, **kw)
                if self._include_raw:
                    parsed = self._safe_parse(raw)
                    return {"raw": raw, "parsed": parsed, "parsing_error": None}
                return self._safe_parse(raw)

            def _inject_system(self, input):
                from langchain_core.messages import SystemMessage
                if isinstance(input, list):
                    # Check if first is system; insert at top
                    if input and getattr(input[0], "type", None) == "system":
                        return input
                    return [SystemMessage(content=self._sys)] + list(input)
                # String input or single message: prepend system as messages
                from langchain_core.messages import HumanMessage
                if isinstance(input, str):
                    return [
                        SystemMessage(content=self._sys),
                        HumanMessage(content=input),
                    ]
                return [SystemMessage(content=self._sys), input]

            def _safe_parse(self, raw):
                # Strategy 1: model returned real tool_calls (LangChain style)
                try:
                    tool_calls = (
                        getattr(raw, "tool_calls", None)
                        or raw.additional_kwargs.get("tool_calls")
                    )
                    if tool_calls:
                        first = tool_calls[0]
                        if isinstance(first, dict):
                            args = first.get("args")
                            if args is None and "function" in first:
                                raw_args = first["function"].get("arguments", "{}")
                                args = _safe_loads(raw_args)
                            if isinstance(args, dict):
                                if self._target_cls:
                                    return self._target_cls(**args)
                                return args
                except Exception:
                    pass

                # Strategy 2: model wrote an OpenAI-style "function call"
                # representation in plain text (very common with MiniMax-M3):
                #   {"type":"function","function":{"name":"X","parameters":{...}}}
                try:
                    content_text = raw.content if isinstance(raw.content, str) else ""
                    if content_text and '"function"' in content_text and '"parameters"' in content_text:
                        wrapper = _safe_loads(content_text)
                        if isinstance(wrapper, dict) and "function" in wrapper:
                            params = wrapper["function"].get("parameters", {})
                            if isinstance(params, dict):
                                if self._target_cls:
                                    return self._target_cls(**params)
                                return params
                except Exception:
                    pass

                # Strategy 3: JsonOutputParser for markdown ```json``` blocks
                try:
                    parsed = self._parser.invoke(raw)
                    if self._target_cls and isinstance(parsed, dict):
                        return self._target_cls(**parsed)
                    return parsed
                except Exception as e:
                    if self._include_raw:
                        return None
                    raise

            def __repr__(self):
                return f"ChatMiniMax.with_structured_output(target={self._target_cls})"

        return _StructuredRunnable(
            model=self,
            sys_prompt=schema_description,
            parser=JsonOutputParser(),
            target_cls=target_cls if isinstance(schema, type) else None,
            include_raw=include_raw,
        )

    def _build_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        anthropic_msgs, system_prompts = _to_anthropic_messages(messages)
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_msgs,
        }
        if system_prompts:
            payload["system"] = "\n\n".join(system_prompts)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if stop:
            payload["stop_sequences"] = stop

        if "tools" in kwargs and kwargs["tools"]:
            payload["tools"] = kwargs["tools"]
        elif hasattr(self, "_bound_tools") and self._bound_tools:
            payload["tools"] = self._bound_tools

        if "tool_choice" in kwargs:
            payload["tool_choice"] = kwargs["tool_choice"]
        elif hasattr(self, "_bound_tool_choice") and self._bound_tool_choice:
            # LangChain passes ``tool_choice="any"`` for with_structured_output;
            # convert that to Anthropic's expected shape.
            choice = self._bound_tool_choice
            if choice == "any":
                payload["tool_choice"] = {"type": "any"}
            elif choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif isinstance(choice, str):
                payload["tool_choice"] = {"type": "tool", "name": choice}
            else:
                payload["tool_choice"] = choice

        return payload

    def _completion(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "x-api-key": self._resolved_api_key(),
        }
        url = f"{self.api_base.rstrip('/')}/v1/messages"
        client = self._get_client()
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"MiniMax API error {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()

    async def _acompletion(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "x-api-key": self._resolved_api_key(),
        }
        url = f"{self.api_base.rstrip('/')}/v1/messages"
        client = self._get_async_client()
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"MiniMax API error {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop, **kwargs)
        data = self._completion(payload)
        ai_msg = _parse_anthropic_response(data)
        gen = ChatGeneration(message=ai_msg)
        return ChatResult(generations=[gen])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop, **kwargs)
        data = await self._acompletion(payload)
        ai_msg = _parse_anthropic_response(data)
        gen = ChatGeneration(message=ai_msg)
        return ChatResult(generations=[gen])

    def bind_tools(self, tools, **kwargs):
        """Bind tools (LangChain standard interface).

        Accepts a list of either LangChain tool objects or Pydantic classes /
        dicts (the latter is what ``with_structured_output`` passes). Also
        persists ``tool_choice`` and any other binding kwargs onto the new
        instance so ``_build_payload`` can apply them.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool
        from pydantic import BaseModel

        anthropic_tools = []
        for t in tools:
            if isinstance(t, type) and issubclass(t, BaseModel):
                # Pydantic schema (used by with_structured_output)
                spec = convert_to_openai_tool(t)
                anthropic_tools.append({
                    "name": spec["function"]["name"],
                    "description": spec["function"].get(
                        "description",
                        t.__doc__ or spec["function"]["name"],
                    ),
                    "input_schema": spec["function"]["parameters"],
                })
            elif isinstance(t, dict):
                if "function" in t:  # OpenAI-style already
                    anthropic_tools.append({
                        "name": t["function"]["name"],
                        "description": t["function"].get("description", ""),
                        "input_schema": t["function"].get("parameters", {}),
                    })
                else:
                    anthropic_tools.append(t)
            elif hasattr(t, "name") and hasattr(t, "description"):
                # LangChain tool object
                schema = getattr(t, "args_schema", None)
                if schema:
                    input_schema = schema.model_json_schema()
                else:
                    input_schema = {"type": "object", "properties": {}}
                anthropic_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": input_schema,
                })
            else:
                raise TypeError(f"Unsupported tool type: {type(t).__name__}")

        new = self.__class__(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            anthropic_version=self.anthropic_version,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        new._bound_tools = anthropic_tools
        for k, v in kwargs.items():
            setattr(new, f"_bound_{k}", v)
        return new

    # ---------------- streaming -----------------
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop=stop, **kwargs)
        if hasattr(self, "_bound_tools") and self._bound_tools:
            payload["tools"] = self._bound_tools
        payload["stream"] = True

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "x-api-key": self._resolved_api_key(),
        }
        url = f"{self.api_base.rstrip('/')}/v1/messages"

        with self._get_client().stream("POST", url, headers=headers, json=payload) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                ev_type = event.get("type")
                if ev_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=text)
                        )
                        if run_manager:
                            run_manager.on_llm_new_token(text)
                        yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop=stop, **kwargs)
        if hasattr(self, "_bound_tools") and self._bound_tools:
            payload["tools"] = self._bound_tools
        payload["stream"] = True

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "x-api-key": self._resolved_api_key(),
        }
        url = f"{self.api_base.rstrip('/')}/v1/messages"

        async with self._get_async_client().stream(
            "POST", url, headers=headers, json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                ev_type = event.get("type")
                if ev_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=text)
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(text)
                        yield chunk