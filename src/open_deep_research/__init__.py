"""open_deep_research package.

Routes ``minimax:`` model names to :class:`ChatMiniMax` and everything else
to langchain's default ``init_chat_model`` dispatcher.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig

from open_deep_research.minimax_chat import ChatMiniMax


def _resolve_chat_model(config):
    """Pick the right chat model based on config['model']."""
    cfg = (config or {}).get("configurable", {}) or {}
    model_name = cfg.get("model", "")
    max_tokens = cfg.get("max_tokens")
    api_key = cfg.get("api_key")
    if model_name.startswith("minimax:"):
        kwargs = {"model": model_name.split(":", 1)[1]}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if api_key:
            kwargs["api_key"] = api_key
        return ChatMiniMax(**kwargs)
    # Fallback: build a ChatModel that uses the standard init_chat_model path.
    from langchain.chat_models import init_chat_model
    kwargs = {}
    if model_name:
        kwargs["model"] = model_name
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if api_key:
        kwargs["api_key"] = api_key
    return init_chat_model(**kwargs)


class _ConfigurableModel(Runnable):
    """A runnable wrapper that dispatches to the right chat model at invoke time.

    Methods like ``bind_tools`` and ``with_structured_output`` are forwarded to
    the resolved model lazily so the user's config (``configurable.model``)
    is honoured.
    """

    def __init__(self):
        super().__init__()

    # ---- core runnable protocol (all forward to resolved model) ----
    def invoke(self, input, config: RunnableConfig = None, **kwargs):
        return _resolve_chat_model(config).invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config: RunnableConfig = None, **kwargs):
        return await _resolve_chat_model(config).ainvoke(
            input, config=config, **kwargs
        )

    def stream(self, input, config: RunnableConfig = None, **kwargs):
        return _resolve_chat_model(config).stream(input, config=config, **kwargs)

    async def astream(self, input, config: RunnableConfig = None, **kwargs):
        async for chunk in _resolve_chat_model(config).astream(
            input, config=config, **kwargs
        ):
            yield chunk

    def batch(self, inputs, config: RunnableConfig = None, **kwargs):
        return _resolve_chat_model(config).batch(inputs, config=config, **kwargs)

    async def abatch(self, inputs, config: RunnableConfig = None, **kwargs):
        return await _resolve_chat_model(config).abatch(
            inputs, config=config, **kwargs
        )

    # ---- chat-model sugar: delegate to resolved model at invoke time ----
    def bind_tools(self, tools, **kwargs):
        return _BoundRun(self, "bind_tools", {"tools": tools, **kwargs})

    def with_structured_output(self, schema, **kwargs):
        return _BoundRun(self, "with_structured_output", {"schema": schema, **kwargs})

    def with_retry(self, **kwargs):
        return _BoundRun(self, "with_retry", kwargs)

    def with_config(self, config=None, **kwargs):
        return super().with_config(config=config, **kwargs)

    @property
    def _llm_type(self):
        return "configurable-router"

    def __repr__(self):
        return "_ConfigurableModel(routes minimax: → ChatMiniMax)"


class _BoundRun(Runnable):
    """Returned by bind_tools / with_structured_output.

    Resolves the underlying model on each call and forwards the stored method.
    """

    def __init__(self, parent, method_name, kwargs):
        super().__init__()
        self._parent = parent
        self._method_name = method_name
        self._kwargs = kwargs

    def _resolve_bound(self, config):
        base = _resolve_chat_model(config)
        return getattr(base, self._method_name)(**self._kwargs)

    def invoke(self, input, config: RunnableConfig = None, **kwargs):
        return self._resolve_bound(config).invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config: RunnableConfig = None, **kwargs):
        return await self._resolve_bound(config).ainvoke(
            input, config=config, **kwargs
        )

    def stream(self, input, config: RunnableConfig = None, **kwargs):
        return self._resolve_bound(config).stream(input, config=config, **kwargs)

    async def astream(self, input, config: RunnableConfig = None, **kwargs):
        async for chunk in self._resolve_bound(config).astream(
            input, config=config, **kwargs
        ):
            yield chunk

    def batch(self, inputs, config: RunnableConfig = None, **kwargs):
        return self._resolve_bound(config).batch(inputs, config=config, **kwargs)

    async def abatch(self, inputs, config: RunnableConfig = None, **kwargs):
        return await self._resolve_bound(config).abatch(
            inputs, config=config, **kwargs
        )

    def __repr__(self):
        return f"_BoundRun(method={self._method_name})"


def create_configurable_model(configurable_fields=("model", "max_tokens", "api_key")):
    """Return a runnable that routes to the right chat model based on
    ``configurable.model``.
    """
    return _ConfigurableModel()


__all__ = ["ChatMiniMax", "create_configurable_model"]