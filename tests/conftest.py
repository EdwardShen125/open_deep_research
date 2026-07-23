"""Pytest 配置:全局 fixtures。

P1.1 修复副作用: server.py 在 import 时从 .env 加载环境变量。
但 unit tests 应在受控 env 下跑 — 它们的 fixture (monkeypatch.delenv) 不能
影响已 import 的 server module 的 os.environ (后者在 module-level 已加载)。
所以这里在每个 test 启动前,清掉 .env 加载可能污染的 env vars (TAVILY_API_KEY /
SEARXNG_URL / POSTGRES_*),让 monkeypatch.delenv 能真正生效。
"""
from __future__ import annotations

import os
import sys

import pytest


# Server 模块 import 时可能从 .env 加载的 env vars
# 注意:不要清 POSTGRES_* — PG 集成测试需要它们
_DOTENV_LEAKED_VARS = ("TAVILY_API_KEY", "SEARXNG_URL")


@pytest.fixture(autouse=True)
def _isolate_dotenv_vars_per_test(monkeypatch):
    """Auto-applied fixture: 每个 test 启动前清掉 .env 加载的 env vars。

    原因: server.py 在 import 时 _load_dotenv_fallback 已运行,可能污染了
    os.environ (例如 TAVILY_API_KEY 即使没有也变成空字符串)。unit tests 用
    monkeypatch.delenv 时,实际值已经被 .env 设置,delenv 无效。

    修法: test 启动前 .delenv 这些 vars,让 monkeypatch 之后能重新控制。
    PG 集成测试 (POSTGRES_HOST 已设) 不受影响 — 我们不清 POSTGRES_*。
    """
    for var in _DOTENV_LEAKED_VARS:
        # 用 monkeypatch.delenv + raising=False 安全清 (env 没设时不报错)
        monkeypatch.delenv(var, raising=False)
    yield