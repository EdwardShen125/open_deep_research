"""V1 audit — fail-fast on silent fixture / mock data paths.

The single most dangerous failure mode is "silent fixture": a search or
extraction path returns hardcoded sample data when the real backend fails.
This produces reports that *look* correct but contain fabricated content.

These tests scan the codebase at test time to confirm no such path
exists. If anyone re-introduces one, CI fails before the regression
hits production reports.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "open_deep_research"

# Forbidden tokens that indicate silent fixture / mock data paths.
# Keep this list small and high-signal — false positives are worse than
# missing a single instance.
FORBIDDEN = re.compile(
    r"\b(fixture|sample_result|_MOCK|demo_data|placeholder_data|hack_data)\b",
    re.IGNORECASE,
)

# File exceptions: tests themselves + comment-only historical notes
EXCLUDE_FILES = {
    "tests",  # test files
    "__pycache__",
    ".pyc",
}

# Allowed lines that *mention* these words without being fixtures
ALLOWED_LINE_PATTERNS = [
    re.compile(r"^\s*#"),  # pure comment line
    re.compile(r"docstring", re.IGNORECASE),
    # historical note about bug fixes
    re.compile(r"silent ReadTimeout.*fixture fallback", re.IGNORECASE),
    re.compile(r"fixture.*fallback", re.IGNORECASE),
    # "测试 fixture" = pytest fixture terminology (NOT sample data path)
    re.compile(r"测试\s*fixture", re.IGNORECASE),
    re.compile(r"test\s*fixture", re.IGNORECASE),
    re.compile(r"pytest\s*fixture", re.IGNORECASE),
]


def _is_allowed(line: str) -> bool:
    for pat in ALLOWED_LINE_PATTERNS:
        if pat.search(line):
            return True
    return False


def test_no_silent_fixture_in_src():
    """Grep src/open_deep_research/ for fixture / mock data paths."""
    offenders = []
    for py in SRC.rglob("*.py"):
        rel = py.relative_to(SRC)
        text = py.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN.search(line) and not _is_allowed(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:120]}")
    assert not offenders, (
        "Silent fixture / mock data path detected — "
        "these produce fabricated reports when real backends fail.\n"
        "Either delete the fixture or convert it to raise/degraded.\n"
        + "\n".join(offenders)
    )


def test_search_providers_does_not_import_fixture_module():
    """search_providers.py must not import a fixture module."""
    sp = SRC / "search_providers.py"
    text = sp.read_text(encoding="utf-8")
    assert "import fixture" not in text, (
        "search_providers.py imported a fixture module — silent fixture regression"
    )
    # Assert AllProvidersFailed is exported and referenced (fail-fast is wired)
    assert "AllProvidersFailed" in text, (
        "search_providers.py missing AllProvidersFailed — fail-fast broken"
    )
    assert "raise AllProvidersFailed" in text, (
        "search_providers.py does not raise AllProvidersFailed — silent fallback?"
    )


def test_run_odr_does_not_have_search_empty_fallback_branch():
    """run_odr.py must not have a 'if no results, use preset data' branch."""
    runner = SRC / "run_odr.py"
    if not runner.exists():
        return
    text = runner.read_text(encoding="utf-8")
    forbidden_patterns = [
        re.compile(r"if not results.*sample", re.IGNORECASE | re.DOTALL),
        re.compile(r"no results.*fixture", re.IGNORECASE),
        re.compile(r"empty.*demo_data", re.IGNORECASE),
    ]
    for pat in forbidden_patterns:
        m = pat.search(text)
        assert not m, f"run_odr.py has silent fallback pattern: {pat.pattern}"