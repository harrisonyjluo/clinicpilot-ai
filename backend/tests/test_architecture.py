"""Import-boundary guards for the agent and RAG packages."""

import ast
from pathlib import Path

from app import agent, rag

FORBIDDEN_AGENT_IMPORTS = (
    "app.db",
    "app.models",
    "app.repositories",
    "app.services",
    "app.api",
)

FORBIDDEN_RAG_IMPORTS = (
    "anthropic",
    "app.agent",
    "app.api",
    "app.db",
    "app.models",
    "app.repositories",
    "app.services",
    "httpx",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _package_dir(module) -> Path:
    return Path(module.__file__).parent


def test_agent_never_imports_persistence_or_services():
    for path in _package_dir(agent).glob("*.py"):
        imports = _imports(path)
        for forbidden in FORBIDDEN_AGENT_IMPORTS:
            assert not any(
                imported == forbidden or imported.startswith(f"{forbidden}.")
                for imported in imports
            ), f"{path.name} imports {forbidden}"


def test_rag_stays_read_only_and_llm_free():
    for path in _package_dir(rag).glob("*.py"):
        imports = _imports(path)
        for forbidden in FORBIDDEN_RAG_IMPORTS:
            assert not any(
                imported == forbidden or imported.startswith(f"{forbidden}.")
                for imported in imports
            ), f"{path.name} imports {forbidden}"


def test_nlu_modules_do_not_import_regular_expressions():
    """Natural-language understanding must come from the LLM (LLM-only NLU rule)."""

    for name in ("router.py", "orchestrator.py"):
        imports = _imports(_package_dir(agent) / name)
        assert "re" not in imports, (
            f"agent/{name} imports re; natural-language input must be "
            "interpreted by the LLM classifier, not by text matching"
        )
