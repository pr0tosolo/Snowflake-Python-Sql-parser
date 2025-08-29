import re
from typing import Optional, List
from sqlglot import expressions as sxe

# Regex helpers
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_COMMENT_LINE1 = re.compile(r"--.*?$", flags=re.MULTILINE)
_COMMENT_LINE2 = re.compile(r"//.*?$", flags=re.MULTILINE)
_SINGLE_QUOTED = re.compile(r"'(?:''|[^'])*'")


def strip_comments(sql: str) -> str:
    """Remove SQL comments and collapse whitespace."""
    if not isinstance(sql, str):
        return sql
    out = _COMMENT_BLOCK.sub(" ", sql)
    out = _COMMENT_LINE1.sub(" ", out)
    out = _COMMENT_LINE2.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def mask_literals(text: str) -> str:
    """Replace single quoted literals with a generic placeholder."""
    return _SINGLE_QUOTED.sub("'<???>'", text)


def flatten_and(expr: Optional[sxe.Expression]) -> List[sxe.Expression]:
    """Flatten an AND expression into a list of atomic predicates."""
    if not expr:
        return []
    if isinstance(expr, sxe.And):
        return flatten_and(expr.left) + flatten_and(expr.right)
    return [expr]
