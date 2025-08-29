"""Self-contained Snowflake SQL parser for Notebook use.
This file bundles all utilities and the SnowflakeQueryParser so it can be
pasted directly into a Snowflake Python Notebook without installing a
package. It exposes:
  - load_catalog(session, table)
  - SnowflakeQueryParser(catalog_maps)
  - parse_query_history(session, catalog_table, ...)
which returns a Snowpark DataFrame with joins, filters, and canonical SQL.
"""

from __future__ import annotations
import re, json
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import sqlglot
from sqlglot import expressions as sxe
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

# -------------------- Utility helpers --------------------
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_COMMENT_LINE1 = re.compile(r"--.*?$", flags=re.MULTILINE)
_COMMENT_LINE2 = re.compile(r"//.*?$", flags=re.MULTILINE)
_SINGLE_QUOTED = re.compile(r"'(?:''|[^'])*'")

def strip_comments(sql: str) -> str:
    if not isinstance(sql, str):
        return sql
    out = _COMMENT_BLOCK.sub(" ", sql)
    out = _COMMENT_LINE1.sub(" ", out)
    out = _COMMENT_LINE2.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()

def mask_literals(text: str) -> str:
    return _SINGLE_QUOTED.sub("'<???>'", text)

def flatten_and(expr: Optional[sxe.Expression]) -> List[sxe.Expression]:
    if not expr:
        return []
    if isinstance(expr, sxe.And):
        return flatten_and(expr.left) + flatten_and(expr.right)
    return [expr]

# ---------------- Configurable placeholder rules ----------------
DEFAULT_NAME_TYPE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)(^|[_])date($|[_])|_dt$|_date$"), "<date>"),
    (re.compile(r"(?i)timestamp|_ts$|_tms$|_time$"), "<ts>"),
    (re.compile(r"(?i)(^|[_])(id|key)($|[_])"), "<num>"),
    (re.compile(r"(?i)amount|qty|quantity|price|cost|rate|pct|percent|score|count|num$"), "<num>"),
    (re.compile(r"(?i)^is_|^has_|_flag$|_yn$"), "<bool>"),
]

# ---------------- Version-safe operator classes ----------------
def _exists(name: str) -> bool: return hasattr(sxe, name)
def _get(name: str): return getattr(sxe, name, None)

_EQ = [_get(n) for n in ("EQ", "EQF", "SafeEQ") if _exists(n)]
EQUALS = tuple([c for c in _EQ if c])

_CMP = list(EQUALS) + [_get(n) for n in ("GT", "GTE", "LT", "LTE") if _exists(n)]
COMPARISONS = tuple(c for c in _CMP if c)

LIKE_CLASSES = tuple(c for c in (_get("Like"), _get("ILike")) if c)

# ---------------- Catalog loading ----------------
def load_catalog(session: Session, table: str) -> Dict[str, Dict[str, set]]:
    """Build maps from a catalog view of columns."""
    view_to_cols: Dict[str, set] = defaultdict(set)
    col_to_views: Dict[str, set] = defaultdict(set)

    df = session.table(table).select(
        col("DATABASE_NAME"), col("SCHEMA_NAME"), col("VIEW_NAME"), col("COLUMN_NAME")
    )
    for r in df.to_local_iterator():
        db = (r["DATABASE_NAME"] or "").strip().upper()
        sc = (r["SCHEMA_NAME"] or "").strip().upper()
        vw = (r["VIEW_NAME"] or "").strip().upper()
        cn = (r["COLUMN_NAME"] or "").strip().upper()
        if not (db and sc and vw and cn):
            continue
        vkey = f"{db}.{sc}.{vw}"
        view_to_cols[vkey].add(cn)
        col_to_views[cn].add(vkey)

    return {"view_to_cols": view_to_cols, "col_to_views": col_to_views}

# ---------------- Canonicalization helpers ----------------
def canonicalize_sql(node: sxe.Expression) -> str:
    n = node.copy()
    def _rw(x: sxe.Expression) -> sxe.Expression:
        if isinstance(x, sxe.Boolean):
            return sxe.Var(this="<bool>")
        if isinstance(x, sxe.Null):
            return sxe.Var(this="<null>")
        return x
    n = n.transform(_rw)
    if EQUALS and isinstance(n, EQUALS[0]):
        if n.left.sql(dialect="snowflake") > n.right.sql(dialect="snowflake"):
            n.set("left", n.right)
            n.set("right", n.left)
    s = n.sql(dialect="snowflake")
    return re.sub(r"\s+", " ", s).strip()

def normalize_table_name(name: str) -> str:
    return name.strip().replace("..", ".").strip(".").upper()

def relation_label(node: sxe.Expression) -> str:
    if isinstance(node, sxe.Alias):
        return node.alias_or_name
    if isinstance(node, sxe.Table):
        catalog = node.args.get("catalog")
        schema = node.args.get("db")
        obj = node.this
        def _n(x):
            return x if isinstance(x, str) else getattr(x, "name", str(x))
        parts = [p for p in [_n(catalog) if catalog else None,
                             _n(schema) if schema else None,
                             _n(obj) if obj else None] if p]
        fq = ".".join(parts).upper()
        return fq if fq else getattr(node.this, "name", str(node.this)).upper()
    if isinstance(node, sxe.Subquery):
        if node.args.get("alias"):
            return node.args["alias"].this.name
        return f"<SUBQ:{hash(node.sql()) & 0xffff:x}>"
    if isinstance(node, sxe.Lateral):
        return relation_label(node.this)
    return node.sql()

def build_alias_map_and_join_scope(parsed: sxe.Expression) -> Tuple[Dict[str, str], List[str]]:
    amap: Dict[str, str] = {}
    join_scope: List[str] = []

    def _register_rel(r: sxe.Expression):
        if isinstance(r, sxe.Alias):
            base_fq = relation_label(r.this)
            amap[r.alias_or_name] = base_fq
            join_scope.append(base_fq)
        elif isinstance(r, sxe.Table):
            fq = relation_label(r)
            amap[fq] = fq
            join_scope.append(fq)
        elif isinstance(r, sxe.Subquery) and r.args.get("alias"):
            amap[r.args["alias"].this.name] = relation_label(r)

    from_ = parsed.args.get("from")
    if from_:
        for r in (from_.expressions or [from_.this]):
            _register_rel(r)

    for j in parsed.find_all(sxe.Join):
        _register_rel(j.this)

    with_ = parsed.args.get("with")
    if with_:
        for cte in (with_.expressions or []):
            if isinstance(cte, sxe.CTE) and cte.alias:
                amap[cte.alias] = relation_label(cte.this)

    seen = set()
    scope_unique = []
    for v in join_scope:
        if v not in seen:
            scope_unique.append(v)
            seen.add(v)

    return amap, scope_unique

# ---------------- Placeholder typing ----------------
def _name_based_hint(col_name: str, rules: List[Tuple[re.Pattern, str]]) -> Optional[str]:
    for pat, ph in rules:
        if pat.search(col_name):
            return ph
    return None

def _infer_placeholder(e: sxe.Expression,
                       view_to_cols: Dict[str, set],
                       col_to_views: Dict[str, set],
                       join_scope: List[str],
                       name_type_rules: List[Tuple[re.Pattern, str]]) -> str:
    lit = None
    if hasattr(e, 'right') and isinstance(e.right, sxe.Literal):
        lit = e.right
    elif isinstance(e, sxe.In) and isinstance(e.expression, sxe.Tuple):
        for v in e.expression.expressions or []:
            if isinstance(v, sxe.Literal):
                lit = v
                break
    elif isinstance(e, sxe.Between):
        if isinstance(e.args.get('low'), sxe.Literal):
            lit = e.args.get('low')
        elif isinstance(e.args.get('high'), sxe.Literal):
            lit = e.args.get('high')

    if isinstance(lit, sxe.Literal):
        if lit.is_string:
            txt = str(lit.this)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", txt):
                return "<date>"
            return "<str>"
        if lit.is_number:
            return "<num>"

    if isinstance(e, sxe.Is) and e.expression:
        if isinstance(e.expression, sxe.Null):
            return "<null>"
        if isinstance(e.expression, sxe.Boolean):
            return "<bool>"

    def _hint(x: sxe.Expression):
        if isinstance(x, sxe.Cast):
            to = x.args.get('to')
            if to:
                t = to.sql(dialect='snowflake').upper()
                if 'DATE' in t and 'TIME' not in t:
                    return '<date>'
                if 'TIMESTAMP' in t or 'TIME' in t:
                    return '<ts>'
                if any(n in t for n in ('CHAR','STRING','TEXT','VAR')):
                    return '<str>'
                if any(n in t for n in ('INT','NUM','DEC','FLOAT','DOUBLE')):
                    return '<num>'
        if isinstance(x, sxe.Anonymous):
            fn = (x.name or '').upper()
            if fn in {'TO_DATE','DATE','DATE_FROM_PARTS'}:
                return '<date>'
            if fn in {'TO_TIMESTAMP','TO_TIME','TIME_FROM_PARTS'}:
                return '<ts>'
        return None

    for side in ('right','expression','this'):
        node = getattr(e, side, None)
        if isinstance(node, sxe.Expression):
            h = _hint(node)
            if h:
                return h

    target_col: Optional[sxe.Column] = None
    if hasattr(e, 'left') and isinstance(e.left, sxe.Column):
        target_col = e.left
    elif isinstance(e, sxe.In) and isinstance(e.this, sxe.Column):
        target_col = e.this
    elif isinstance(e, sxe.Between) and isinstance(e.this, sxe.Column):
        target_col = e.this

    if target_col is not None:
        name_hint = _name_based_hint(target_col.name.upper(), name_type_rules)
        if name_hint:
            return name_hint
        return "<str>"

    return "<lit>"

# ---------------- Column FQN resolution ----------------
def _fq_for_column(col: sxe.Column,
                   alias_map: Dict[str,str],
                   view_to_cols: Dict[str,set],
                   col_to_views: Dict[str,set],
                   join_scope: List[str]) -> str:
    cname = col.name.upper()
    talias = col.table if hasattr(col, "table") else None

    if talias:
        base = alias_map.get(talias, talias)
        parts = base.split(".")
        if len(parts) == 3:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.{cname}"

    candidates = col_to_views.get(cname, set())
    if candidates:
        jset = set(v.upper() for v in join_scope)
        cand_in_scope = [v for v in candidates if v in jset]
        if len(cand_in_scope) == 1:
            return f"{cand_in_scope[0]}.{cname}"
        if len(cand_in_scope) > 1:
            return f"<AMBIG:{'|'.join(sorted(cand_in_scope))}>.{cname}"

    if len(join_scope) == 1 and join_scope[0].count(".") == 2:
        return f"{join_scope[0]}.{cname}"

    return f"<UNKNOWN>.{cname}"

# ---------------- Predicate canonicalization ----------------
def canonicalize_predicate(e: sxe.Expression,
                           alias_map: Dict[str,str],
                           view_to_cols: Dict[str,set],
                           col_to_views: Dict[str,set],
                           join_scope: List[str],
                           name_type_rules: List[Tuple[re.Pattern, str]]) -> str:
    def q(c: sxe.Column) -> str:
        return _fq_for_column(c, alias_map, view_to_cols, col_to_views, join_scope)

    if isinstance(e, COMPARISONS) and isinstance(e.left, sxe.Column) and isinstance(e.right, sxe.Column):
        l = q(e.left)
        r = q(e.right)
        if EQUALS and isinstance(e, EQUALS) and r < l:
            l, r = r, l
        return f"{l} {e.key.upper()} {r}"

    if (isinstance(e, COMPARISONS) or isinstance(e, LIKE_CLASSES)) and isinstance(e.left, sxe.Column):
        ph = _infer_placeholder(e, view_to_cols, col_to_views, join_scope, name_type_rules)
        return f"{q(e.left)} {e.key.upper()} {ph}"

    if isinstance(e, sxe.In) and isinstance(e.this, sxe.Column):
        ph = _infer_placeholder(e, view_to_cols, col_to_views, join_scope, name_type_rules)
        k = len(e.expression.expressions or []) if isinstance(e.expression, sxe.Tuple) else 0
        suffix = f"<IN:{k}x{ph.strip('<>')}>" if k else f"<IN:{ph.strip('<>')}>"
        return f"{q(e.this)} IN {suffix}"

    if isinstance(e, sxe.Between) and isinstance(e.this, sxe.Column):
        ph = _infer_placeholder(e, view_to_cols, col_to_views, join_scope, name_type_rules)
        return f"{q(e.this)} BETWEEN {ph} AND {ph}"

    if isinstance(e, sxe.Is) and isinstance(e.this, sxe.Column):
        right_key = e.expression.key.upper() if e.expression else "UNKNOWN"
        return f"{q(e.this)} IS {right_key}"

    return canonicalize_sql(e)

# ---------------- UNION helpers ----------------
def is_set_op(node: sxe.Expression) -> bool:
    return isinstance(node, (sxe.Union, sxe.Except, sxe.Intersect))

def flatten_union(node: sxe.Expression):
    branches: List[sxe.Expression] = []
    ops: List[str] = []

    def rec(n: sxe.Expression):
        if isinstance(n, sxe.Union):
            left = n.this
            right = n.expression
            rec(left)
            rec(right)
            ops.append("UNION" if (n.args.get("distinct", True)) else "UNION_ALL")
        else:
            branches.append(n)

    rec(node)
    return branches, ops

# ---------------- Core parsing for a single AST ----------------
def extract_from_ast(parsed: sxe.Expression,
                     catalog_maps: Dict[str, set],
                     branch_id: int,
                     name_type_rules: List[Tuple[re.Pattern, str]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"branch": branch_id}

    alias_map, join_scope = build_alias_map_and_join_scope(parsed)

    # FROM list
    from_expr = parsed.args.get("from")
    if from_expr:
        tables = list(from_expr.find_all(sxe.Table))
        res["from"] = {"branch": branch_id, "from": [relation_label(t) for t in tables] if tables else from_expr.sql(dialect="snowflake")}
    else:
        res["from"] = {"branch": branch_id, "from": None}

    # JOINs
    join_list = []
    left_ctx = join_scope[0] if join_scope else None
    for j in parsed.find_all(sxe.Join):
        right_name = relation_label(j.this)
        pairs = []
        using_cols = j.args.get("using")
        on_expr = j.args.get("on")
        if using_cols:
            for c in using_cols.expressions:
                cname = c.name
                can = canonicalize_sql(sxe.EQ(this=sxe.Column(this=cname), expression=sxe.Column(this=cname)))
                pairs.append({
                    "branch": branch_id,
                    "left_table": left_ctx, "left_col": cname,
                    "right_table": right_name, "right_col": cname,
                    "canonical": can
                })
        elif on_expr:
            for e in flatten_and(on_expr):
                if EQUALS and isinstance(e, EQUALS) and isinstance(e.left, sxe.Column) and isinstance(e.right, sxe.Column):
                    lc = canonicalize_sql(e)
                    pairs.append({
                        "branch": branch_id,
                        "left_table": left_ctx, "left_col": e.left.name,
                        "right_table": right_name, "right_col": e.right.name,
                        "canonical": lc
                    })
        join_list.append({
            "branch": branch_id,
            "left": left_ctx,
            "right": right_name,
            "pairs": pairs,
            "raw": j.sql(dialect="snowflake"),
        })
        left_ctx = right_name
    res["joins"] = join_list

    # SELECT
    sel_exprs, alias_to_expr = [], {}
    if getattr(parsed, "expressions", None):
        for sel in parsed.expressions:
            name = sel.alias_or_name if isinstance(sel, sxe.Alias) else None
            node = sel.this if isinstance(sel, sxe.Alias) else sel
            can = canonicalize_sql(node)
            sel_exprs.append({"branch": branch_id, "expr": can, "alias": name})
            if name:
                alias_to_expr[name] = can
    res["select_expressions"] = sel_exprs
    res["alias_to_expression"] = {"branch": branch_id, "map": alias_to_expr}

    # WHERE / HAVING / QUALIFY
    def atoms(expr: Optional[sxe.Expression]) -> List[Dict[str, Any]]:
        if not expr:
            return []
        arr = []
        chain = flatten_and(expr if not isinstance(expr, sxe.Having) else expr.this)
        v2c = catalog_maps["view_to_cols"]
        c2v = catalog_maps["col_to_views"]
        for a in chain:
            canon = canonicalize_predicate(a, alias_map, v2c, c2v, join_scope, name_type_rules)
            canon = mask_literals(canon)
            arr.append({"branch": branch_id, "canon": canon})
        return arr

    where_e = parsed.args.get("where")
    having_e = parsed.args.get("having")
    qualify_e = parsed.args.get("qualify")

    res["where"] = atoms(where_e.this) if where_e else None
    res["having"] = atoms(having_e) if having_e else None
    res["qualify"] = atoms(qualify_e.this) if qualify_e else None

    # GROUP / ORDER / LIMIT
    group_e = parsed.args.get("group")
    if group_e and hasattr(group_e, "expressions"):
        res["group_by"] = [{"branch": branch_id, "expr": canonicalize_sql(e)} for e in group_e.expressions]
    else:
        res["group_by"] = [{"branch": branch_id, "expr": group_e.sql(dialect="snowflake")}] if group_e else None

    order_e = parsed.args.get("order")
    if order_e and hasattr(order_e, "expressions"):
        res["order_by"] = [{"branch": branch_id, "expr": canonicalize_sql(e)} for e in order_e.expressions]
    else:
        res["order_by"] = [{"branch": branch_id, "expr": order_e.sql(dialect="snowflake")}] if order_e else None

    lim_e = parsed.args.get("limit")
    res["limit"] = {"branch": branch_id, "expr": lim_e.sql(dialect="snowflake")} if lim_e else None

    return res

# ---------------- Extractor orchestration ----------------
def extract_query_objects(sql_text: str,
                          catalog_maps: Dict[str, set],
                          name_type_rules: List[Tuple[re.Pattern, str]]) -> Dict[str, Any]:
    clean_sql = strip_comments(sql_text)
    try:
        parsed = sqlglot.parse_one(clean_sql, read="snowflake")
    except Exception:
        try:
            parsed = sqlglot.parse_one(clean_sql, read="ansi")
        except Exception:
            return {}

    if is_set_op(parsed):
        branches, ops = flatten_union(parsed)
        merged: Dict[str, Any] = {
            "set_op": ops,
            "branch_count": len(branches)
        }
        accum = {
            "from": [], "joins": [], "select_expressions": [],
            "where": [], "having": [], "qualify": [],
            "group_by": [], "order_by": [], "limit": []
        }
        for i, b in enumerate(branches):
            bx = extract_from_ast(b, catalog_maps, i, name_type_rules)
            for k in accum.keys():
                v = bx.get(k)
                if v is None:
                    continue
                if isinstance(v, list):
                    accum[k].extend(v)
                else:
                    accum[k].append(v)
        merged.update(accum)
        return merged

    return extract_from_ast(parsed, catalog_maps, 0, name_type_rules)

# ---------------- Canonical SQL builder ----------------
def _branch_items(items: Optional[List[dict]], branch: int) -> List[dict]:
    if not items:
        return []
    return [x for x in items if isinstance(x, dict) and x.get("branch") == branch]

def _build_join_clause(branch_joins: List[dict]) -> str:
    parts = []
    for j in branch_joins:
        right = j.get("right")
        pairs = j.get("pairs") or []
        if pairs:
            conds = [p.get("canonical") for p in pairs if p.get("canonical")]
            join_txt = f"JOIN {right} ON (" + " AND ".join(conds) + ")"
            parts.append(join_txt)
        else:
            parts.append(j.get("raw", ""))
    return " ".join([p for p in parts if p])

def _build_where_clause(branch_where: List[dict]) -> str:
    if not branch_where:
        return ""
    atoms = [w.get("canon") for w in branch_where if w.get("canon")]
    if not atoms:
        return ""
    return "WHERE " + " AND ".join(atoms)

def _build_list_clause(keyword: str, items: List[dict]) -> str:
    if not items:
        return ""
    exprs = [e.get("expr") for e in items if e.get("expr")]
    if not exprs:
        return ""
    return f"{keyword} " + ", ".join(exprs)

def _build_select_list(branch_selects: List[dict]) -> str:
    if not branch_selects:
        return "*"
    out = []
    for s in branch_selects:
        expr = s.get("expr")
        alias = s.get("alias")
        if not expr:
            continue
        out.append(f"{expr} AS {alias}" if alias else expr)
    return ", ".join(out) if out else "*"

def build_canonical_sql_from_extracted(extracted: Dict[str, Any]) -> str:
    branch_count = int(extracted.get("branch_count")) if extracted.get("branch_count") is not None else 1
    if branch_count == 1 and not extracted.get("set_op"):
        branch_count = 1
    ops = extracted.get("set_op") or []

    def build_branch_sql(b: int) -> str:
        from_entries = _branch_items(extracted.get("from"), b)
        base = None
        if from_entries and from_entries[0].get("from"):
            base_list = from_entries[0].get("from")
            if isinstance(base_list, list) and base_list:
                base = base_list[0]
            elif isinstance(base_list, str):
                base = base_list
        joins = _branch_items(extracted.get("joins"), b)
        join_sql = _build_join_clause(joins)
        from_sql = f"FROM {base}" if base else ""
        sels = _branch_items(extracted.get("select_expressions"), b)
        select_sql = "SELECT " + _build_select_list(sels)
        where_sql = _build_where_clause(_branch_items(extracted.get("where"), b))
        group_sql = _build_list_clause("GROUP BY", _branch_items(extracted.get("group_by"), b))
        having_sql = _build_list_clause("HAVING", _branch_items(extracted.get("having"), b))
        qualify_sql = _build_list_clause("QUALIFY", _branch_items(extracted.get("qualify"), b))
        order_sql = _build_list_clause("ORDER BY", _branch_items(extracted.get("order_by"), b))
        limit_items = _branch_items(extracted.get("limit"), b)
        limit_sql = f"LIMIT {limit_items[0].get('expr')}" if limit_items and limit_items[0].get('expr') else ""
        parts = [select_sql, from_sql, join_sql, where_sql, group_sql, having_sql, qualify_sql, order_sql, limit_sql]
        canon = " ".join([p for p in parts if p])
        canon = re.sub(r"\s+", " ", canon).strip()
        return f"( {canon} )"

    if branch_count <= 1:
        return build_branch_sql(0)

    branches_sql = [build_branch_sql(i) for i in range(branch_count)]
    out = branches_sql[0]
    for i in range(1, branch_count):
        op = (ops[i-1] if i-1 < len(ops) else "UNION_ALL")
        out = f"{out} {op.replace('_', ' ')} {branches_sql[i]}"
    return out

# ---------------- High level parser class ----------------
class SnowflakeQueryParser:
    """Parse SQL text into join/filter details and canonical SQL."""
    def __init__(self, catalog_maps: Dict[str, set],
                 name_type_rules: Optional[List[Tuple[re.Pattern, str]]] = None):
        self.catalog_maps = catalog_maps
        self.name_type_rules = name_type_rules or DEFAULT_NAME_TYPE_RULES

    def parse(self, sql_text: str) -> Dict[str, Any]:
        ex = extract_query_objects(sql_text, self.catalog_maps, self.name_type_rules)
        if not ex:
            return {"joins": [], "filters": [], "canonical_sql": ""}
        canonical_sql = build_canonical_sql_from_extracted(ex)
        canonical_sql = mask_literals(canonical_sql)
        canonical_sql = re.sub(r"<(?!UNKNOWN|AMBIG:[^>]+)[^>]+>", "<???>", canonical_sql)
        joins = []
        for j in ex.get("joins", []):
            conds = [p.get("canonical") for p in j.get("pairs", []) if p.get("canonical")]
            if not conds and j.get("raw"):
                conds = [j.get("raw")]
            joins.append({
                "left": j.get("left"),
                "right": j.get("right"),
                "conditions": conds,
            })
        filters = [re.sub(r"<(?!UNKNOWN|AMBIG:[^>]+)[^>]+>", "<???>", w.get("canon")) for w in (ex.get("where") or []) if w.get("canon")]
        return {"joins": joins, "filters": filters, "canonical_sql": canonical_sql}

# ---------------- Notebook helper ----------------
QUERY_HISTORY_TABLE = "SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY"
START_EXPR = "DATEADD('day', -7, CURRENT_TIMESTAMP())"
END_EXPR = "CURRENT_TIMESTAMP()"
MAX_ROWS = 5000
QUERY_TYPES = ("SELECT","CREATE_VIEW","CREATE OR REPLACE VIEW","INSERT","MERGE","UPDATE")

def parse_query_history(session: Session,
                        catalog_table: str,
                        query_history_table: str = QUERY_HISTORY_TABLE,
                        start_expr: str = START_EXPR,
                        end_expr: str = END_EXPR,
                        max_rows: int = MAX_ROWS,
                        query_types: Tuple[str, ...] = QUERY_TYPES):
    """Parse recent query history and return a Snowpark DataFrame with joins, filters, and canonical SQL."""
    catalog_maps = load_catalog(session, catalog_table)
    parser = SnowflakeQueryParser(catalog_maps)

    df = session.table(query_history_table) \
        .select(col("QUERY_ID"), col("QUERY_TEXT"), col("QUERY_TYPE"), col("START_TIME")) \
        .where(f"START_TIME BETWEEN {start_expr} AND {end_expr}")
    if query_types:
        qtypes = ",".join([f"'{t}'" for t in query_types])
        df = df.where(f"QUERY_TYPE IN ({qtypes})")
    if max_rows:
        df = df.limit(max_rows)

    rows: List[Tuple] = []
    schema = ["QUERY_ID", "JOINS", "FILTERS", "CANONICAL_SQL"]

    for r in df.to_local_iterator():
        qid = r["QUERY_ID"]
        qtxt = r["QUERY_TEXT"]
        if not isinstance(qtxt, str) or not qtxt.strip():
            continue
        parsed = parser.parse(qtxt)
        rows.append((
            str(qid),
            json.dumps(parsed["joins"], ensure_ascii=False),
            json.dumps(parsed["filters"], ensure_ascii=False),
            parsed["canonical_sql"],
        ))

    return session.create_dataframe(rows, schema=schema)
