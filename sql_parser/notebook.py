import json
from typing import Tuple, List, Optional

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

from .parser import SnowflakeQueryParser, load_catalog

# Default configuration for query history
# Customer-specific table storing query history rows
QUERY_HISTORY_TABLE = "USER_QUERY_HISTORY"
START_EXPR = "DATEADD('day', -7, CURRENT_TIMESTAMP())"
END_EXPR = "CURRENT_TIMESTAMP()"
MAX_ROWS = 5000
QUERY_TYPES = ("SELECT","CREATE_VIEW","CREATE OR REPLACE VIEW","INSERT","MERGE","UPDATE")

# Catalog view holding column metadata for resolving column references
CATALOG_TABLE = "PROD_110575_ICDW_DB.PUBLIC_GLOBAL.CONSTELLATION_VIEW_COLUMN_WITH_PCI_POLICIES"


def _iter_history(session: Session,
                  query_history_table: str,
                  start_expr: str,
                  end_expr: str,
                  max_rows: int,
                  query_types: Tuple[str, ...]):
    df = session.table(query_history_table) \
        .select(col("QUERY_ID"), col("QUERY_TEXT"), col("QUERY_TYPE"), col("START_TIME")) \
        .where(f"START_TIME BETWEEN {start_expr} AND {end_expr}")
    if query_types:
        qtypes = ",".join([f"'{t}'" for t in query_types])
        df = df.where(f"QUERY_TYPE IN ({qtypes})")
    if max_rows:
        df = df.limit(max_rows)
    return df.to_local_iterator()


def parse_query_history(session: Session,
                        catalog_table: str = CATALOG_TABLE,
                        query_history_table: str = QUERY_HISTORY_TABLE,
                        start_expr: str = START_EXPR,
                        end_expr: str = END_EXPR,
                        max_rows: int = MAX_ROWS,
                        query_types: Tuple[str, ...] = QUERY_TYPES):
    """Parse recent query history and return a Snowpark DataFrame with joins, filters, and canonical SQL."""
    catalog_maps = load_catalog(session, catalog_table)
    parser = SnowflakeQueryParser(catalog_maps)

    rows: List[Tuple] = []
    schema = ["QUERY_ID", "JOINS", "FILTERS", "CANONICAL_SQL"]

    for r in _iter_history(session, query_history_table, start_expr, end_expr, max_rows, query_types):
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


def main(session: Session):
    """Snowflake Python Notebook entry point."""
    return parse_query_history(session)
