"""
Real-MySQL validation of the optimizer (audit Phase 5).

For each benchmark query: run the original and the optimizer's output on the same
data, check the results are identical, and record median wall time, EXPLAIN access
types / keys / Extra (filesort, temporary) and EXPLAIN ANALYZE totals. Each query is
measured twice: with only the base schema indexes, then after creating the indexes
the optimizer recommended.

Usage:
    QUERYOPT_MYSQL_SOCKET=/path/to/sock python scripts/mysql_benchmark.py --sizes 20000 200000
Writes experiments/results/mysql_perf/{report.md,results.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymysql  # noqa: E402

from sqlopt import optimize_sql  # noqa: E402

SCHEMA = """
CREATE TABLE customers (customer_id INT PRIMARY KEY, email VARCHAR(100), country VARCHAR(30), is_active TINYINT(1), created_at DATETIME);
CREATE TABLE products (product_id INT PRIMARY KEY, category VARCHAR(30), price DECIMAL(10,2));
CREATE TABLE orders (order_id INT PRIMARY KEY, customer_id INT, status VARCHAR(20), total DECIMAL(10,2), created_at DATETIME, KEY idx_orders_customer (customer_id));
CREATE TABLE order_items (item_id INT PRIMARY KEY, order_id INT, product_id INT, quantity INT, unit_price DECIMAL(10,2), KEY idx_items_order (order_id));
"""

QUERIES = {
    "year_filter": "SELECT order_id, total FROM orders WHERE YEAR(created_at) = 2025 AND status = 'paid'",
    "date_upper_bound": "SELECT COUNT(*) FROM orders WHERE DATE(created_at) <= '2023-01-31'",
    "or_to_in": "SELECT COUNT(*) FROM orders WHERE status = 'paid' OR status = 'shipped' OR status = 'refunded'",
    "count_to_exists": "SELECT c.customer_id FROM customers c WHERE (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) > 0",
    "having_to_where": "SELECT status, COUNT(*), SUM(total) FROM orders GROUP BY status HAVING status <> 'cancelled'",
    "distinct_in_subquery": "SELECT COUNT(*) FROM orders WHERE customer_id IN (SELECT DISTINCT customer_id FROM customers WHERE country = 'India')",
    "left_to_inner": "SELECT COUNT(*) FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id WHERE c.country = 'India'",
    "user_report": open(Path(__file__).with_name("bench_user_query.sql")).read() if Path(__file__).with_name("bench_user_query.sql").exists() else None,
    "six_correlated (suggestion)": (
        "SELECT c.customer_id,"
        " (SELECT COUNT(*) FROM orders o1 WHERE o1.customer_id = c.customer_id) AS n_orders,"
        " (SELECT COUNT(*) FROM orders o2 WHERE o2.customer_id = c.customer_id AND o2.status = 'paid') AS n_paid,"
        " (SELECT SUM(o3.total) FROM orders o3 WHERE o3.customer_id = c.customer_id AND o3.status = 'paid') AS paid_total,"
        " (SELECT AVG(o4.total) FROM orders o4 WHERE o4.customer_id = c.customer_id) AS avg_total,"
        " (SELECT MAX(o5.created_at) FROM orders o5 WHERE o5.customer_id = c.customer_id) AS last_order,"
        " (SELECT MIN(o6.created_at) FROM orders o6 WHERE o6.customer_id = c.customer_id AND o6.total > 500) AS first_big "
        "FROM customers c WHERE c.is_active = 1"),
}
QUERIES = {k: v for k, v in QUERIES.items() if v}


def connect():
    return pymysql.connect(unix_socket=os.environ.get("QUERYOPT_MYSQL_SOCKET"),
                           host=os.environ.get("QUERYOPT_MYSQL_HOST", "127.0.0.1"),
                           port=int(os.environ.get("QUERYOPT_MYSQL_PORT", "3306")),
                           user=os.environ.get("QUERYOPT_MYSQL_USER", "root"),
                           password=os.environ.get("QUERYOPT_MYSQL_PASSWORD", ""), autocommit=True)


def load(cur, n_orders: int) -> None:
    cur.execute("DROP DATABASE IF EXISTS qo_bench")
    cur.execute("CREATE DATABASE qo_bench")
    cur.execute("USE qo_bench")
    for s in filter(str.strip, SCHEMA.split(";")):
        cur.execute(s)
    cur.execute("SET SESSION cte_max_recursion_depth = 10000000")
    n_cust = max(100, n_orders // 10)
    seq = "WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < {})"
    cur.execute("INSERT INTO customers " + seq.format(n_cust) + """
        SELECT n, CONCAT('u', n, '@x.com'),
          ELT(1 + n % 5, 'India', 'india', 'USA', 'Germany', NULL), n % 3 > 0,
          '2022-01-01' + INTERVAL (n * 37) % 1400 DAY FROM seq""")
    cur.execute("INSERT INTO products " + seq.format(1000) + """
        SELECT n, ELT(1 + n % 4, 'electronics', 'books', 'toys', 'garden'), (n * 7919) % 2000 + 0.99 FROM seq""")
    cur.execute("INSERT INTO orders " + seq.format(n_orders) + f"""
        SELECT n, 1 + (n * 7919) % {n_cust},
          ELT(1 + n % 6, 'paid', 'shipped', 'cancelled', 'refunded', 'open', NULL), (n * 104729) % 100000 / 100,
          '2022-01-01' + INTERVAL (n * 7) % 1460 DAY + INTERVAL n % 86400 SECOND FROM seq""")
    cur.execute("INSERT INTO order_items " + seq.format(n_orders * 2) + f"""
        SELECT n, 1 + n % {n_orders}, 1 + (n * 31) % 1000, n % 5, (n % 300) + 0.5 FROM seq""")
    cur.execute("ANALYZE TABLE customers, products, orders, order_items")
    cur.fetchall()


def timed(cur, sql: str, reps: int = 5) -> tuple[float, list]:
    times, rows = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        cur.execute(sql)
        rows = cur.fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), list(rows)


def explain(cur, sql: str) -> dict:
    cur.execute("EXPLAIN " + sql)
    cols = [d[0] for d in cur.description]
    plan = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.execute("EXPLAIN ANALYZE " + sql)
    analyze = cur.fetchone()[0]
    m = re.search(r"actual time=[\d.]+\.\.([\d.]+) rows=([\d.e+]+)", analyze)
    extras = " ".join(str(p.get("Extra") or "") for p in plan)
    return {
        "access": [f"{p['table']}:{p['type']}:{p['key'] or '-'}" for p in plan],
        "est_rows_examined": sum(int(p["rows"] or 0) for p in plan),
        "filesort": "filesort" in extras,
        "temporary": "temporary" in extras,
        "analyze_root_ms": float(m.group(1)) if m else None,
    }


def norm(rows):
    return Counter(tuple(float(v) if hasattr(v, "as_tuple") else v for v in r) for r in rows)


def run(sizes: list[int]) -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    out = {"mysql_version": version, "sizes": {}}
    for n in sizes:
        print(f"loading {n} orders ...", flush=True)
        load(cur, n)
        res = {}
        recommended: set[str] = set()
        plans = {}
        for name, sql in QUERIES.items():
            r = optimize_sql(sql, "mysql", SCHEMA)
            opt = r.optimized_sql
            if name.endswith("(suggestion)"):
                opt = next(c.after for c in r.changes if c.title.startswith("Merge "))
            idx = next((c.before for c in r.changes if c.title.startswith("Indexes")), "")
            recommended |= {l for l in idx.splitlines() if l.startswith("CREATE INDEX")}
            plans[name] = (sql, opt, [c.title for c in r.changes if c.kind == "applied"])
        for phase in ("base indexes", "with recommended indexes"):
            if phase == "with recommended indexes":
                for ddl in sorted(recommended):
                    try:
                        cur.execute(ddl)
                    except pymysql.err.MySQLError as e:
                        print("  skip index:", ddl, e)
                cur.execute("ANALYZE TABLE customers, products, orders, order_items")
                cur.fetchall()
            for name, (sql, opt, applied) in plans.items():
                t_o, rows_o = timed(cur, sql)
                t_n, rows_n = timed(cur, opt)
                res.setdefault(name, {"applied": applied})[phase] = {
                    "original_ms": round(t_o, 2), "optimized_ms": round(t_n, 2),
                    "speedup": round(t_o / t_n, 2) if t_n else None,
                    "same_rows": norm(rows_o) == norm(rows_n), "rows": len(rows_o),
                    "plan_original": explain(cur, sql), "plan_optimized": explain(cur, opt),
                }
                print(f"  [{phase}] {name}: {t_o:.1f} ms -> {t_n:.1f} ms  same={norm(rows_o) == norm(rows_n)}", flush=True)
        out["sizes"][n] = {"queries": res, "indexes_created": sorted(recommended)}
    return out


def report(data: dict) -> str:
    L = [f"# MySQL validation report\n\nServer: MySQL {data['mysql_version']}. Median of 5 runs, warm cache, same data "
         "for original and optimized. `same rows` compares result multisets.\n"]
    for n, d in data["sizes"].items():
        L.append(f"\n## {int(n):,} orders ({int(n) * 2:,} order items, {max(100, int(n) // 10):,} customers)\n")
        for phase in ("base indexes", "with recommended indexes"):
            L.append(f"\n### {phase}\n\n| query | original ms | optimized ms | speed-up | same rows | plan change |\n|---|---:|---:|---:|:-:|---|")
            for name, q in d["queries"].items():
                p = q[phase]
                po, pn = p["plan_original"], p["plan_optimized"]
                change = []
                if po["access"] != pn["access"]:
                    change.append(f"{', '.join(po['access'])} → {', '.join(pn['access'])}")
                if po["filesort"] != pn["filesort"]:
                    change.append(f"filesort {po['filesort']}→{pn['filesort']}")
                if po["temporary"] != pn["temporary"]:
                    change.append(f"temporary {po['temporary']}→{pn['temporary']}")
                L.append(f"| {name} | {p['original_ms']} | {p['optimized_ms']} | {p['speedup']}× | "
                         f"{'yes' if p['same_rows'] else '**NO**'} | {'; '.join(change) or 'same plan'} |")
        L.append("\nIndexes created for the second pass:\n```sql\n" + "\n".join(d["indexes_created"]) + "\n```")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[20000, 200000])
    a = ap.parse_args()
    data = run(a.sizes)
    out = Path("experiments/results/mysql_perf")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, indent=1, default=str))
    (out / "report.md").write_text(report(data))
    print("wrote", out / "report.md")
