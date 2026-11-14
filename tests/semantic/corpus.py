"""
Structured MySQL optimization corpus (audit Phase 2).

Each case states what the engine must do:
  apply    — titles (substrings) that must appear as *applied* changes
  suggest  — titles that must appear as *suggestions*
  keep     — True when the query text must come back unchanged (no rewrite allowed)
Every case that is rewritten is additionally executed (original vs optimized) by
tests/semantic/test_corpus.py on generated data to prove equivalence.
"""
from __future__ import annotations

SCHEMA = """
CREATE TABLE customers (
  customer_id INT PRIMARY KEY, email VARCHAR(100), country VARCHAR(30), is_active TINYINT(1),
  segment VARCHAR(20), created_at DATETIME
);
CREATE TABLE orders (
  order_id INT PRIMARY KEY, customer_id INT, status VARCHAR(20), total DECIMAL(10,2),
  created_at DATETIME, ship_date DATE, KEY idx_orders_customer (customer_id)
);
CREATE TABLE order_items (
  item_id INT PRIMARY KEY, order_id INT, product_id INT, quantity INT, unit_price DECIMAL(10,2),
  KEY idx_items_order (order_id)
);
CREATE TABLE products (product_id INT PRIMARY KEY, category VARCHAR(30), price DECIMAL(10,2), name VARCHAR(50));
"""

C = dict  # brevity

CASES: list[dict] = [
    # ── A. basic ────────────────────────────────────────────────────────────
    C(id="A01", cat="basic", sql="SELECT * FROM orders", suggest=["List the columns"], keep=True),
    C(id="A02", cat="basic", sql="SELECT order_id FROM orders WHERE 1 = 1 AND status = 'paid'", apply=["Removed redundant conditions"]),
    C(id="A03", cat="basic", sql="SELECT order_id FROM orders WHERE status = 'paid' AND status = 'paid'", apply=["Removed redundant conditions"]),
    C(id="A04", cat="basic", sql="SELECT order_id FROM orders WHERE status = 'paid' OR status = 'shipped' OR status = 'Paid'", apply=["Combined OR conditions into IN"]),
    C(id="A05", cat="basic", sql="SELECT order_id FROM orders WHERE status = 'paid' OR total = 5", keep=True),
    C(id="A06", cat="basic", sql="SELECT order_id FROM orders WHERE status = 'paid' OR status IS NULL", keep=True),
    C(id="A07", cat="basic", sql="SELECT order_id FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers WHERE is_active = 0)",
      suggest=["NOT EXISTS instead of NOT IN"], keep=True),
    C(id="A08", cat="basic", sql="SELECT order_id FROM orders o WHERE EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id)", keep=True),
    C(id="A09", cat="basic", sql="SELECT order_id FROM orders WHERE total BETWEEN 10 AND 1000", keep=True),
    C(id="A10", cat="basic", sql="SELECT name FROM products WHERE name LIKE '%phone'", suggest=["Leading wildcard"], keep=True),
    C(id="A11", cat="basic", sql="SELECT name FROM products WHERE name LIKE 'pho%'", keep=True),
    C(id="A12", cat="basic", sql="SELECT DISTINCT status FROM orders", keep=True),
    C(id="A13", cat="basic", sql="SELECT order_id FROM orders WHERE status = NULL", suggest=["Comparison with NULL"], keep=True),
    C(id="A14", cat="basic", sql="SELECT order_id FROM orders WHERE total = total", keep=True),  # NULL rows excluded: not a tautology
    C(id="A15", cat="basic", sql="SELECT order_id FROM orders WHERE 'a' = 'A' AND status = 'paid'", keep=True),  # collation-dependent
    C(id="A16", cat="basic", sql="SELECT order_id FROM orders WHERE total > 2 + 3", keep=True),

    # ── B. function-based filtering ─────────────────────────────────────────
    C(id="B01", cat="functions", sql="SELECT order_id FROM orders WHERE YEAR(created_at) = 2025", apply=["date filter index-friendly"]),
    C(id="B02", cat="functions", sql="SELECT order_id FROM orders WHERE YEAR(created_at) < 2025", apply=["date filter index-friendly"]),
    C(id="B03", cat="functions", sql="SELECT order_id FROM orders WHERE YEAR(created_at) >= 2025 AND YEAR(created_at) <= 2025", apply=["date filter index-friendly"]),
    C(id="B04", cat="functions", sql="SELECT order_id FROM orders WHERE YEAR(created_at) BETWEEN 2024 AND 2025", apply=["date filter index-friendly"]),
    C(id="B05", cat="functions", sql="SELECT order_id FROM orders WHERE DATE(created_at) = '2025-12-31'", apply=["date filter index-friendly"]),
    C(id="B06", cat="functions", sql="SELECT order_id FROM orders WHERE DATE(created_at) <= '2025-12-31'", apply=["date filter index-friendly"]),
    C(id="B07", cat="functions", sql="SELECT order_id FROM orders WHERE DATE(created_at) > '2025-01-01'", apply=["date filter index-friendly"]),
    C(id="B08", cat="functions", sql="SELECT order_id FROM orders WHERE DATE(created_at) = '2025-12-31 10:00:00'", keep=True),
    C(id="B09", cat="functions", sql="SELECT order_id FROM orders WHERE YEAR(ship_date) = 2025", apply=["date filter index-friendly"]),
    C(id="B10", cat="functions", sql="SELECT order_id FROM orders WHERE MONTH(created_at) = 12", suggest=["Function applied"], keep=True),
    C(id="B11", cat="functions", sql="SELECT order_id FROM orders WHERE DAY(created_at) = 31", suggest=["Function applied"], keep=True),
    C(id="B12", cat="functions", sql="SELECT customer_id FROM customers WHERE LOWER(country) = 'india'", suggest=["Function applied"], keep=True),
    C(id="B13", cat="functions", sql="SELECT customer_id FROM customers WHERE UPPER(country) = 'INDIA'", suggest=["Function applied"], keep=True),
    C(id="B14", cat="functions", sql="SELECT order_id FROM orders WHERE CAST(total AS SIGNED) = 5", suggest=["Function applied"], keep=True),
    C(id="B15", cat="functions", sql="SELECT order_id FROM orders WHERE CONVERT(status, CHAR) = 'paid'", suggest=["Function applied"], keep=True),
    C(id="B16", cat="functions", sql="SELECT order_id FROM orders WHERE COALESCE(status, 'none') = 'none'", suggest=["Function applied"], keep=True),
    C(id="B17", cat="functions", sql="SELECT order_id FROM orders WHERE IFNULL(total, 0) > 100", suggest=["Function applied"], keep=True),
    C(id="B18", cat="functions", sql="SELECT order_id FROM orders WHERE NOT (YEAR(created_at) = 2025)", apply=["date filter index-friendly"]),
    C(id="B19", cat="functions", sql="SELECT order_id FROM orders WHERE YEAR(created_at) = 2025 OR status = 'paid'", apply=["date filter index-friendly"]),
    C(id="B20", cat="functions", sql="SELECT order_id FROM orders WHERE YEAR(created_at) = 2025 AND DATE(created_at) <= '2025-12-31'",
      apply=["date filter index-friendly", "already implied"]),

    # ── C. joins ────────────────────────────────────────────────────────────
    C(id="C01", cat="joins", sql="SELECT o.order_id, c.email FROM orders o, customers c WHERE o.customer_id = c.customer_id AND o.total > 10",
      apply=["explicit JOIN"]),
    C(id="C02", cat="joins", sql="SELECT o.order_id FROM orders o, customers c", suggest=["No join condition"], keep=True),
    C(id="C03", cat="joins", sql="SELECT o.order_id FROM orders o CROSS JOIN products p", suggest=["No join condition"], keep=True),
    C(id="C04", cat="joins", sql="SELECT o.order_id, c.email FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id WHERE c.country = 'India'",
      apply=["LEFT JOIN"]),
    C(id="C05", cat="joins", sql="SELECT o.order_id FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id WHERE c.customer_id IS NULL", keep=True),
    C(id="C06", cat="joins", sql="SELECT o.order_id FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id WHERE COALESCE(c.country, 'x') = 'x'",
      keep=True, suggest=["Function applied"]),
    C(id="C07", cat="joins", sql="SELECT o.order_id FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id WHERE c.country = 'India' OR o.total > 5", keep=True),
    C(id="C08", cat="joins", sql="SELECT o.order_id FROM orders o RIGHT JOIN customers c ON c.customer_id = o.customer_id", keep=True),
    C(id="C09", cat="joins", sql="SELECT o.order_id, p.name FROM orders o JOIN order_items i ON i.order_id = o.order_id JOIN products p ON p.product_id = i.product_id WHERE p.price > 1000",
      suggest=["Indexes"], keep=True),
    C(id="C10", cat="joins", sql="SELECT o.order_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id", suggest=["Join to customers"], keep=True),
    C(id="C11", cat="joins", sql="SELECT o.order_id FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id AND c.country = 'India'", keep=True),
    C(id="C12", cat="joins", sql="SELECT o.order_id, c.email FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id WHERE c.is_active = 1 AND o.total > 0",
      apply=["LEFT JOIN"]),

    # ── D. aggregation ──────────────────────────────────────────────────────
    C(id="D01", cat="aggregation", sql="SELECT status, COUNT(*) FROM orders GROUP BY status HAVING status <> 'cancelled'", apply=["HAVING to WHERE"]),
    C(id="D02", cat="aggregation", sql="SELECT status, COUNT(*) FROM orders GROUP BY status HAVING COUNT(*) > 1", keep=True),
    C(id="D03", cat="aggregation", sql="SELECT status AS s, COUNT(*) FROM orders GROUP BY status HAVING s = 'paid'", keep=True),
    C(id="D04", cat="aggregation", sql="SELECT status, COUNT(*) FROM orders GROUP BY status WITH ROLLUP HAVING status = 'paid'", keep=True),
    C(id="D05", cat="aggregation", sql="SELECT DISTINCT status, COUNT(*) FROM orders GROUP BY status", apply=["redundant DISTINCT"]),
    C(id="D06", cat="aggregation", sql="SELECT DISTINCT COUNT(*) FROM orders GROUP BY status", keep=True),
    C(id="D07", cat="aggregation", sql="SELECT customer_id, SUM(total), COUNT(total), COUNT(*), AVG(total), MIN(total), MAX(total) FROM orders GROUP BY customer_id", keep=True),
    C(id="D08", cat="aggregation", sql="SELECT c.customer_id, c.email, COUNT(o.order_id) FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.email",
      suggest=["GROUP BY"], keep=True),
    C(id="D09", cat="aggregation", sql="SELECT COUNT(DISTINCT status) FROM orders", keep=True),

    # ── E. subqueries ───────────────────────────────────────────────────────
    C(id="E01", cat="subqueries", sql="SELECT c.customer_id FROM customers c WHERE (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) > 0",
      apply=["COUNT(...) > 0 with EXISTS"]),
    C(id="E02", cat="subqueries", sql="SELECT c.customer_id FROM customers c WHERE (SELECT COUNT(o.total) FROM orders o WHERE o.customer_id = c.customer_id) > 0", keep=True),
    C(id="E03", cat="subqueries", sql="SELECT c.customer_id FROM customers c WHERE (SELECT COUNT(DISTINCT o.status) FROM orders o WHERE o.customer_id = c.customer_id) >= 1", keep=True),
    C(id="E04", cat="subqueries", sql="SELECT c.customer_id FROM customers c WHERE (SELECT COUNT(1) FROM orders o WHERE o.customer_id = c.customer_id) >= 1",
      apply=["COUNT(...) > 0 with EXISTS"]),
    C(id="E05", cat="subqueries", sql="SELECT order_id FROM orders WHERE customer_id IN (SELECT DISTINCT customer_id FROM customers WHERE is_active = 1)",
      apply=["DISTINCT inside"]),
    C(id="E06", cat="subqueries", sql="SELECT order_id FROM orders WHERE customer_id IN (SELECT customer_id FROM customers WHERE country = 'India' UNION SELECT customer_id FROM customers WHERE is_active = 1)",
      apply=["UNION ALL inside"]),
    C(id="E07", cat="subqueries", sql="SELECT order_id FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers UNION SELECT order_id FROM orders)",
      suggest=["NOT EXISTS instead of NOT IN"], apply=["UNION ALL inside"]),
    C(id="E08", cat="subqueries", sql="SELECT t.order_id FROM (SELECT order_id, total FROM orders ORDER BY total) AS t", apply=["ORDER BY from a subquery"]),
    C(id="E09", cat="subqueries", sql="SELECT t.order_id FROM (SELECT order_id, total FROM orders ORDER BY total DESC LIMIT 3) AS t", keep=True),
    C(id="E10", cat="subqueries", sql="WITH a AS (SELECT order_id FROM orders), b AS (SELECT customer_id FROM customers) SELECT customer_id FROM b", apply=["unused CTE"]),
    C(id="E11", cat="subqueries", sql="WITH RECURSIVE n AS (SELECT 1 AS k UNION ALL SELECT k + 1 FROM n WHERE k < 5), unused AS (SELECT 1) SELECT k FROM n", keep=True),
    C(id="E12", cat="subqueries", sql=(
        "SELECT c.customer_id,"
        " (SELECT COUNT(*) FROM orders o1 WHERE o1.customer_id = c.customer_id) AS n_orders,"
        " (SELECT COUNT(*) FROM orders o2 WHERE o2.customer_id = c.customer_id AND o2.status = 'paid') AS n_paid,"
        " (SELECT SUM(o3.total) FROM orders o3 WHERE o3.customer_id = c.customer_id AND o3.status = 'paid') AS paid_total,"
        " (SELECT AVG(o4.total) FROM orders o4 WHERE o4.customer_id = c.customer_id) AS avg_total,"
        " (SELECT MAX(o5.created_at) FROM orders o5 WHERE o5.customer_id = c.customer_id) AS last_order,"
        " (SELECT COUNT(DISTINCT o6.status) FROM orders o6 WHERE o6.customer_id = c.customer_id AND o6.total > 5) AS n_status "
        "FROM customers c WHERE c.is_active = 1"), suggest=["Merge 6 correlated", "read many times"], keep=True, verify_suggestion="Merge"),
    C(id="E13", cat="subqueries", sql=(
        "SELECT c.customer_id,"
        " (SELECT SUM(i.quantity) FROM orders o JOIN order_items i ON i.order_id = o.order_id WHERE o.customer_id = c.customer_id) AS q,"
        " (SELECT MAX(o2.total) FROM orders o2 WHERE o2.customer_id = c.customer_id) AS m FROM customers c"),
      suggest=["Subquery in the SELECT list"], keep=True),
    C(id="E14", cat="subqueries", sql=(
        "SELECT c.customer_id,"
        " (SELECT MAX(o1.total) FROM orders o1 WHERE o1.customer_id = c.customer_id ORDER BY o1.total LIMIT 1) AS a,"
        " (SELECT MIN(o2.total) FROM orders o2 WHERE o2.customer_id = c.customer_id) AS b FROM customers c"), keep=True, forbid=["Merge"]),

    # ── F. indexes ──────────────────────────────────────────────────────────
    C(id="F01", cat="indexes", sql="SELECT order_id FROM orders WHERE customer_id = 3 AND created_at >= '2025-01-01'", suggest=["Indexes"], keep=True,
      index_has=["orders (customer_id, created_at)"]),
    C(id="F02", cat="indexes", sql="SELECT customer_id FROM customers WHERE is_active = 1", keep=True, index_lacks=["customers (is_active)"]),
    C(id="F03", cat="indexes", sql="SELECT p.name FROM products p WHERE p.product_id = 7", keep=True, index_lacks=["products (product_id)"]),
    C(id="F04", cat="indexes", sql="SELECT order_id FROM orders WHERE customer_id = 9", keep=True, index_lacks=["CREATE INDEX idx_orders_customer_id ON"]),
    C(id="F05", cat="indexes", sql="SELECT order_id FROM orders WHERE status = 'paid' AND total > 100 ORDER BY created_at", keep=True,
      index_has=["orders (status, total)"]),

    # ── G. sorting & pagination ─────────────────────────────────────────────
    C(id="G01", cat="sorting", sql="SELECT order_id FROM orders ORDER BY created_at", suggest=["Sorting the full result"], keep=True),
    C(id="G02", cat="sorting", sql="SELECT order_id FROM orders ORDER BY created_at DESC LIMIT 10", keep=True, forbid=["Sorting the full result"]),
    C(id="G03", cat="sorting", sql="SELECT order_id FROM orders ORDER BY order_id LIMIT 20 OFFSET 50000", suggest=["Large OFFSET"], keep=True),
    C(id="G04", cat="sorting", sql="SELECT order_id FROM orders LIMIT 10", suggest=["LIMIT without ORDER BY"], keep=True),
    C(id="G05", cat="sorting", sql="SELECT order_id FROM orders ORDER BY RAND() LIMIT 5", suggest=["ORDER BY random"], keep=True),
    C(id="G06", cat="sorting", sql="SELECT customer_id, SUM(total) AS s FROM orders GROUP BY customer_id ORDER BY s DESC", keep=True),

    # ── H. advanced ─────────────────────────────────────────────────────────
    C(id="H01", cat="advanced", sql="SELECT status FROM orders UNION SELECT country FROM customers", suggest=["UNION removes duplicates"], keep=True),
    C(id="H02", cat="advanced", sql="SELECT order_id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS rn FROM orders", keep=True),
    C(id="H03", cat="advanced", sql="SELECT order_id, CASE WHEN total > 100 THEN CASE WHEN status = 'paid' THEN 'big-paid' ELSE 'big' END ELSE 'small' END AS band FROM orders", keep=True),
    C(id="H04", cat="advanced", sql="SELECT order_id FROM orders o WHERE EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.order_id AND i.quantity > 1 AND i.unit_price > 10)", keep=True),
    C(id="H05", cat="advanced", sql="SELECT x.customer_id, x.s FROM (SELECT customer_id, SUM(total) AS s FROM orders WHERE YEAR(created_at) = 2025 GROUP BY customer_id) AS x WHERE x.s > 100",
      apply=["date filter index-friendly"]),
    C(id="H06", cat="advanced", sql="SELECT order_id FROM orders WHERE (status = 'paid' OR status = 'shipped') AND (total > 10 OR total IS NULL) AND YEAR(created_at) = 2025",
      apply=["Combined OR conditions into IN", "date filter index-friendly"]),
]
