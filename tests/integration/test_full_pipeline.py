"""
Integration test for full database pipeline: SQL String -> Execution -> Results.
"""
from pathlib import Path
from common.types import ExecutionStatus
from queryopt.engine import QueryEngine


def test_full_sql_pipeline_crud_and_queries(tmp_path: Path):
    engine = QueryEngine(data_dir=tmp_path, db_name="testdb")

    # 1. CREATE TABLE
    res = engine.execute("CREATE TABLE products (id INTEGER, name TEXT, price FLOAT, category TEXT)")
    assert res.status == ExecutionStatus.SUCCESS

    # 2. INSERT
    res = engine.execute(
        "INSERT INTO products (id, name, price, category) VALUES "
        "(1, 'Laptop', 1200.0, 'Electronics'), "
        "(2, 'Mouse', 25.0, 'Electronics'), "
        "(3, 'Desk', 350.0, 'Furniture'), "
        "(4, 'Chair', 150.0, 'Furniture'), "
        "(5, 'Monitor', 400.0, 'Electronics')"
    )
    assert res.status == ExecutionStatus.SUCCESS

    # 3. Simple SELECT with WHERE
    res = engine.execute("SELECT id, name FROM products WHERE price < 100.0")
    assert res.status == ExecutionStatus.SUCCESS
    assert res.rows == [(2, "Mouse")]

    # 4. ORDER BY and LIMIT
    res = engine.execute("SELECT name, price FROM products ORDER BY price DESC LIMIT 2")
    assert res.status == ExecutionStatus.SUCCESS
    assert res.rows == [("Laptop", 1200.0), ("Monitor", 400.0)]

    # 5. Aggregate with GROUP BY
    res = engine.execute("SELECT category, COUNT(id) AS cnt FROM products GROUP BY category ORDER BY category ASC")
    assert res.status == ExecutionStatus.SUCCESS
    assert len(res.rows) == 2
    assert res.rows[0] == ("Electronics", 3)
    assert res.rows[1] == ("Furniture", 2)
