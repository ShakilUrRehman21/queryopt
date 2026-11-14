"""
Browser tests for the single-page optimizer (audit Phase 7).

Runs only when QUERYOPT_E2E=1 and the app is reachable at QUERYOPT_E2E_URL
(default http://localhost:5173, backend on :8000). Uses the bundled Chromium.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("QUERYOPT_E2E") != "1", reason="set QUERYOPT_E2E=1 with the app running")
URL = os.environ.get("QUERYOPT_E2E_URL", "http://localhost:5173")


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def _page(browser, **kw):
    ctx = browser.new_context(**kw)
    ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=URL)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    # external font loads can fail offline; only app errors count
    page.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errors.append(m.text))
    page.goto(URL)
    page.wait_for_selector("text=Make your SQL faster.")
    return page, errors


def _optimize(page, sql, dialect="mysql"):
    page.fill("textarea.editor", sql)
    page.select_option("select[aria-label='SQL dialect']", dialect)
    page.click("button.go")
    page.wait_for_selector(".verdict, .note.err", timeout=10000)


def test_count_matches_applied_items_and_sections(browser):
    page, errors = _page(browser)
    _optimize(page, "SELECT DISTINCT status, COUNT(*) FROM orders WHERE YEAR(created_at) = 2025 AND 1 = 1 "
                    "GROUP BY status HAVING status <> 'x'")
    applied_items = page.locator("section.group:has(h2:text('What we changed')) li.item").count()
    assert applied_items >= 3
    assert page.locator(".verdict b").inner_text() == str(applied_items)
    checking = page.locator("section.group:has(h2:text('Worth checking')) li.item h3").all_inner_texts()
    changed = page.locator("section.group:has(h2:text('What we changed')) li.item h3").all_inner_texts()
    assert not set(checking) & set(changed)
    assert all(t in ("high", "medium", "low") for t in page.locator(".impact").all_inner_texts())
    assert not errors


def test_original_toggle_and_copy(browser):
    page, _ = _page(browser)
    sql = "SELECT a FROM t WHERE x = 1 OR x = 2"
    _optimize(page, sql)
    assert "IN (1, 2)" in page.inner_text("pre.code")
    page.click(".switch button:text('Original')")
    assert page.inner_text("pre.code").strip() == sql
    page.click(".switch button:text('Optimized')")
    page.click("button.ghost:text('Copy')")
    assert page.locator("button.ghost:text('Copied')").count() == 1
    assert "IN (1, 2)" in page.evaluate("navigator.clipboard.readText()")


def test_nothing_to_rewrite_shows_no_count(browser):
    page, _ = _page(browser)
    _optimize(page, "SELECT id FROM users WHERE id = 7 ORDER BY id LIMIT 1")
    assert "Already in good shape" in page.inner_text(".verdict")
    assert page.locator("section.group:has(h2:text('What we changed'))").count() == 0


def test_empty_and_invalid_input(browser):
    page, _ = _page(browser)
    page.fill("textarea.editor", "   ")
    assert page.locator("button.go").is_disabled()
    _optimize(page, "SELEC broken")
    assert "doesn't look like" in page.inner_text(".note.err")
    _optimize(page, "SELECT FROM WHERE (")
    assert "Could not parse" in page.inner_text(".note.err")


def test_large_and_multi_statement_queries(browser):
    page, errors = _page(browser)
    big = "SELECT a FROM t WHERE " + " AND ".join(f"c{i} = {i}" for i in range(400))
    _optimize(page, big)
    assert page.locator(".verdict").count() == 1
    _optimize(page, "SELECT a FROM t WHERE 1 = 1 AND b = 2; SELECT c FROM u WHERE d = 1 OR d = 2")
    code = page.inner_text("pre.code")
    assert code.count("SELECT") == 2 and "IN (1, 2)" in code
    assert not errors


def test_schema_box(browser):
    page, _ = _page(browser)
    page.click("button.schema-toggle")
    page.fill(".schema-box textarea", "CREATE TABLE orders (order_id INT PRIMARY KEY, status VARCHAR(10));")
    _optimize(page, "SELECT * FROM orders")
    assert "SELECT order_id, status" in page.inner_text("section.group")


@pytest.mark.parametrize("vp", [{"width": 390, "height": 844}, {"width": 1440, "height": 900}])
def test_responsive_no_horizontal_scroll(browser, vp):
    page, _ = _page(browser, viewport=vp)
    _optimize(page, "SELECT * FROM orders o WHERE YEAR(o.created_at) = 2025 AND o.customer_id IN "
                    "(SELECT c.customer_id FROM customers c WHERE c.country = 'India') ORDER BY o.created_at DESC")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
