SELECT c.customer_id, c.email,
 (SELECT SUM(oi.quantity * oi.unit_price) FROM orders o2 JOIN order_items oi ON o2.order_id = oi.order_id WHERE o2.customer_id = c.customer_id AND YEAR(o2.created_at) = 2025 AND o2.status != 'cancelled') AS total_spent,
 (SELECT COUNT(*) FROM orders o3 WHERE o3.customer_id = c.customer_id AND YEAR(o3.created_at) = 2025 AND DATE(o3.created_at) <= '2025-12-31') AS total_orders,
 (SELECT MAX(o4.created_at) FROM orders o4 WHERE o4.customer_id = c.customer_id) AS last_order_date
FROM customers c
WHERE LOWER(c.country) = 'india' AND c.is_active = 1
AND c.customer_id IN (SELECT o.customer_id FROM orders o JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id
 WHERE YEAR(o.created_at) = 2025 AND (o.status = 'paid' OR o.status = 'shipped') AND LOWER(p.category) = 'electronics' AND p.price > 1000 AND oi.quantity > 0)
ORDER BY total_spent DESC