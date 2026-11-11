-- Changes made between the first and second sync: 7 inserts, 5 updates.
-- The second (CDC) sync should replay all 12 from the WAL.

INSERT INTO customers (name, email, city, tier) VALUES
  ('Maya Patel',  'maya@example.com',  'Pune',      'silver'),
  ('Lucas Silva', 'lucas@example.com', 'São Paulo', 'bronze');

INSERT INTO orders (customer_id, product, amount, status) VALUES
  (9,  'Laptop',   1299.00, 'placed'),
  (9,  'Mouse',      25.50, 'placed'),
  (10, 'Monitor',   349.99, 'placed'),
  (3,  'Keyboard',   89.00, 'placed'),
  (7,  'Webcam',     79.90, 'placed');

UPDATE customers SET tier = 'gold',      updated_at = NOW() WHERE name = 'Liam Carter';
UPDATE customers SET city = 'Abu Dhabi', updated_at = NOW() WHERE name = 'Zara Ali';
UPDATE orders    SET status = 'delivered', updated_at = NOW() WHERE id IN (1, 4, 7);
