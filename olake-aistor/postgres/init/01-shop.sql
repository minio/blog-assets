-- Sample source data, the OLake replication user, and the CDC publication + slot.
-- Runs once, when the Postgres volume is first created.

CREATE TABLE customers (
  id         SERIAL PRIMARY KEY,
  name       VARCHAR(100),
  email      VARCHAR(200),
  city       VARCHAR(100),
  tier       VARCHAR(20),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE orders (
  id          SERIAL PRIMARY KEY,
  customer_id INT REFERENCES customers(id),
  product     VARCHAR(100),
  amount      NUMERIC(10,2),
  status      VARCHAR(20),
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO customers (name, email, city, tier) VALUES
  ('Ava Johnson',  'ava@example.com',   'Austin',    'gold'),
  ('Liam Carter',  'liam@example.com',  'Denver',    'silver'),
  ('Noah Kim',     'noah@example.com',  'Seattle',   'bronze'),
  ('Emma Rossi',   'emma@example.com',  'Milan',     'gold'),
  ('Zara Ali',     'zara@example.com',  'Dubai',     'silver'),
  ('Kenji Sato',   'kenji@example.com', 'Osaka',     'bronze'),
  ('Sofia Lopez',  'sofia@example.com', 'Madrid',    'silver'),
  ('Oliver Brown', 'oliver@example.com','London',    'bronze');

-- 40 orders, 5 per customer
INSERT INTO orders (customer_id, product, amount, status)
SELECT ((g - 1) % 8) + 1,
       (ARRAY['Laptop','Mouse','Monitor','Keyboard','Webcam'])[((g - 1) % 5) + 1],
       round((10 + g * 7.25)::numeric, 2),
       'placed'
FROM generate_series(1, 40) AS g;

-- Log full old rows on UPDATE/DELETE, so unchanged TOAST values aren't lost in CDC.
ALTER TABLE customers REPLICA IDENTITY FULL;
ALTER TABLE orders    REPLICA IDENTITY FULL;

-- OLake replication user: REPLICATION to read the WAL, SELECT for the backfill.
CREATE ROLE olake WITH LOGIN REPLICATION PASSWORD 'olake';
GRANT USAGE ON SCHEMA public TO olake;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO olake;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO olake;

-- One publication and one pgoutput slot per OLake job.
CREATE PUBLICATION olake_pub FOR TABLE customers, orders;
SELECT pg_create_logical_replication_slot('olake_slot', 'pgoutput');
