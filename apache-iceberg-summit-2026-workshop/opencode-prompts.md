# Workshop 01 — OpenCode Prompts

## Start here

```bash
make clean       # full wipe
make up          # starts MinIO + Trino, creates warehouse/namespace/table, inserts 15 seed rows
make prompt      # opens OpenCode — AGENTS.md loads automatically
```

## Prompt 0 - set the context
```
After each prompt below, OpenCode creates a `.py` file.  
Run it with .venv/bin/python3 prompt-output/<filename>.py
```
```bash
.venv/bin/python3 prompt-output/<filename>.py
```

---

## Prompt 1 — Explore the table

```
Connect to retail.sales using AGENTS.md patterns. Scan all rows and print:
total row count, first 5 rows as a DataFrame, and column names from the schema.
Save as prompt-output/explore.py
```

```bash
.venv/bin/python3 prompt-output/explore.py
```

---

## Prompt 2 — Orders and revenue per region

```
Connect to retail.sales. Scan all rows, group by region, count orders and
sum total_amount per region. Print sorted by order count descending.
Save as prompt-output/by_region.py
```

```bash
.venv/bin/python3 prompt-output/by_region.py
```

---

## Prompt 3 — Revenue by product category

```
Connect to retail.sales. Group by category: order_count, total_revenue (sum),
avg_order_value (mean). Format revenue as USD. Sort by total_revenue descending.
Save as prompt-output/by_category.py
```

```bash
.venv/bin/python3 prompt-output/by_category.py
```

---

## Prompt 4 — Filter with PyIceberg

```
Connect to retail.sales. Use PyIceberg row_filter to scan only rows where
category == 'Electronics' AND region == 'East'.
Print the count and show customer_name, product, total_amount.
Save as prompt-output/filter_east_electronics.py
```

```bash
.venv/bin/python3 prompt-output/filter_east_electronics.py
```

---

## Prompt 5 — Trino: connect and run SQL

```
Connect to Trino at localhost:8090, user=admin, catalog=iceberg, schema=retail.
Write trino_query(sql) returning a pandas DataFrame.
Run these 3 queries and print results:
  1. SELECT region, COUNT(*) orders FROM iceberg.retail.sales GROUP BY region ORDER BY 2 DESC
  2. SELECT category, ROUND(SUM(total_amount),2) revenue FROM iceberg.retail.sales GROUP BY category ORDER BY 2 DESC
  3. SELECT rep_name, COUNT(*) orders, ROUND(SUM(total_amount),2) revenue FROM iceberg.retail.sales GROUP BY rep_name ORDER BY 3 DESC
Save as prompt-output/trino_analytics.py
```

```bash
.venv/bin/python3 prompt-output/trino_analytics.py
```

---

## Prompt 6 — Trino: sales rep leaderboard with RANK()

```
Connect to Trino at localhost:8090. Write trino_query(sql).
SQL: rank sales reps by total revenue using
RANK() OVER (ORDER BY SUM(total_amount) DESC).
Show rep_name, orders, total_revenue, rank from iceberg.retail.sales.
Save as prompt-output/leaderboard.py
```

```bash
.venv/bin/python3 prompt-output/leaderboard.py
```

---

## Prompt 7 — Trino: monthly revenue trend

```
Connect to Trino at localhost:8090. Write trino_query(sql).
SQL: extract YYYY-MM from sale_date (string), compute order_count and
ROUND(SUM(total_amount),2) total_revenue per month from iceberg.retail.sales.
Sort chronologically.
Save as prompt-output/monthly_trend.py
```

```bash
.venv/bin/python3 prompt-output/monthly_trend.py
```

---

## Prompt 8 — Trino: top 5 orders

```
Connect to Trino at localhost:8090. Write trino_query(sql).
SQL: the 5 orders with the highest total_amount from iceberg.retail.sales.
Show order_id, customer_name, product, total_amount, rep_name, region.
Save as prompt-output/top_orders.py
```

```bash
.venv/bin/python3 prompt-output/top_orders.py
```

---

## Prompt 9 — Trino: region × category cross-tab

```
Connect to Trino at localhost:8090. Write trino_query(sql).
SQL: total revenue and order_count for every combination of region and category
from iceberg.retail.sales. Sort by region, then total_revenue DESC.
Save as prompt-output/cross_tab.py
```

```bash
.venv/bin/python3 prompt-output/cross_tab.py
```

---

## Prompt 10 — Trino: revenue % per region

```
Connect to Trino at localhost:8090. Write trino_query(sql).
SQL: each region's revenue as a percentage of the grand total.
Show region, total_revenue, revenue_pct rounded to 1dp, sorted DESC.
Use a subquery for the grand total. From iceberg.retail.sales.
Save as prompt-output/revenue_pct.py
```

```bash
.venv/bin/python3 prompt-output/revenue_pct.py
```

---

## Prompt 11 — Sales dashboard (4 charts)

```
Connect to retail.sales. Scan all rows into df_all.
Using matplotlib, create a 2x2 figure:
  top-left:  horizontal bar — total revenue by region
  top-right: horizontal bar — order count by category
  bot-left:  horizontal bar — revenue by sales rep
  bot-right: vertical bar   — monthly revenue (sale_date[:7])
Add value labels to each bar. Use tight_layout(). Save to sales_dashboard.png.
Save script as dashboard.py
```

```bash
.venv/bin/python3 prompt-output/dashboard.py
open sales_dashboard.png
```

---

## Prompt 12 — Add more data

```
Connect to retail.sales using AGENTS.md patterns.
Append 10 new rows for Apr-Jun 2025 covering all 4 regions,
all 3 categories, all 4 reps (Sarah Chen, Mike Torres, James Wilson, Anna Park).
Use order_ids starting at 2001. Only append — do not touch existing rows.
Print the new total row count after inserting.
Save as prompt-output/add_data.py
```

```bash
.venv/bin/python3 prompt-output/add_data.py
```

---

## Prompt 13 — Snapshot history and time travel

```
Connect to retail.sales via PyIceberg.
Print all snapshot IDs, operation type, and total-records for each snapshot.
Then read the table at snapshot index 0 using table.scan(snapshot_id=...).
Compare that row count to the current count. Print both.
Save as prompt-output/time_travel.py
```

```bash
.venv/bin/python3 prompt-output/time_travel.py
```

---

## Prompt 14 — Schema evolution

```
Using PyIceberg schema evolution, add a nullable float64 column called
discount_pct to the retail.sales table.
Print the updated schema to confirm the new column exists.
Save as prompt-output/add_column.py
```

```bash
.venv/bin/python3 prompt-output/add_column.py
```

---

## Prompt 15 — Create retail.customers and join in Trino

```
Connect to the AIStor catalog using AGENTS.md patterns.
Create a new table retail.customers with columns:
  customer_name (string), tier (string), since_date (string)
Derive one row per unique customer_name from retail.sales.
Assign tiers based on total spend: >$3000=Gold, >$1500=Silver, else=Bronze.
Then write a Trino SQL query joining iceberg.retail.customers to
iceberg.retail.sales on customer_name. Show tier, order_count, total_revenue
grouped by tier, sorted by total_revenue DESC.
Save as prompt-output/customers.py
```

```bash
.venv/bin/python3 prompt-output/customers.py
```

---

## Challenges

**Challenge 1 — Which customer has spent the most?**
```
Connect to Trino at localhost:8090.
Trino SQL: top 5 customers by total spend in iceberg.retail.sales.
Show customer_name, order_count, total_spend sorted DESC.
Save as prompt-output/challenge1.py
```

**Challenge 2 — Who dominates the East region?**
```
Connect to Trino at localhost:8090.
Trino SQL: for region='East' only, show rep_name, order_count, total_revenue
from iceberg.retail.sales. Sort by total_revenue DESC.
Save as prompt-output/challenge2.py
```

**Challenge 3 — Which product sells the most units?**
```
Connect to Trino at localhost:8090.
Trino SQL: group by product, sum quantity as total_units_sold and
sum total_amount as revenue from iceberg.retail.sales.
Sort by total_units_sold DESC. Top 5.
Save as prompt-output/challenge3.py
```

**Challenge 4 — Materialise a summary table**
```
Connect to Trino at localhost:8090.
Trino SQL: CREATE TABLE iceberg.retail.revenue_summary AS
SELECT region, category, COUNT(*) as orders,
       ROUND(SUM(total_amount),2) as revenue
FROM iceberg.retail.sales
GROUP BY region, category
ORDER BY region, revenue DESC.
Save as prompt-output/challenge4.py
```

---

## When done

```bash
make clean    # stops containers and wipes all data
```

---

## Browse your data

At any point open the MinIO Console to see the Parquet files and Iceberg metadata  
that your scripts created inside the `saleswarehouse` bucket:

```
http://localhost:9001   (minioadmin / minioadmin)
```

Trino query planner UI:

```
http://localhost:8090
```
