.PHONY: up down status logs clean open prompt restart-trino

# ─────────────────────────────────────────────────────────────────────────────
# make up
#   1. Start MinIO AIStor + Trino
#   2. Install Python dependencies
#   3. Auto-run setup.py  (warehouse → namespace → table → seed data)
#   4. Restart Trino so it connects to the now-existing warehouse
#   5. Print ready message
# ─────────────────────────────────────────────────────────────────────────────
up:
	@echo "Starting MinIO AIStor + Trino..."
	docker compose up -d
	@echo ""
	@echo "Waiting for MinIO to be healthy..."
	@until curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; do \
		printf "."; sleep 2; \
	done
	@echo " ready"
	@echo ""
	@echo "Setting up Python environment..."
	@test -d .venv || uv venv .venv
	@uv pip install -q -r requirements.txt --python .venv/bin/python3
	@echo ""
	@echo "Running base setup (warehouse → namespace → table → seed data)..."
	.venv/bin/python3 setup.py
	@echo ""
	@echo "Restarting Trino to connect to the warehouse..."
	docker compose restart trino > /dev/null 2>&1
	@until curl -sf http://localhost:8090/v1/info > /dev/null 2>&1; do \
		printf "."; sleep 3; \
	done
	@echo " ready"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  ✅  Workshop 01 is ready"
	@echo ""
	@echo "  MinIO Console → http://localhost:9001  (minioadmin / minioadmin)"
	@echo "  Trino Web UI  → http://localhost:8090"
	@echo ""
	@echo "  Next step:"
	@echo "    make prompt   — open OpenCode and use opencode-prompts.md"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─────────────────────────────────────────────────────────────────────────────
# make prompt
#   Open OpenCode in this directory.
#   AGENTS.md loads automatically — all AIStor rules are in context.
#   Use the prompts in opencode-prompts.md.
# ─────────────────────────────────────────────────────────────────────────────
prompt:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  OpenCode starting — AGENTS.md loaded automatically"
	@echo "  Paste prompts from: opencode-prompts.md"
	@echo "  Run generated scripts with: .venv/bin/python3 <script>.py"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	opencode .

# ─────────────────────────────────────────────────────────────────────────────
# Utility targets
# ─────────────────────────────────────────────────────────────────────────────

# Stop containers (keep data volume)
down:
	docker compose down

# Show container status
status:
	docker compose ps

# Stream logs from all containers
logs:
	docker compose logs -f

# Wipe everything — containers + data volume (full clean slate)
clean:
	docker compose down -v
	@echo "Containers stopped and data volume wiped."

# Open service UIs in the browser (macOS)
open:
	open http://localhost:9001
	open http://localhost:8090
