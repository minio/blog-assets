#!/bin/bash
#
# AIStor Tables + Dremio Enterprise Integration
# Starts MinIO AIStor, Dremio Enterprise, and Jupyter
#

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_header() {
	echo ""
	echo -e "${BLUE}===============================================${NC}"
	echo -e "${BLUE}$1${NC}"
	echo -e "${BLUE}===============================================${NC}"
	echo ""
}

print_msg() {
	local color=$1
	local msg=$2
	echo -e "${color}${msg}${NC}"
}

usage() {
	cat <<EOF
AIStor Tables + Dremio Enterprise Integration

USAGE:
  $0 [options]

DESCRIPTION:
  Starts MinIO AIStor (with Tables), Dremio Enterprise, and Jupyter notebook.

  NOTE: Dremio Enterprise is required for the RESTCATALOG source type.
  Get a 30-day trial at https://www.dremio.com/get-started/

OPTIONS:
  --binary-path PATH     Use pre-built MinIO binary (linux/amd64)
  --stop                 Stop all services
  -h, --help             Show this help message

PREREQUISITES:
  1. Set your AIStor license in .env:
     MINIO_LICENSE=your-license-key-here

  2. Login to quay.io to pull the Dremio Enterprise image:
     docker login quay.io -u '<trial-robot-account>' -p '<trial-token>'

ACCESSING SERVICES:
  MinIO Console:    http://localhost:9001  (minioadmin/minioadmin)
  Dremio UI:        http://localhost:9047  (first-time setup required)
  Jupyter Notebook: http://localhost:8888

EXAMPLES:
  # Start all services
  $0

  # Start with custom MinIO binary
  $0 --binary-path ./minio

  # Stop all services
  $0 --stop

EOF
	exit 0
}

# Parse arguments
BINARY_PATH=""
STOP_MODE=false

while [[ $# -gt 0 ]]; do
	case $1 in
	--binary-path)
		BINARY_PATH="$2"
		shift 2
		;;
	--stop)
		STOP_MODE=true
		shift
		;;
	-h | --help)
		usage
		;;
	*)
		echo "Unknown option: $1"
		usage
		;;
	esac
done

# Stop services
if [ "$STOP_MODE" = true ]; then
	print_header "Stopping AIStor Tables + Dremio Services"
	cd "$PROJECT_ROOT"
	docker compose down
	print_msg "$GREEN" "Services stopped"
	exit 0
fi

# Load .env
load_config() {
	if [ -f "${PROJECT_ROOT}/.env" ]; then
		print_msg "$YELLOW" "Loading configuration from .env..."
		set -a
		source "${PROJECT_ROOT}/.env"
		set +a
	fi
}

# Build from binary if provided
build_from_binary() {
	if [ -z "$BINARY_PATH" ]; then
		return 0
	fi

	print_header "Building Docker Image from Binary"

	if [ ! -f "$BINARY_PATH" ]; then
		print_msg "$RED" "MinIO binary not found at $BINARY_PATH"
		exit 1
	fi

	if ! file "$BINARY_PATH" | grep -q "ELF 64-bit.*x86-64"; then
		print_msg "$RED" "Binary must be built for linux/amd64"
		exit 1
	fi

	print_msg "$GREEN" "Binary verified (linux/amd64)"

	TEMP_DIR=$(mktemp -d)
	cp "$BINARY_PATH" "$TEMP_DIR/minio"

	cat > "$TEMP_DIR/Dockerfile" <<'DOCKERFILE'
FROM ubuntu:22.04
COPY minio /usr/bin/minio
RUN chmod +x /usr/bin/minio
ENTRYPOINT ["/usr/bin/minio"]
DOCKERFILE

	print_msg "$YELLOW" "Building Docker image: aistor-tables-dremio:latest..."
	if docker build -t aistor-tables-dremio:latest "$TEMP_DIR"; then
		print_msg "$GREEN" "Docker image built successfully"
	else
		print_msg "$RED" "Failed to build Docker image"
		rm -rf "$TEMP_DIR"
		exit 1
	fi

	rm -rf "$TEMP_DIR"
	export MINIO_TEST_IMAGE=aistor-tables-dremio:latest
}

# Start services
start_services() {
	print_header "Starting AIStor Tables + Dremio Enterprise"

	cd "$PROJECT_ROOT"

	print_msg "$YELLOW" "Stopping any existing services..."
	docker compose down 2>/dev/null || true

	print_msg "$YELLOW" "Starting MinIO, Dremio Enterprise, and Jupyter..."
	if ! docker compose up -d; then
		print_msg "$RED" "Failed to start services"
		print_msg "$RED" ""
		print_msg "$RED" "If you see 'pull access denied', make sure you've logged into quay.io:"
		print_msg "$RED" "  docker login quay.io -u '<trial-robot-account>' -p '<trial-token>'"
		exit 1
	fi

	# Wait for MinIO
	print_msg "$YELLOW" "Waiting for MinIO to be ready..."
	for i in {1..30}; do
		if curl -s -f http://localhost:9000/minio/health/live >/dev/null 2>&1; then
			print_msg "$GREEN" "MinIO is ready"
			break
		fi
		if [ $i -eq 30 ]; then
			print_msg "$RED" "MinIO failed to start"
			exit 1
		fi
		sleep 2
	done

	# Wait for Dremio (takes longer to start)
	print_msg "$YELLOW" "Waiting for Dremio to be ready (this may take 60-90 seconds)..."
	for i in {1..60}; do
		HTTP_CODE=$(curl -sf -o /dev/null -w '%{http_code}' http://localhost:9047 2>/dev/null || echo "000")
		if echo "$HTTP_CODE" | grep -qE '(200|302|403)'; then
			print_msg "$GREEN" "Dremio is ready"
			break
		fi
		if [ $i -eq 60 ]; then
			print_msg "$RED" "Dremio failed to start"
			print_msg "$RED" "Check logs with: docker compose logs dremio"
			exit 1
		fi
		sleep 3
	done

	# Wait for Jupyter
	print_msg "$YELLOW" "Waiting for Jupyter to be ready..."
	sleep 5

	print_header "All Services Ready!"
	print_msg "$GREEN" "MinIO Console:    http://localhost:9001"
	print_msg "$GREEN" "                  (user: minioadmin, password: minioadmin)"
	print_msg "$GREEN" ""
	print_msg "$GREEN" "Dremio UI:        http://localhost:9047"
	print_msg "$YELLOW" "                  (first login: accept EULA + create admin account)"
	print_msg "$GREEN" ""
	print_msg "$GREEN" "Jupyter Notebook: http://localhost:8888"
	print_msg "$GREEN" "                  (no password required)"
	print_msg "$GREEN" ""
	print_msg "$BLUE" "Next Steps:"
	print_msg "$BLUE" "  1. Open Dremio at http://localhost:9047"
	print_msg "$BLUE" "     - Create your admin account (first-time setup)"
	print_msg "$BLUE" "  2. Open Jupyter at http://localhost:8888"
	print_msg "$BLUE" "  3. Run the DremioAIStorTables.ipynb notebook"
	print_msg "$BLUE" "     - In Step 7, set DREMIO_USER and DREMIO_PASS to match your admin account"
	print_msg "$GREEN" ""
	print_msg "$YELLOW" "To stop services: $0 --stop"
}

# Main
main() {
	load_config
	build_from_binary
	start_services
}

main
