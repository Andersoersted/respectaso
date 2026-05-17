#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────
# RespectASO — Migrate data from Docker to native macOS app
#
# This script copies your SQLite database from the Docker volume
# to ~/Library/Application Support/RespectASO/ so you can
# continue using all your data in the native app.
#
# Usage:
#   chmod +x migrate-from-docker.sh
#   ./migrate-from-docker.sh
# ─────────────────────────────────────────────────────────────

NATIVE_DIR="$HOME/Library/Application Support/RespectASO"
DOCKER_VOLUME="respectaso_aso_data"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  RespectASO — Docker to Native App Migration     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH."
    echo "   This script needs Docker to access your existing data."
    exit 1
fi

# Check if the Docker volume exists
if ! docker volume inspect "$DOCKER_VOLUME" &> /dev/null; then
    echo "❌ Docker volume '$DOCKER_VOLUME' not found."
    echo "   It seems RespectASO was never run with Docker, or the volume was deleted."
    echo ""
    echo "   You can start the native app fresh — no migration needed."
    exit 1
fi

# Create native data directory
mkdir -p "$NATIVE_DIR"

# Check if native app already has data
if [ -f "$NATIVE_DIR/db.sqlite3" ]; then
    echo "⚠️  A database already exists at:"
    echo "   $NATIVE_DIR/db.sqlite3"
    echo ""
    read -p "   Overwrite with Docker data? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Migration cancelled. Your existing native app data is unchanged."
        exit 0
    fi
    # Backup existing
    cp "$NATIVE_DIR/db.sqlite3" "$NATIVE_DIR/db.sqlite3.backup.$(date +%Y%m%d%H%M%S)"
    echo "   ✅ Backed up existing database."
fi

echo "📦 Copying database from Docker volume..."

# Copy the SQLite database from the Docker volume
docker run --rm \
    -v "$DOCKER_VOLUME":/source:ro \
    -v "$NATIVE_DIR":/target \
    alpine:latest \
    cp /source/db.sqlite3 /target/db.sqlite3

# Also copy the secret key if it exists
docker run --rm \
    -v "$DOCKER_VOLUME":/source:ro \
    -v "$NATIVE_DIR":/target \
    alpine:latest \
    sh -c 'test -f /source/.secret_key && cp /source/.secret_key /target/.secret_key || true'

echo ""
echo "✅ Migration complete!"
echo ""
echo "   Database copied to: $NATIVE_DIR/db.sqlite3"
echo ""
echo "   Next steps:"
echo "   1. Open the RespectASO app from your Applications folder"
echo "   2. All your apps, keywords, and search history will be there"
echo ""
echo "   To remove Docker data (optional, after verifying):"
echo "   docker compose down"
echo "   docker volume rm $DOCKER_VOLUME"
echo ""
