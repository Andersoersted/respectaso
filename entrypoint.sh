#!/bin/bash
set -e

# Auto-generate SECRET_KEY on first run
if [ -z "$SECRET_KEY" ]; then
    if [ -f /app/data/.secret_key ]; then
        export SECRET_KEY=$(cat /app/data/.secret_key)
    else
        export SECRET_KEY=$(python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
        mkdir -p /app/data
        echo "$SECRET_KEY" > /app/data/.secret_key
    fi
fi

if python manage.py migrate --check --noinput >/dev/null 2>&1; then
    echo "No pending migrations."
else
    echo "Applying pending migrations..."
    max_attempts=5
    attempt=1
    until python manage.py migrate --noinput; do
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "Migration failed after ${max_attempts} attempts."
            exit 1
        fi
        sleep_seconds=$((attempt * 2))
        echo "Migration attempt ${attempt}/${max_attempts} failed. Retrying in ${sleep_seconds}s..."
        sleep "$sleep_seconds"
        attempt=$((attempt + 1))
    done
fi

python manage.py collectstatic --noinput

echo ""
echo "============================================"
echo ""
echo "  RespectASO is ready!"
echo ""
echo "  → http://localhost"
echo ""
echo "============================================"
echo ""

exec "$@"
