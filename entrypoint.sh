#!/bin/bash
set -e

# Install ms-inspect from the volume-mounted source.
# This runs at container start so the live source is always used.
if [ -f /opt/ms-inspect/pyproject.toml ]; then
    echo "[entrypoint] Installing ms-inspect from /opt/ms-inspect..."
    pip install --quiet --no-cache-dir --upgrade -e "/opt/ms-inspect[casa]"
    echo "[entrypoint] ms-inspect installed."
else
    echo "[entrypoint] WARNING: /opt/ms-inspect not mounted — ms-inspect will not be available."
fi

# Bootstrap CASA measures data directory so casatools can import cleanly.
# Without this, casatools raises AutoUpdatesNotAllowed on first import and
# ms-inspect starts in a degraded mode where all tools return CASA_NOT_AVAILABLE.
# Creating the directory triggers casatools to download casarundata (~336MB) on
# first run; subsequent starts are instant.
echo "[entrypoint] Ensuring CASA data directory exists..."
mkdir -p /root/.casa/data
echo "[entrypoint] CASA data directory ready."

exec supervisord -n -c /etc/supervisord.conf
