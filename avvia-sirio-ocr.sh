#!/usr/bin/env bash
# Sirio OCR - avvio (Linux). Al primo avvio scarica automaticamente Python e tutte le dipendenze.
cd "$(dirname "$0")" || exit 1
exec bash "./launcher/avvia.sh" "$@"
