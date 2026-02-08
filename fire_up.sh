#!/usr/bin/env bash
# Fire up the Quantum Seeker 2.0 report server

echo "Generating latest reports..."
if [ -f "config.json" ]; then
    python run_analysis.py --config config.json
else
    python run_analysis.py
fi

echo "Starting report server at http://localhost:8000"
echo "Press Ctrl+C to stop."

python -m http.server 8000 --directory output
