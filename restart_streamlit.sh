#!/bin/bash

# Find and kill any existing Streamlit process using port 8501
PID=$(lsof -ti :8501)
if [ ! -z "$PID" ]; then
    echo "Stopping existing Streamlit process on port 8501..."
    kill -9 $PID
    sleep 2  # Wait for process to stop
fi

# Restart Streamlit
echo "Starting Streamlit..."
streamlit run streamlit_app.py --server.port 8501

