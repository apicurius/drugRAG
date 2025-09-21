#!/bin/bash

# Quick start server (Qwen or Llama) in background tmux window and run evaluation
# Supports sequential evaluation - one model at a time

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║              Quick Start: Sequential Server + Evaluation              ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Select which server to start
echo "Select server to start:"
echo "  1) Qwen (port 8002)"
echo "  2) Llama (port 8003)"
echo ""
read -p "Choice [1-2]: " choice

case $choice in
    1)
        MODEL="qwen"
        PORT=8002
        SCRIPT="./qwen.sh"
        WINDOW_NAME="qwen"
        MODEL_CHECK="Qwen"
        ;;
    2)
        MODEL="llama"
        PORT=8003
        SCRIPT="./llama.sh"
        WINDOW_NAME="llama"
        MODEL_CHECK="Llama"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Clean up any existing servers on selected port
echo "🧹 Cleaning up existing processes on port $PORT..."
pkill -f "port $PORT" 2>/dev/null
sleep 2

# Check if we're in a tmux session
if [ -z "$TMUX" ]; then
    echo "⚠️  Not in a tmux session!"
    echo ""
    echo "Options:"
    echo "  1) Run server in background with nohup"
    echo "  2) Create new tmux session"
    echo ""
    read -p "Choice [1-2]: " tmux_mode

    if [ "$tmux_mode" = "2" ]; then
        echo "Creating new tmux session 'drugrag'..."
        tmux new-session -s drugrag "$0"
        exit 0
    else
        # Run in background without nohup (cluster restriction)
        echo "🚀 Starting $MODEL_CHECK server in background..."
        mkdir -p logs
        $SCRIPT > logs/${MODEL}_server.log 2>&1 &
        SERVER_PID=$!
        disown $SERVER_PID
        echo "   Server PID: $SERVER_PID"
        echo "   Monitor logs: tail -f logs/${MODEL}_server.log"
        echo ""
    fi
else
    # We're in tmux, try to create new window
    echo "🚀 Starting $MODEL_CHECK server in background tmux window..."
    mkdir -p logs

    # Kill existing window if it exists
    tmux kill-window -t $WINDOW_NAME 2>/dev/null || true

    # Create new window
    tmux new-window -d -n $WINDOW_NAME "$SCRIPT 2>&1 | tee logs/${MODEL}_server.log" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "⚠️  Could not create tmux window. Starting in background instead..."
        # Use simple background process without nohup (cluster restriction)
        $SCRIPT > logs/${MODEL}_server.log 2>&1 &
        SERVER_PID=$!
        disown $SERVER_PID
        echo "   Server PID: $SERVER_PID"
        echo "   Monitor logs: tail -f logs/${MODEL}_server.log"
    fi
fi

# Wait for server to be ready
echo "⏳ Waiting for $MODEL_CHECK server to initialize on port $PORT..."
READY=false
for i in {1..60}; do
    if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "$MODEL_CHECK"; then
        echo ""
        echo "✅ $MODEL_CHECK server ready on port $PORT!"
        READY=true
        break
    fi
    sleep 5
    echo -n "."
done

if [ "$READY" = false ]; then
    echo ""
    echo "❌ Server failed to start. Check logs: logs/${MODEL}_server.log"
    echo "   Or view tmux window: tmux select-window -t $WINDOW_NAME"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"

# Show appropriate instructions based on how server was started
if [ -n "$SERVER_PID" ]; then
    echo "Server is running in background (PID: $SERVER_PID)"
    echo ""
    echo "Commands:"
    echo "  • View logs: tail -f logs/${MODEL}_server.log"
    echo "  • Stop server: kill $SERVER_PID"
elif [ -n "$TMUX" ]; then
    echo "Server is running in background tmux window '$WINDOW_NAME'"
    echo ""
    echo "Tmux commands:"
    echo "  • View server output: Ctrl+b, then type window number for '$WINDOW_NAME'"
    echo "  • List windows: tmux list-windows"
    echo "  • Return here: Ctrl+b, then press current window number"
else
    echo "Server is running"
fi

echo ""
echo "Run evaluation commands:"
echo ""
if [ "$MODEL" = "qwen" ]; then
    echo "  # Quick test (10 samples):"
    echo "  ./run_evaluations.sh --llm qwen --query binary --strategy all --test-size-binary 10 --no-auto-start"
    echo ""
    echo "  # Standard test (100 samples):"
    echo "  ./run_evaluations.sh --llm qwen --query binary --strategy all --test-size-binary 100 --no-auto-start"
else
    echo "  # Quick test (10 samples):"
    echo "  ./run_evaluations.sh --llm llama3 --query binary --strategy all --test-size-binary 10 --no-auto-start"
    echo ""
    echo "  # Standard test (100 samples):"
    echo "  ./run_evaluations.sh --llm llama3 --query binary --strategy all --test-size-binary 100 --no-auto-start"
fi
echo ""
echo "To switch servers: Run this script again and select the other model"
echo "═══════════════════════════════════════════════════════════════════════"