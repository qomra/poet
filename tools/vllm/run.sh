#!/bin/bash

MODEL="meta-llama/Meta-Llama-3-8B"
PORT=8000
HOST=127.0.0.1
tp=4
dp=1
pp=1
dtype="bfloat16"
max_model_len=8192

# Default values
BACKGROUND=false
OUTPUT_FILE=""
LOG_FILE="vllm.log"

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -b, --background    Run in background"
            echo "  -o, --output FILE   Map output to file (default: stdout)"
            echo "  -l, --log FILE      Log file for background mode (default: vllm.log)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python3 -m vllm.entrypoints.openai.api_server --model $MODEL --runner auto --convert auto --tokenizer-mode auto --dtype $dtype --max-model-len $max_model_len  --pipeline-parallel-size $pp --tensor-parallel-size $tp --data-parallel-size $dp --host $HOST --port $PORT"

# Add output redirection if specified
if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD > $OUTPUT_FILE 2>&1"
fi

# Execute based on mode
if [ "$BACKGROUND" = true ]; then
    echo "Starting vLLM server in background..."
    echo "Log file: $LOG_FILE"
    echo "PID file: vllm.pid"
    
    # Run in background and save PID
    nohup $CMD > "$LOG_FILE" 2>&1 &
    echo $! > vllm.pid
    
    echo "vLLM server started with PID: $(cat vllm.pid)"
    echo "To stop the server: kill \$(cat vllm.pid)"
    echo "To view logs: tail -f $LOG_FILE"
else
    echo "Starting vLLM server in foreground..."
    eval $CMD
fi