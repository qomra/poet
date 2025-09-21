MODEL="meta-llama/Meta-Llama-3-8B"
PORT=8000
HOST=127.0.0.1
tp=4
dp=1
pp=1
dtype="bfloat16"
python3 -m vllm.entrypoints.openai.api_server --model $MODEL --runner auto --convert auto --tokenizer-mode auto --dtype $dtype --max-model-len 2048 --max-logprobs 20  --pipeline-parallel-size $pp --tensor-parallel-size $tp --data-parallel-size $dp --host $HOST --port $PORT