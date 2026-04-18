# Fine-Tashkeel service

FastAPI wrapper around [basharalrfooh/Fine-Tashkeel](https://huggingface.co/basharalrfooh/Fine-Tashkeel)
(ByT5-Large, classical-Arabic-tuned diacritization).

## Endpoints

- `GET  /health` — liveness + model info
- `POST /diacritize` — body `{ "texts": ["...", "..."] }` → `{ "diacritized": [...] }`

## Run (GPU host)

```bash
docker build -t poet-tashkeel .

docker run -d --name poet-tashkeel \
    --gpus '"device=0"' \
    --restart unless-stopped \
    -e DEVICE=cuda:0 \
    -e DTYPE=float16 \
    -p 8502:8502 \
    poet-tashkeel
```

First start pulls the model (~2.5 GB in fp16) into `/root/.cache/huggingface` inside the container.

## Smoke test

```bash
curl -s http://localhost:8502/health
curl -s http://localhost:8502/diacritize \
     -H 'content-type: application/json' \
     -d '{"texts":["قال الشاعر والشعر ديوان العرب"]}'
```

## Environment

| var | default | notes |
|---|---|---|
| `MODEL_ID` | `basharalrfooh/Fine-Tashkeel` | any seq2seq Arabic diacritizer |
| `DEVICE` | `cuda:0` | or `cpu` |
| `DTYPE` | `float16` | `bfloat16` / `float32` also supported |
| `DEFAULT_MAX_NEW_TOKENS` | `256` | generation cap |
| `MAX_BATCH` | `64` | server-side batch limit |
