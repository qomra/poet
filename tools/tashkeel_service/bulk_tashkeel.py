"""
Bulk diacritization runner — intended to run ON the GPU host.

Reads a parquet (columns: id: str, text: str), runs the Fine-Tashkeel model
in-process (no HTTP), writes another parquet (id, text_tashkeel). Idempotent:
skips rows already present in the output parquet.

Usage (on GPU host):
    python bulk_tashkeel.py \
        --input  /data/verses_to_fill.parquet \
        --output /data/verses_tashkeel.parquet \
        --batch-size 512 \
        --max-new-tokens 48 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="basharalrfooh/Fine-Tashkeel")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--flush-every", type=int, default=10_000,
                    help="Rows per output shard write.")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        args.dtype, torch.float32
    )

    print(f"[load] model={args.model} device={args.device} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()

    # ── Resume: skip ids already in output ────────────────────────────────────
    done_ids: set[str] = set()
    if args.output.exists():
        prior = pq.read_table(args.output, columns=["id"])
        done_ids = set(prior.column("id").to_pylist())
        print(f"[resume] {len(done_ids):,} ids already processed")

    # ── Stream input parquet, in-memory batches ───────────────────────────────
    pf = pq.ParquetFile(args.input)
    total_rows = pf.metadata.num_rows
    remaining = total_rows - len(done_ids)
    print(f"[input] {total_rows:,} rows total — {remaining:,} to process")

    # Collect output rows; flush every N
    out_ids: list[str] = []
    out_diacs: list[str] = []
    processed = 0
    started = time.time()
    last_print = started

    def _flush() -> None:
        if not out_ids:
            return
        tbl = pa.table({"id": out_ids, "text_tashkeel": out_diacs})
        if args.output.exists():
            # append by reading + concatenating; for simplicity write new file per flush
            existing = pq.read_table(args.output)
            tbl = pa.concat_tables([existing, tbl])
        pq.write_table(tbl, args.output, compression="zstd")
        out_ids.clear()
        out_diacs.clear()

    try:
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=["id", "text"]):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()

            # Filter resumable
            work_ids: list[str] = []
            work_texts: list[str] = []
            for i, t in zip(ids, texts):
                if i in done_ids:
                    continue
                work_ids.append(i)
                work_texts.append(t)

            if not work_ids:
                continue

            enc = tok(work_texts, return_tensors="pt", padding=True, truncation=True).to(args.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            decoded = [
                tok.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for seq in gen
            ]

            out_ids.extend(work_ids)
            out_diacs.extend(decoded)
            processed += len(work_ids)

            now = time.time()
            if now - last_print > 10:
                rate = processed / (now - started)
                eta = (remaining - processed) / rate if rate else float("inf")
                print(
                    f"[progress] {processed:,}/{remaining:,} "
                    f"({100*processed/remaining:.1f}%) "
                    f"{rate:.1f} v/s — ETA {eta/3600:.1f}h",
                    flush=True,
                )
                last_print = now

            if len(out_ids) >= args.flush_every:
                _flush()

        _flush()
    except KeyboardInterrupt:
        print("\n[interrupt] flushing partial output")
        _flush()
        sys.exit(1)

    elapsed = time.time() - started
    print(f"[done] {processed:,} rows in {elapsed:.1f}s ({processed/elapsed:.1f} v/s)")


if __name__ == "__main__":
    main()
