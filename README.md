# الشاعر — Setup from scratch

End-to-end setup for the Arabic poetry ETL + qafiya annotation workflow on a
fresh machine. Gets you to: **241,964 poems loaded · 6.98M verses diacritized
· rules r001–r003 applied · annotator UI running at localhost:8501**.

---

## Prerequisites

- **Docker** (with Compose v2)
- **uv** — Python package manager
  - Install once: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **git** + SSH access to the repo

No Python needed on the host — `uv` manages its own interpreter.

## 1. Clone and install

```bash
git clone git@github.com:qomra/poet.git
cd poet
make install      # resolves and installs all workspace packages into ~/.venvs/alshaer
```

## 2. Start infrastructure

```bash
make up
```

This brings up three containers:

| service           | port | purpose                                    |
|-------------------|------|--------------------------------------------|
| `alshaer-postgres`      | 5433 | main data store (poems, verses, rules)      |
| `alshaer-qdrant`        | 6333 | vector store (reserved — unused currently) |
| `alshaer-qafiya-annotator` | 8501 | web UI for rule proposal + tashkeel toggle |

## 3. Apply database migrations

```bash
make migrate
```

Creates: `poets`, `poems`, `verses`, `rules`, `qafiya_annotations`, `videos`,
`video_segments`, `sessions`, `messages`, plus the qafiya classification
columns on `poems` and the `text_tashkeel` column on `verses`.

## 4. Download the `arbml/ashaar` dataset

```bash
uv run etl ashaar-download
```

Pulls the 254,630-record Arrow dataset from HuggingFace into
`dataset/ashaar/`. ~100 MB on disk. No auth token needed (public dataset).

## 5. ETL into Postgres

```bash
make etl-ashaar
```

Normalizes, dedupes, and inserts into `poets` / `poems` / `verses`.
End state: ~7,139 poets · ~241,964 poems · ~6.98M verses.

## 6. Pull pre-computed tashkeel (diacritization)

```bash
uv run etl tashkeel-pull
```

Downloads `mysamai/ashaar-tashkeel` (~287 MB) from HuggingFace and applies the
`text_tashkeel` column to every row in `verses`. This replaces the ~41-hour
GPU inference run with a ~5-minute bulk UPDATE.

> To re-generate it yourself: see `tools/tashkeel_service/README.md`.

## 7. Seed and apply rules

```bash
uv run etl rules-seed          # inserts r001, r002, r003 into the rules table
uv run etl qafiya-apply --all  # runs each rule over the unclassified pool
```

End state: ~63k poems classified by r001–r003. The remaining ~127k await new
rules (or r004, once defined).

### How rules travel between machines

The **code** is the source of truth. Each rule lives at
`etl/src/etl/qafiya_rules/rNNN_slug.py` as a `RULE = Rule(...)` object — the
match function, the Arabic title, and the prose description are all in that
file, committed to git.

`rules-seed` reads those files and upserts each one into the local `rules`
table. `qafiya-apply` then runs the match function, populating classification
results in `poems` and match counts back in `rules`. The DB row is a
runtime-state pointer; the logic is the git file.

**What does _not_ migrate via git:**
- Rules a user proposed through the UI but no-one has coded yet
  (`status='proposed'`, no `.py` file). These are DB-local and will be lost
  if you skip exporting the `rules` table. Check first:
  `docker exec alshaer-postgres psql -U alshaer -d alshaer -c "SELECT code, title_ar FROM rules WHERE status='proposed';"`
- Poem-level annotations under `qafiya_annotations` (we don't use that table
  in the current workflow, but if you did, back it up separately).

## 8. Open the annotator

<http://localhost:8501>

- Left column: a random unclassified poem.
- Middle column: "propose a rule" form (title + Arabic description + expected
  qafiya fields for the sample).
- Right column, tab 1: qafiya-theory reference. Tab 2: list of proposed rules
  (copy / edit / delete).
- Header: **تشكيل** toggle to flip the poem between raw and diacritized text.

## Making further progress (the rule-writing loop)

1. Open the UI, pick a random poem.
2. If you spot a pattern, hit **➕ أضف القاعدة** — it lands in `rules` with
   `status='proposed'`.
3. Tell the maintainer: *"code rule X"*. They:
   - Write `etl/src/etl/qafiya_rules/rNNN_slug.py` exporting `RULE = Rule(...)`.
   - `uv run etl rules-seed` to register it (sets `status='coded'`).
   - `uv run etl qafiya-apply rNNN` to apply it.
4. Refresh the UI — the unclassified pool shrinks, new poem, repeat.

## Useful CLI commands

| command                                    | purpose                                             |
|--------------------------------------------|-----------------------------------------------------|
| `uv run etl ashaar-download`               | pull `arbml/ashaar` from HF                         |
| `uv run etl ashaar`                        | run the ashaar ETL pipeline                         |
| `uv run etl tashkeel-pull`                 | pull pre-computed tashkeel from HF, apply           |
| `uv run etl tashkeel-export PATH`          | dump NULL-tashkeel rows to a parquet                |
| `uv run etl tashkeel-import PATH`          | apply a parquet of `(id, text_tashkeel)` to verses  |
| `uv run etl rules-seed`                    | upsert discovered rules from code to the DB         |
| `uv run etl qafiya-apply <code>`           | apply a single rule over the unclassified pool      |
| `uv run etl qafiya-apply --all`            | apply every rule in code-name order                 |
| `uv run etl qafiya-reapply <code>`         | reset a rule's matches + re-apply                   |
| `uv run etl qafiya-reset`                  | NULL every poem's qafiya_* fields + qafiya_rule_id  |

## Running tashkeel generation on your own GPU (optional)

Only needed if you want to re-generate the tashkeel column (e.g. with a
different model or updated hyperparameters). See
[`tools/tashkeel_service/README.md`](tools/tashkeel_service/README.md).

Rough recipe: copy the `dataset/ashaar` parquet + this repo to a CUDA box,
build `poet-tashkeel`, run `bulk_tashkeel.py` with `--batch-size 128
--max-new-tokens 256 --max-input-len 128`, then `tashkeel-import` the result
parquet on the DB host.

## Data sources

- **`arbml/ashaar`** — poetry corpus, public on HuggingFace.
- **`mysamai/ashaar-tashkeel`** — companion diacritized parquet (this repo
  uploads it; anyone can pull).
- **`basharalrfooh/Fine-Tashkeel`** — the ByT5-Large diacritization model used
  to generate the tashkeel parquet.

## Troubleshooting

- **`make migrate` fails with a connection error** — wait a few seconds after
  `make up` for Postgres to finish its healthcheck.
- **`etl tashkeel-pull` hangs at 0 bytes** — transient HF rate-limit. Kill and
  retry; the cache is at `~/.cache/huggingface/hub/datasets--mysamai--…` so
  partial downloads resume.
- **UI tashkeel toggle is greyed out** — that poem hasn't been diacritized yet
  (only happens if you ran `tashkeel-pull` with a partial parquet). Check
  `SELECT count(*) FILTER (WHERE text_tashkeel IS NULL) FROM verses;`
