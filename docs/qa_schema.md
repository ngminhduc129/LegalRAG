# QA dataset schema

The evaluation set is a JSONL file, one question per line. It is frozen once
used for a comparison: do not edit the test split while tuning. Files under
`data/qa/*.jsonl` are git-ignored because they contain annotation work.

## Required fields

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | unique within the file (`qid`, `question_id`, `query_id` also accepted) |
| `question` | string | the question as asked (`query` or `text` also accepted) |
| `relevant` | list | gold references; at least one (`relevant_articles`, `gold_articles`, `articles`, `relevant_article_ids`, `gold` also accepted) |

Each reference is either a string (`"labor_code_2019:113"` or `"113"`) or an
object:

```json
{"document_id": "labor_code_2019", "article": "113", "clause": "1", "point": "a"}
```

`article` accepts `"Điều 113"` or `113`; only the number is used for matching.
`clause` and `point` are optional. `document_id` must match
`configs/corpus/documents.yaml`; when omitted it defaults to
`labor_code_2019` for backwards compatibility.

## Optional fields

| Field | Type | Notes |
|-------|------|-------|
| `split` | string | `train` / `dev` / `test` (default `test`) |
| `question_type` | string | free-form category used for error analysis |
| `mentions_article_number` | bool | whether the question names an article number; derived from the text when omitted |
| `reference_answer` | string | gold answer for generation metrics |
| `notes` | string | annotation notes |

## Example line (schema illustration, not benchmark data)

```json
{"id": "q_example_001", "question": "Người lao động làm việc đủ 12 tháng được nghỉ hằng năm bao nhiêu ngày trong điều kiện bình thường?", "relevant": [{"document_id": "labor_code_2019", "article": "113", "clause": "1"}], "split": "test", "question_type": "factoid", "mentions_article_number": false}
```

## Checks before a retrieval run

1. `legal-rag evaluate coverage <qa.jsonl> --chunks <run>/all_legal_aware.jsonl`
   must report `ref_coverage = 1.0`. A missing reference means the chunk corpus
   is incomplete for that question (often an OCR page failure).
2. Questions that mention an article number and questions that do not must be
   evaluated separately: metadata text can make the retriever match the number
   instead of the legal content. `evaluate retrieval` reports both groups under
   `metrics.by_mentions_article_number`.
3. The same `id` must never appear twice, and the same question must not appear
   in more than one split.
