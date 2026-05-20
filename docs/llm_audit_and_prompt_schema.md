# LLM Audit and Prompt Schema

The artifact does not release raw LLM responses. It releases the prompt schema,
candidate schema, and redaction policy needed to audit the proposal pipeline.

Required metadata:

```json
{
  "prompt_schema_version": "v1",
  "llm_provider": "OpenAI",
  "model_name": "gpt-5.2",
  "temperature": 0.7,
  "candidate_count": 5,
  "parser": "json-first-with-validation",
  "malformed_output_policy": "reject-or-repair-then-log",
  "bounds_policy": "clip-to-declared-search-space",
  "duplicate_policy": "deduplicate-before-scoring",
  "redacted_fields": ["api_key", "raw_response", "local_path"]
}
```

Candidate schema:

```json
{
  "schema_version": "v1",
  "candidates": [
    {
      "params": {},
      "mu": 0.0,
      "sigma": 0.0,
      "reason": "short rationale"
    }
  ]
}
```
