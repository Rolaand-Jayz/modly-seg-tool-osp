# Process Runs Router Integration

The process-run router is registered by `api/main.py`. It accepts the canonical start body
`{"process":"<declared-extension-id>","input":{...}}` and exposes start, status, and cancel routes:

- `POST /process-runs`
- `GET /process-runs/{run_id}`
- `POST /process-runs/{run_id}/cancel`

These are local automation routes. Browser requests carrying an `Origin` header are rejected; the stdlib Modly CLI sends requests without that header. Run status records the measured extension digest and labels installed code `local-unpinned` unless a separate host-owned reference workflow verifies it.
