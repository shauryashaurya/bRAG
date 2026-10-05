# Claims Pipeline Walkthrough

How to run the system, connect the MCP server to Claude Code, and see how the parts coordinate.

## 0. An honest note on "agents"

Today there are no separate autonomous agents. There is one **pipeline of five stages**, run in a fixed order by plain Python, and one (mocked) model call inside the extraction stage. The model proposes; ordinary code validates, routes and records. That split is the main pattern to learn from this project.

The real coordination happens in two places:

| Layer | Who coordinates | What it does |
|---|---|---|
| Pipeline | `process_claim()` in `main.py` | Calls each stage in order and passes typed records between them. |
| MCP | Claude Code (the client) | Decides which tool to call, in what order, and reads the results. |

If you later want true multi-agent behavior, the natural split is one agent per stage (extractor, assessor, reviewer-assistant) with the router staying deterministic. See section 6.

## 1. Setup (once)

```powershell
cd C:\shaurya\lab\cca-p\claims-proc
pip install -r requirements.txt
```

Python 3.11 was used. The pipeline itself needs no packages; only the MCP server needs `mcp`.

## 2. Run the pipeline directly

```powershell
python main.py
```

Expected output:

```
claim_1.txt -> auto_approve (high confidence and amount under $10,000)
claim_2.txt -> review (needs human review (moderate confidence or amount >= $10,000))
claim_3.txt -> escalate (extraction failed schema validation after 3 attempts)
```

## 3. Watch the stages hand off to each other

This snippet wraps each stage so you can see the order and the type of record that moves between them. Run it from the project folder:

```powershell
python -c "
import main

def trace(mod, name, label):
    fn = getattr(mod, name)
    def wrapper(*a, **k):
        out = fn(*a, **k)
        print(f'[{label}] {name} -> {type(out).__name__}')
        return out
    setattr(mod, name, wrapper)

trace(main, 'tokenize_pii', '1 PII boundary')
trace(main, 'extract_claim', '2 extraction')
trace(main, 'assess_claim', '3 assessment')
trace(main, 'route_claim', '4 routing')
trace(main, 'log_decision', '5 audit log')
print(main.process_claim(open('test_data/claim_2.txt').read()).action)
"
```

Output:

```
[1 PII boundary] tokenize_pii -> str
[2 extraction] extract_claim -> dict
[3 assessment] assess_claim -> Assessment
[4 routing] route_claim -> Decision
[5 audit log] log_decision -> NoneType
review
```

Reading it: text goes in, a safe string comes out, a schema-valid dict is extracted, an Assessment (confidence plus reasoning) is built, a Decision is made, and the Decision is written to the database.

### Show the failure and retry paths

```python
from main import process_claim
from common.fake_client import FakeClient

doc = open("test_data/claim_1.txt").read()

c = FakeClient(fail_first_n=2)       # model output is invalid twice
print(process_claim(doc, c).action, "calls:", c.calls)   # auto_approve, 3 calls

c = FakeClient(always_invalid=True)  # model never produces valid output
print(process_claim(doc, c).action, "calls:", c.calls)   # escalate, 3 calls
```

This is the validation-retry loop from CLAUDE.md: at most 3 attempts, then a human gets the claim.

## 4. Add the MCP server to Claude Code

The project already contains `.mcp.json`, so there are two ways to connect.

**Option A: project file (already set up).** Start Claude Code in the project folder:

```powershell
cd C:\shaurya\lab\cca-p\claims-proc
claude
```

Claude Code asks you to approve the project server `claims-pipeline` the first time. Approve it, then type `/mcp` and confirm it shows as connected.

**Option B: register it with the CLI.**

```powershell
claude mcp add claims-pipeline --env CLAIMS_DB=claims.db --env CLAIMS_DIR=test_data -- python claims_mcp_server.py
claude mcp list
```

Run this from the project folder so the relative paths resolve. Restart the session afterwards.

**If it does not connect:** run `python claims_mcp_server.py` by hand. It should sit silently waiting for input (press Ctrl+C to stop). An import error here is the usual cause, so check that `pip install -r requirements.txt` finished.

Settings:

| Variable | Default | Meaning |
|---|---|---|
| `CLAIMS_DB` | `claims.db` | SQLite file where decisions persist across runs. |
| `CLAIMS_DIR` | `test_data` | Folder of `.txt` claims the server may process. |

## 4b. Using the Claude Code extension in VS Code

The extension and the terminal CLI share the same MCP configuration, so `.mcp.json` works in both. Open the `claims-proc` folder itself as the VS Code workspace (File > Open Folder), so the relative paths in `.mcp.json` resolve.

**Slash commands, typed in the extension's chat box:**

| Command | What it does |
|---|---|
| `/mcp` | Opens the MCP server manager: shows each server, its status, and its tools. Use it to confirm `claims-pipeline` is connected, and to reconnect or enable/disable a server. |
| `/help` | Lists all available slash commands in your version. |
| `/clear` | Starts a fresh conversation (tool results from earlier calls leave the context). |

The exact look of `/mcp` varies by extension version. If a command is missing, check `/help`.

**Adding a server from the integrated terminal** (Terminal > New Terminal, with the workspace as the current folder). The `claude mcp` commands are CLI commands, not chat slash commands:

```powershell
claude mcp add claims-pipeline --env CLAIMS_DB=claims.db --env CLAIMS_DIR=test_data -- python claims_mcp_server.py
claude mcp list
claude mcp get claims-pipeline
claude mcp remove claims-pipeline
```

The `--` separates Claude's own flags from the command that launches your server. Add `--scope project` to write into a shared `.mcp.json`, `--scope user` to make it available in every project, or leave the default (`local`, private to you in this project). You do not need to run `claude mcp add` at all if you keep the existing `.mcp.json`; the two routes do the same job.

**First-time flow in VS Code:**
1. Open the folder, then open the Claude Code panel.
2. Approve the project server when prompted (project servers need your consent because they run a command on your machine).
3. Type `/mcp` and check that `claims-pipeline` shows as connected with 5 tools.
4. If you changed `.mcp.json` or edited the server code, use `/mcp` to reconnect, or start a new session.

**Troubleshooting on Windows:**
- The server is started with plain `python`, so the Python found first on your PATH must have the `mcp` package installed. Check with `python -c "import mcp"` in the same terminal.
- If you use a virtual environment, either activate it before launching VS Code or put its full interpreter path in the `command` field of `.mcp.json`.
- Server output on stdout is the protocol channel. If you add `print()` calls to the server, it will break; log to stderr instead.

### What is npx, and does it matter here?

`npx` is a tool that ships with Node.js. It downloads and runs a package from the npm registry in one step, for example `npx -y @modelcontextprotocol/server-filesystem C:\some\folder`. Many published MCP servers (filesystem, GitHub, browser tools) are Node packages, and their install instructions use `npx` as the launch command, which is why you see it in many MCP examples:

```json
{ "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\data"] }
```

It is **not relevant to this project**. Our server is a Python file launched with `python claims_mcp_server.py`, so you do not need Node or npx installed. You would only need it if you add a third-party Node-based MCP server next to ours. Two notes if you do: on Windows such entries often need `"command": "cmd", "args": ["/c", "npx", ...]`, and only add third-party servers you trust, since they run code with your permissions. The Python counterpart to npx is `uvx` (from the `uv` tool), which you could use if you publish this server as a package; it is optional.

## 5. Drive it from Claude Code and watch the coordination

Inside the Claude Code session, try these prompts in order. Tool names appear as `mcp__claims-pipeline__<tool>`.

1. `What claim files are available?` -> Claude calls `list_claim_files`.
2. `Process all of them and summarize.` -> Claude calls `process_claim_file` once per file. This is the model coordinating: it chooses the order, uses the file names from step 1, and combines the results.
3. `Which claims need a human, and why?` -> Claude calls `list_decisions` (it may filter by `review` and `escalate`).
4. `Show the full record for the $48,500 claim.` -> Claude takes the `claim_ref` from earlier results and calls `get_claim_record`.
5. `What does the rubric score on?` -> `get_rubric`.

Things to point out when demonstrating:

- **Tool chaining.** Each call depends on the previous result (file names, then refs). Claude Code shows every tool call and result in the transcript.
- **No raw PII crosses the boundary.** No tool accepts or returns claim text. Ask Claude "what is the claimant's email?" and it cannot answer from these tools.
- **The model cannot approve.** There is no tool to approve or override a review item, so a prompt like "approve claim_2" has nothing to call. Humans stay in the loop.
- **Server-side guards.** Try `process_claim_file` with `../CLAUDE.md`. The server rejects it and tells the model to use `list_claim_files`.

Inspect what was recorded:

```powershell
python -c "import sqlite3; c=sqlite3.connect('claims.db'); [print(r) for r in c.execute('select claim_ref, action, reason from decisions')]"
```

## 6. Where this goes next (real multi-agent)

To turn the stages into cooperating agents while keeping the safety rules:

- **Extractor agent:** the only component that sees tokenized text; its output must pass the JSON schema.
- **Assessor agent:** scores the rubric as a judge, returning scores and written reasoning.
- **Router:** stays plain code. Money and compliance rules should not be left to a model.
- **Orchestrator:** `process_claim()` already acts as one; it would call each agent, enforce the retry limit, and log every handoff.
- **Review assistant:** a human-facing agent that reads `list_decisions` and drafts summaries for reviewers, with no power to approve.

Keep these invariants whatever you build: tokenize PII before any model call, validate every model output against a schema, never auto-approve $10,000 or more, and log each decision with its reasoning.

## 7. File map

| File | Role |
|---|---|
| `main.py` | `process_claim()`: wires the stages together. |
| `extraction/pii_boundary.py` | Tokenizes PII before any model call. |
| `extraction/extractor.py` | Schema, forced tool call, validate-retry loop. |
| `assessment/evaluator.py` | Rubric and mock assessor. |
| `routing/router.py` | Deterministic routing thresholds. |
| `db/store.py` | SQLite tables for claims, decisions, audit log. |
| `common/` | Shared models, mocked helpers, fake model client. |
| `claims_mcp_server.py` | MCP server exposing the pipeline as tools. |
| `.mcp.json` | Tells Claude Code how to launch the server. |
| `test_data/` | Three sample claims (low value, high value, missing date). |
