# Using MCM from a coding agent

MCM is a research prototype, and this page is written on that basis. It tells you
how to connect it to an agent, what each tool is for, and where the thing is
likely to disappoint you. If you want the evidence for whether it helps, read
[evaluation.md](evaluation.md) first; the honest summary is that MCM retrieves
better than the baselines it was tested against and that its context-efficiency
lead over vector RAG does not survive tuning the baselines on equal terms.

## What connects to what

Coding agents reach outside tools through the **Model Context Protocol**. MCM
ships an MCP server that speaks stdio, which every client below supports:

```
your agent  ──JSON-RPC over stdio──>  mcm-mcp  ──>  mcm.db  (SQLite)
                                                       ^
                                         mcm ingest ────┘
```

The agent spawns `mcm-mcp` as a child process. The server reads a SQLite index
built by `mcm ingest`. Nothing listens on a port, nothing leaves the machine, and
no API key is involved.

There is also a REST API (`mcm.api`) for calling MCM over a network from a
service. It is a different surface for a different caller, and it is not what an
agent uses. Its routes are documented in the README.

## Install

```bash
git clone https://github.com/abusuraihsakhri/mcm.git
cd mcm
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e .
```

Python 3.12 or newer. `pip install -e .` puts two commands on your PATH: `mcm`
and `mcm-mcp`.

## Build an index

Every tool reads an index. Build one per repository:

```bash
mcm ingest /path/to/your/repo --db /path/to/mcm.db
```

On a 94-file repository this takes about 2.4 seconds. Roughly 70% of that is AST
normalisation for the equivalence engine, so it scales with the amount of code
rather than with repository history.

The index goes stale as you edit. Re-run `mcm ingest` to refresh it, or call the
`mcm_ingest` tool from inside the agent session. Re-ingesting is not destructive:
relations that disappeared are closed with an end date rather than deleted, so a
query against an earlier moment still answers.

Put the database outside the repository, or add it to `.gitignore`. It is derived
data and it is not small.

## Connect your agent

Every configuration below runs the same command. Use **absolute paths** for both
the executable and the database. Agents do not reliably inherit your shell's PATH
or working directory, and a relative path is the most common reason a server
shows up as "failed to start" with no further explanation.

To find the absolute path to the executable:

```bash
# macOS / Linux
which mcm-mcp
# Windows PowerShell
(Get-Command mcm-mcp).Source
```

### Claude Code

Project scope, checked into version control, in `.mcp.json` at the repository
root:

```json
{
  "mcpServers": {
    "mcm": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/mcm-mcp",
      "args": ["--db", "/absolute/path/to/mcm.db"],
      "env": {}
    }
  }
}
```

Or from the CLI, where everything after `--` is passed to the server untouched:

```bash
claude mcp add --scope project --transport stdio mcm \
  -- /absolute/path/to/.venv/bin/mcm-mcp --db /absolute/path/to/mcm.db
```

Use `--scope user` instead to make it available in every project. Verify with
`/mcp` inside a session; the five tools should be listed.

### Cursor

Project scope in `.cursor/mcp.json`, or global in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "mcm": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/mcm-mcp",
      "args": ["--db", "/absolute/path/to/mcm.db"]
    }
  }
}
```

Cursor supports `${workspaceFolder}` and `${userHome}` interpolation, which is
worth using if the index lives beside the code:

```json
"args": ["--db", "${workspaceFolder}/.mcm/index.db"]
```

### Codex CLI

Codex uses TOML, and the table is `mcp_servers` in snake_case. Global config is
`~/.codex/config.toml`; project config is `.codex/config.toml` and applies to
trusted projects only:

```toml
[mcp_servers.mcm]
command = "/absolute/path/to/.venv/bin/mcm-mcp"
args = ["--db", "/absolute/path/to/mcm.db"]
startup_timeout_sec = 20
```

Or from the CLI:

```bash
codex mcp add mcm -- /absolute/path/to/.venv/bin/mcm-mcp --db /absolute/path/to/mcm.db
```

Check it registered with `codex mcp list`.

### OpenCode

OpenCode differs from the others in two ways that will bite you if you copy a
Claude Code config: the top-level key is `mcp`, not `mcpServers`, and `command`
is a **single array** holding the executable and its arguments rather than a
string plus a separate `args`. Project scope is `opencode.json` at the repository
root; global is `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mcm": {
      "type": "local",
      "command": ["/absolute/path/to/.venv/bin/mcm-mcp", "--db", "/absolute/path/to/mcm.db"],
      "enabled": true,
      "environment": {}
    }
  }
}
```

### Antigravity

Antigravity shares one MCP configuration across the IDE and the CLI. Global is
`~/.gemini/config/mcp_config.json`; workspace scope is `.agents/mcp_config.json`:

```json
{
  "mcpServers": {
    "mcm": {
      "command": "/absolute/path/to/.venv/bin/mcm-mcp",
      "args": ["--db", "/absolute/path/to/mcm.db"],
      "env": {}
    }
  }
}
```

In the IDE you can reach the same file through the agent side panel: `…` →
**MCP Servers** → **Manage MCP Servers** → **View raw config**.

### Anything else

Any MCP client that supports stdio works. The command is `mcm-mcp --db <path>`,
or equivalently `python -m mcm.mcp_server --db <path>` if you would rather not
rely on the console script. The database path can also come from the `MCM_DB`
environment variable, which is useful where a client makes it awkward to pass
arguments.

## The tools

| Tool | Answers | Reach for it when |
| --- | --- | --- |
| `mcm_search` | Which definitions relate to this description? | Starting in unfamiliar code, instead of guessing filenames |
| `mcm_impact` | What would this change break? | Before editing a signature, renaming, or deleting |
| `mcm_context` | What is the minimum I need to know? | At the start of a task, instead of reading whole files |
| `mcm_constraints` | Did this violate an architectural rule? | After a change, as a check |
| `mcm_ingest` | Index or refresh a repository | Once per repository, then whenever the index is stale |

`mcm_ingest` builds the retrieval projections as part of the same call, so search
works immediately afterwards without a separate step.

### The one that is actually different

`mcm_search` and `mcm_context` are better versions of things an agent can already
do badly. `mcm_impact` is the one with no equivalent: it answers what a change
would break *before* the change is written, by walking the call graph transitively
and reporting a confidence per hop.

```
mcm_impact(target="Signer.sign", change_kind="SIGNATURE")

  must_update:
    Serializer.dumps            hop 1   c=0.95
    TimestampSigner.unsign      hop 2   c=0.76
  tests_at_risk:
    test_roundtrip              covers Serializer.dumps
    test_tampered               covers Signer.unsign
```

Confidence decays with distance, because a two-hop inference is a weaker claim
than a one-hop one. Treat a low-confidence hop as a lead to verify, not a fact.

### Telling the agent to use them

Tool descriptions alone are usually not enough to change an agent's habits. It
will still grep, because grepping is what it knows. Put the policy in the
project's instruction file (`CLAUDE.md`, `.cursorrules`, `AGENTS.md`, depending
on the client):

```markdown
## Code navigation

This repository has an MCM index. Prefer it over grep for semantic questions.

- Before editing any function that other code may call, run `mcm_impact` on it
  and read the result. Do not skip this for renames.
- To find code by what it does rather than by name, use `mcm_search`.
- If a tool reports a stale or missing index, run `mcm_ingest` and retry.
```

## When it goes wrong

**The server does not start.** Almost always a path problem. Run the command
yourself first — `mcm-mcp --db /path/to/mcm.db` should print one line to stderr
and then wait for input. If that works and the agent still fails, the agent is
resolving a different path; use absolute ones everywhere.

**Every tool says there is no index.** You have not run `mcm ingest`, or the
`--db` path in your config does not match the one you ingested into.

**Search returns nothing useful.** The default embedding provider is feature
hashing, not a trained model. It matches on token overlap, so it does well on
"validate token" and poorly on "make sure the user is who they say they are".
Install the optional extra for real embeddings:

```bash
pip install -e ".[embeddings]"
```

**Impact analysis misses a caller.** MCM resolves calls statically, and Python
resolves plenty of them at runtime. Dynamic dispatch, `getattr`, and anything
reached through a registry will not appear. There is a runtime tracing channel
(`mcm.reasoning.observation`) that feeds real execution back into the graph, but
it has to be run; a static index alone will under-report.

**The index disagrees with the code.** It is a snapshot. Re-ingest.

## Limits worth knowing before you rely on it

- **Python only.** The parser and every extractor are Python-specific.
- **Static analysis only, by default.** See the dynamic-dispatch note above.
- **Single machine, single user.** The index is a local SQLite file with no
  concurrency story. Two agents writing to one database is untested.
- **Unproven in the loop.** The benchmark measures what is put in front of an
  agent, not what the agent then does with it. Whether any of this improves task
  completion is unmeasured. That is stated as limitation 8 in
  [evaluation.md](evaluation.md) and it is the honest state of the work.
