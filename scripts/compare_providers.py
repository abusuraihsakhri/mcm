"""Compare two evaluation sweeps that differ only in the embedding provider.

``evaluation_suite.py`` writes one JSON per repository. Run it twice into
different directories, once on the default feature hashing and once with
``--provider sentence-transformers:...``, and this reports the difference per
system, paired by repository.

Paired is the point. An earlier attempt compared two django runs at different
horizons, which meant different task sets and a comparison that could only ever
be suggestive. Here the repositories, horizons and task caps are identical and
the provider is the only variable.

**Watch the ``graph`` row.** Baseline B embeds nothing, so its numbers *must* be
identical between the two sweeps. If that row is not exactly zero then something
other than the provider differs, and nothing else in the output can be trusted.
A comparison without a working negative control is a comparison you cannot
distinguish from a bug.

    python scripts/compare_providers.py results-hashing/ results-trained/
"""

from __future__ import annotations

import json
import pathlib
import sys

from mcm.benchmark.metrics import paired_bootstrap

SYSTEMS = ["vector-rag", "graph", "hybrid", "mcm"]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compare_providers.py <sweep-a-dir> <sweep-b-dir>",
              file=sys.stderr)
        return 2
    left_dir, right_dir = pathlib.Path(argv[0]), pathlib.Path(argv[1])

    shared = sorted({p.stem for p in left_dir.glob("*.json")}
                    & {p.stem for p in right_dir.glob("*.json")})
    if not shared:
        print("no repository is scored under both sweeps yet", file=sys.stderr)
        return 1

    print(f"Paired provider comparison, {len(shared)} repositories scored under both")
    print("Same repositories, same horizons, same task caps. Only the embedding differs.\n")
    print(f"  {'repository':<15}{'system':<12}{'A':>9}{'B':>9}{'delta':>9}")

    deltas: dict[str, list[float]] = {s: [] for s in SYSTEMS}
    for name in shared:
        a = json.loads((left_dir / f"{name}.json").read_text())
        b = json.loads((right_dir / f"{name}.json").read_text())
        for system in SYSTEMS:
            x = a["overall"][system]["efficiency"]
            y = b["overall"][system]["efficiency"]
            deltas[system].append(y - x)
            label = name if system == SYSTEMS[0] else ""
            print(f"  {label:<15}{system:<12}{x:>9.3f}{y:>9.3f}{y - x:>+9.3f}")
        print()

    print("Mean change from sweep A to sweep B, per system")
    for system in SYSTEMS:
        d = deltas[system]
        mean = sum(d) / len(d)
        r = paired_bootstrap(d, [0.0] * len(d))
        sep = "separated" if (r.low > 0 or r.high < 0) else "overlapping"
        print(f"  {system:<12}{mean:>+8.4f}  [{r.low:+.4f}, {r.high:+.4f}]  {sep}")

    control = deltas["graph"]
    ok = all(abs(value) < 1e-12 for value in control)
    print(f"\n  Negative control: graph changed by "
          f"{'exactly zero in every repository' if ok else 'a NON-ZERO amount'}.")
    if ok:
        print("  Baseline B embeds nothing, so that is the expected result, and it")
        print("  confirms the provider was the only thing that varied.")
    else:
        print("  Baseline B embeds nothing, so this should be impossible. Something")
        print("  other than the provider differs between these sweeps. Do not report")
        print("  anything above until that is found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
