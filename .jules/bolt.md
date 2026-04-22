## 2026-04-22 - Extract inline frozenset
**Learning:** In Python, inline set creations like `in {val1, val2}` where `val1` is an enum attribute (e.g. `JobStatus.SUCCEEDED.value`) are not compiled into a constant `frozenset` by the interpreter due to the attribute lookup. This forces O(N) set allocations in hot loops or list comprehensions.
**Action:** Always extract inline sets containing attribute lookups to module-level `frozenset` constants to ensure O(1) performance and avoid re-allocation in loops.
