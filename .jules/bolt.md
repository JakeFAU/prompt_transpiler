## 2026-05-03 - Optimize Token Usage Delta Calculations
**Learning:** In hot loops, calculating deltas by creating an intermediate dictionary and checking `any(dict.values())` incurs unnecessary object allocation and iteration overhead. Using a module-level constant for `dict.get()` defaults and performing explicit boolean checks (e.g. `p_delta > 0 or c_delta > 0`) is significantly faster.
**Action:** Replace dictionary-based delta checks with explicit boolean logic and module-level constants in performance-critical sections to reduce allocation and processing time.
