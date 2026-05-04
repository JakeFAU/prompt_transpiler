## 2026-05-04 - Dynamic SQL Column Vulnerability
**Vulnerability:** SQL injection vulnerability through dynamic string-based column creation (`_to_update_clause`) in `UPDATE` queries.
**Learning:** Because column names cannot be parameterized in SQL, dynamically building them from user-provided dictionary keys creates a severe SQL injection vector (`B608`).
**Prevention:** Always enforce a strict allowlist (`frozenset`) for dynamically updatable fields, and use `.isidentifier()` to validate that the column name is a safe SQL identifier before string formatting.
