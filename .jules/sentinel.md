## 2026-05-03 - Dynamic SQL Construction Risk
**Vulnerability:** SQL injection vector via dynamically constructed column names in UPDATE queries.
**Learning:** Standard parameterized execution cannot be used for SQL identifiers like column names. This creates a risk if arbitrary keys are passed to update methods.
**Prevention:** Use `<string>.isidentifier()` to validate that any dynamically injected table or column names are safe, strictly alphanumeric plus underscore, preventing SQL injection before adding `# nosec B608` to bypass linter warnings.
