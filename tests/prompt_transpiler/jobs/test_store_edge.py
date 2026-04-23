import sqlite3
from unittest.mock import patch

import pytest

from prompt_transpiler.jobs.store import SQLiteJobStore


def test_sqlite_store_duplicate_job_id_raises_integrity_error(tmp_path):
    """
    Test that SQLiteJobStore raises sqlite3.IntegrityError when generate_job_id
    returns a duplicate ID. This verifies the edge case of UUID collision and
    ensures the database properly enforces the PRIMARY KEY constraint on job_id.
    """
    db_path = tmp_path / "jobs.sqlite"
    store = SQLiteJobStore(str(db_path))

    with patch("prompt_transpiler.jobs.store.generate_job_id", return_value="duplicate-id"):
        # First insertion should succeed
        store.create_job({"raw_prompt": "first"})

        # Second insertion with the same UUID should raise an IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            store.create_job({"raw_prompt": "second"})
