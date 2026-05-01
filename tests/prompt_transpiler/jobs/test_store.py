import pytest

from prompt_transpiler.jobs.models import JobStatus
from prompt_transpiler.jobs.store import MemoryJobStore, SQLiteJobStore


def _assert_basic_flow(store):
    job_id = store.create_job({"raw_prompt": "hello"})
    created = store.get_job(job_id)
    assert created is not None
    assert created["status"] == JobStatus.QUEUED.value
    assert created["request"] == {"raw_prompt": "hello"}

    store.update_job(job_id, progress={"step": "queued"})
    updated = store.get_job(job_id)
    assert updated is not None
    assert updated["progress"] == {"step": "queued"}

    claimed = store.claim_next_job("worker-1")
    assert claimed is not None
    assert claimed["job_id"] == job_id
    assert claimed["status"] == JobStatus.RUNNING.value
    assert claimed["worker_id"] == "worker-1"

    store.complete_job(job_id, {"ok": True})
    completed = store.get_job(job_id)
    assert completed is not None
    assert completed["status"] == JobStatus.SUCCEEDED.value
    assert completed["result"] == {"ok": True}


def _assert_purge(store):
    job_id = store.create_job({"raw_prompt": "hello"})
    store.update_job(
        job_id,
        status=JobStatus.SUCCEEDED.value,
        completed_at="2000-01-01T00:00:00+00:00",
    )
    purged = store.purge_expired("2010-01-01T00:00:00+00:00")
    assert purged == 1
    assert store.get_job(job_id) is None


def _assert_cancel_requested(store):
    job_id = store.create_job({"raw_prompt": "hello"})
    store.update_job(job_id, cancel_requested=True)
    updated = store.get_job(job_id)
    assert updated is not None
    assert updated["cancel_requested"] is True


def test_update_job_invalid_identifier():
    store = MemoryJobStore()
    job_id = store.create_job({"raw_prompt": "hello"})
    with pytest.raises(ValueError, match="Invalid column name: invalid key"):
        store.update_job(job_id, **{"invalid key": "value"})


def test_memory_store_flow():
    store = MemoryJobStore()
    _assert_basic_flow(store)
    _assert_purge(store)
    _assert_cancel_requested(store)


def test_sqlite_store_flow(tmp_path):
    db_path = tmp_path / "jobs.sqlite"
    store = SQLiteJobStore(str(db_path))
    _assert_basic_flow(store)
    _assert_purge(store)
    _assert_cancel_requested(store)


def test_duckdb_store_flow(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    assert duckdb is not None
    from prompt_transpiler.jobs.store import DuckDBJobStore  # noqa: PLC0415

    db_path = tmp_path / "jobs.duckdb"
    store = DuckDBJobStore(str(db_path))
    _assert_basic_flow(store)
    _assert_purge(store)
    _assert_cancel_requested(store)
