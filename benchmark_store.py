import time

from prompt_transpiler.jobs.store import MemoryJobStore


def run_benchmark() -> None:
    store = MemoryJobStore()

    print("Creating jobs...")
    for i in range(100000):
        store.create_job({"test": i})

    print("Claiming jobs...")
    start = time.time()
    for _ in range(100):
        store.claim_next_job("worker_1")
    end = time.time()

    print(f"Time to claim 100 jobs out of 100000: {end - start:.4f} seconds")


if __name__ == "__main__":
    run_benchmark()
