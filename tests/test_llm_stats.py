"""llm_stats counts calls, failures and busy time per branch."""

import threading

import pytest

from pipeline_modules.business import llm_stats


@pytest.fixture(autouse=True)
def clean_stats():
    llm_stats.reset()
    yield
    llm_stats.reset()


def test_successful_calls_are_counted_per_branch():
    with llm_stats.record(llm_stats.ZONE_CHECK):
        pass
    with llm_stats.record(llm_stats.ZONE_CHECK):
        pass
    with llm_stats.record(llm_stats.NOT_ALLOWED_RERANK):
        pass

    taken = llm_stats.snapshot()
    assert taken[llm_stats.ZONE_CHECK].calls == 2
    assert taken[llm_stats.NOT_ALLOWED_RERANK].calls == 1
    assert taken[llm_stats.ZONE_CHECK].failures == 0


def test_failed_call_is_counted_and_reraised():
    with pytest.raises(RuntimeError):
        with llm_stats.record(llm_stats.ZONE_CHECK):
            raise RuntimeError("backend is down")

    stats = llm_stats.snapshot()[llm_stats.ZONE_CHECK]
    assert (stats.calls, stats.failures) == (1, 1)


def test_reset_clears_previous_run():
    with llm_stats.record(llm_stats.ZONE_CHECK):
        pass
    llm_stats.reset()

    assert llm_stats.snapshot() == {}
    assert llm_stats.format_summary() == "calls=0"


def test_counts_survive_parallel_workers():
    def work():
        for _ in range(50):
            with llm_stats.record(llm_stats.ZONE_CHECK_DEEP):
                pass

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert llm_stats.snapshot()[llm_stats.ZONE_CHECK_DEEP].calls == 200


def test_summary_names_every_branch_used():
    with llm_stats.record(llm_stats.ZONE_CHECK):
        pass
    with llm_stats.record(llm_stats.NOT_ALLOWED_RERANK):
        pass

    summary = llm_stats.format_summary()
    assert llm_stats.ZONE_CHECK in summary
    assert llm_stats.NOT_ALLOWED_RERANK in summary
    assert "total: calls=2" in summary
