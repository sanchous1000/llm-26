"""Ограничивает время запуска отдельным процессом и останавливает его при таймауте."""

from multiprocessing import get_context
from time import monotonic


def run_with_timeout(target, timeout, args=()):
    process = get_context("spawn").Process(target=target, args=args)
    started = monotonic()
    try:
        process.start()
        process.join(max(0, timeout - (monotonic() - started)))
        if process.is_alive():
            raise TimeoutError(f"Истёк общий таймаут запуска: {timeout} с.")
        return process.exitcode
    finally:
        # Не оставляем модель или инструмент выполняться в фоне.
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()
