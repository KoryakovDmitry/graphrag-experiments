import logging
from pathlib import Path

from graphrag.index.progress import (
    NullProgressReporter,
    PrintProgressReporter,
    ProgressReporter,
)
from graphrag.index.progress.rich import RichProgressReporter


def _get_progress_reporter(reporter_type: str | None) -> ProgressReporter:
    if reporter_type is None or reporter_type == "rich":
        return RichProgressReporter("GraphRAG Indexer ")
    if reporter_type == "print":
        return PrintProgressReporter("GraphRAG Indexer ")
    if reporter_type == "none":
        return NullProgressReporter()

    msg = f"Invalid progress reporter type: {reporter_type}"
    raise ValueError(msg)


def _enable_logging(root_dir: str, run_id: str, verbose: bool) -> None:
    # logging_file = (
    #     Path(root_dir) / "output" / run_id / "reports" / "indexing-engine.log"
    # )
    # logging_file.parent.mkdir(parents=True, exist_ok=True)
    #
    # logging_file.touch(exist_ok=True)

    logging.basicConfig(
        # filename=str(logging_file),
        # filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
