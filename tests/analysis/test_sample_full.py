from pathlib import Path

from inspect_ai.analysis._dataframe.samples.columns import (
    SampleColumn,
    SampleSummary,
)
from inspect_ai.analysis._dataframe.samples.table import samples_df

LOGS_DIR = Path(__file__).parent / "test_logs"

POPULARITY_LOG = LOGS_DIR / "2025-05-12T20-28-13-04-00_popularity.json"
READ_SAMPLE_EVAL_LOG = (
    Path(__file__).parents[1] / "log" / "test_eval_log" / "log_read_sample.eval"
)


def test_sample_not_full():
    df = samples_df(POPULARITY_LOG)
    assert "metadata_label_confidence" in df.columns
    assert "metadata_nested" in df.columns


def test_sample_metadata_full():
    df = samples_df(
        POPULARITY_LOG,
        columns=SampleSummary
        + [SampleColumn("metadata_*", path="metadata", full=True)],
    )
    assert "metadata_label_confidence" in df.columns
    assert "metadata_nested" in df.columns


def test_sample_param_full():
    df = samples_df(POPULARITY_LOG, columns=SampleSummary, full=True)
    assert "metadata_label_confidence" in df.columns
    assert "metadata_nested" in df.columns


def test_sample_param_full_exclude_fields():
    columns = [
        SampleColumn("id", path="id", required=True, type=str),
        SampleColumn(
            "event_count",
            path=lambda sample: len(sample.events),
            required=True,
            type=int,
            full=True,
        ),
    ]

    df = samples_df(READ_SAMPLE_EVAL_LOG, columns=columns, full=True)
    df_excluded = samples_df(
        READ_SAMPLE_EVAL_LOG,
        columns=columns,
        full=True,
        exclude_fields={"events"},
    )

    assert int(df.iloc[0]["event_count"]) > 0
    assert int(df_excluded.iloc[0]["event_count"]) == 0
