"""Tests for the golden-set PHI eval harness."""

from __future__ import annotations

import json
import sys

from auscult import cli
from auscult.eval import evaluate, format_report, load_corpus
from auscult.sanitizer import Sanitizer


def test_load_default_corpus() -> None:
    corpus = load_corpus()
    assert len(corpus) >= 10
    assert any(ex.entities for ex in corpus)
    assert any(not ex.entities for ex in corpus)
    for ex in corpus:
        for span in ex.entities:
            assert ex.text[span.start : span.end]


def test_evaluate_reports_precision_recall() -> None:
    report = evaluate(sanitizer=Sanitizer(nlp_model="en_core_web_sm"))
    assert report.true_positives >= 1
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert report.f1 >= 0.0
    # Negatives (eponyms / labs) should keep precision from collapsing.
    assert report.precision >= 0.5
    assert report.recall >= 0.5


def test_format_report_includes_metrics() -> None:
    report = evaluate(sanitizer=Sanitizer(nlp_model="en_core_web_sm"))
    text = format_report(report, verbose=True)
    assert "precision:" in text
    assert "recall:" in text
    assert "f1:" in text


def test_cmd_eval_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["auscult", "--json", "eval", "--nlp-model", "en_core_web_sm"],
    )
    try:
        cli.main()
    except SystemExit as exc:
        # Low-recall guard may exit 1; still expect JSON on stdout.
        assert exc.code in (0, 1)
    payload = json.loads(capsys.readouterr().out)
    assert "precision" in payload
    assert "recall" in payload
    assert payload["examples"] >= 10
