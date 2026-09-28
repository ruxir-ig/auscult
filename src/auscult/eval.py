"""Golden-set PHI detection evaluation (precision / recall).

The checked-in corpus labels true PHI spans vs clinical negatives
(eponyms, drugs, labs). Use ``auscult eval`` to measure whether model /
threshold changes actually improve detection quality.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .sanitizer import Sanitizer

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "phi_eval_corpus.jsonl"


@dataclass(frozen=True)
class LabeledSpan:
    entity_type: str
    start: int
    end: int

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.entity_type, self.start, self.end)


@dataclass(frozen=True)
class EvalExample:
    id: str
    text: str
    entities: tuple[LabeledSpan, ...]


@dataclass
class ExampleScore:
    example_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted: list[LabeledSpan]
    expected: list[LabeledSpan]


@dataclass
class EvalReport:
    examples: list[ExampleScore]
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": len(self.examples),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "per_example": [
                {
                    "id": ex.example_id,
                    "true_positives": ex.true_positives,
                    "false_positives": ex.false_positives,
                    "false_negatives": ex.false_negatives,
                    "predicted": [asdict(s) for s in ex.predicted],
                    "expected": [asdict(s) for s in ex.expected],
                }
                for ex in self.examples
            ],
        }


def load_corpus(path: Path | str | None = None) -> list[EvalExample]:
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    examples: list[EvalExample] = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_no} of {corpus_path}"
                ) from exc
            entities = tuple(
                LabeledSpan(
                    entity_type=e["entity_type"],
                    start=int(e["start"]),
                    end=int(e["end"]),
                )
                for e in raw.get("entities", [])
            )
            for span in entities:
                if not (0 <= span.start < span.end <= len(raw["text"])):
                    raise ValueError(
                        f"example {raw.get('id')!r}: span {span} out of bounds "
                        f"for text length {len(raw['text'])}"
                    )
            examples.append(
                EvalExample(id=str(raw["id"]), text=str(raw["text"]), entities=entities)
            )
    return examples


def _spans_overlap(a: LabeledSpan, b: LabeledSpan) -> bool:
    return a.entity_type == b.entity_type and a.start < b.end and b.start < a.end


def _match_spans(
    predicted: Sequence[LabeledSpan],
    expected: Sequence[LabeledSpan],
) -> tuple[int, int, int]:
    """Greedy one-to-one matching by overlapping same-type spans."""
    used_pred: set[int] = set()
    used_exp: set[int] = set()
    tp = 0
    for ei, exp in enumerate(expected):
        for pi, pred in enumerate(predicted):
            if pi in used_pred:
                continue
            if _spans_overlap(pred, exp):
                used_pred.add(pi)
                used_exp.add(ei)
                tp += 1
                break
    fp = len(predicted) - len(used_pred)
    fn = len(expected) - len(used_exp)
    return tp, fp, fn


def detect_spans(sanitizer: Sanitizer, text: str) -> list[LabeledSpan]:
    """Run the sanitizer analyzer and return labeled spans (no replacement)."""
    results = sanitizer.analyze(text)
    return [
        LabeledSpan(entity_type=r.entity_type, start=r.start, end=r.end)
        for r in results
    ]


def evaluate(
    examples: Iterable[EvalExample] | None = None,
    *,
    sanitizer: Sanitizer | None = None,
    corpus_path: Path | str | None = None,
) -> EvalReport:
    corpus = list(examples) if examples is not None else load_corpus(corpus_path)
    engine = sanitizer or Sanitizer()
    scores: list[ExampleScore] = []
    tp = fp = fn = 0
    for example in corpus:
        predicted = detect_spans(engine, example.text)
        expected = list(example.entities)
        etp, efp, efn = _match_spans(predicted, expected)
        scores.append(
            ExampleScore(
                example_id=example.id,
                true_positives=etp,
                false_positives=efp,
                false_negatives=efn,
                predicted=predicted,
                expected=expected,
            )
        )
        tp += etp
        fp += efp
        fn += efn
    return EvalReport(
        examples=scores,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def format_report(report: EvalReport, *, verbose: bool = False) -> str:
    lines = [
        "PHI detection eval",
        f"  examples:   {len(report.examples)}",
        f"  precision:  {report.precision:.4f}",
        f"  recall:     {report.recall:.4f}",
        f"  f1:         {report.f1:.4f}",
        f"  tp/fp/fn:   {report.true_positives}/{report.false_positives}/{report.false_negatives}",
    ]
    if verbose:
        lines.append("")
        for ex in report.examples:
            if ex.false_positives or ex.false_negatives:
                lines.append(
                    f"  ! {ex.example_id}: tp={ex.true_positives} "
                    f"fp={ex.false_positives} fn={ex.false_negatives}"
                )
    return "\n".join(lines) + "\n"
