"""PHI detection and replacement for trace text.

Uses Presidio to find PHI entities and Faker to substitute realistic
synthetic values. A Sanitizer instance keeps a mapping of real -> fake
values so the same entity is replaced consistently within one run.

Configuration (environment variables):
  AUSCULT_SPACY_MODEL   spaCy model name (default: en_core_web_lg).
                        Use en_core_web_trf for best PERSON/LOCATION recall,
                        or en_core_web_sm for faster local/dev loads.
  AUSCULT_SCORE_THRESHOLD
                        Minimum Presidio confidence to redact (default: 0.35).
                        Lower = more false positives redacted (safer for PHI).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache, partial

from faker import Faker
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Entity types we detect and replace in trace text.
ENTITY_TYPES: list[str] = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DATE_TIME",
    "LOCATION",
    "US_SSN",
    "MEDICAL_LICENSE",
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    "STREET_ADDRESS",
    "CUSTOM_IDENTIFIER",
]

# Production default favors recall over load time. Override with
# AUSCULT_SPACY_MODEL=en_core_web_sm for fast local/test runs.
DEFAULT_SPACY_MODEL = "en_core_web_lg"

# Lower threshold = more redaction (safer for PHI, more false positives).
# Presidio's per-recognizer defaults are often ~0.4–0.85; 0.35 biases safe.
DEFAULT_SCORE_THRESHOLD = 0.35

# Disease / clinical terms that look like PERSON surnames. Matching is
# case-insensitive whole-token; these are never redacted as PERSON/LOCATION.
# Prefer uncommon eponyms. Deliberately omit common surnames (Wilson, Down,
# Bell) — those risk false-negative PHI leaks if allowlisted.
DEFAULT_CLINICAL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "parkinson",
        "parkinson's",
        "parkinsons",
        "addison",
        "addison's",
        "addisons",
        "alzheimer",
        "alzheimer's",
        "alzheimers",
        "crohn",
        "crohn's",
        "crohns",
        "huntington",
        "huntington's",
        "huntingtons",
        "gehrig",
        "gehrig's",
        "graves",
        "grave's",
        "cushing",
        "cushing's",
        "cushings",
        "paget",
        "paget's",
        "raynaud",
        "raynaud's",
        "raynauds",
        "meniere",
        "meniere's",
        "menieres",
        "hashimoto",
        "hashimoto's",
        "hashimotos",
        "kaposi",
        "kaposi's",
        "asperger",
        "asperger's",
        "aspergers",
        "marfan",
        "marfan's",
        "marfans",
        "tourette",
        "tourette's",
        "tourettes",
        "legionnaires",
        "legionnaire's",
    }
)

# Organization-specific identifiers Presidio will not know about.
# Patterns are applied as CUSTOM_IDENTIFIER recognizers.
DEFAULT_DENYLIST_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (
        "employee_badge",
        r"\b(?:BADGE|EMP|EID)[\s:#-]*[A-Z0-9]{4,12}\b",
        0.9,
    ),
    (
        "internal_case",
        r"\b(?:CASE|TICKET|INC)[\s:#-]*\d{4,12}\b",
        0.85,
    ),
)

_CUSTOM_RECOGNIZERS = [
    PatternRecognizer(
        supported_entity="MEDICAL_RECORD_NUMBER",
        name="MedicalRecordNumberRecognizer",
        patterns=[
            Pattern(
                name="mrn_abbrev",
                regex=r"\b(?:MRN|MR)[\s:#-]*\d{4,12}\b",
                score=0.9,
            ),
            Pattern(
                name="mrn_spelled_out",
                regex=r"\b[Mm]edical [Rr]ecord (?:[Nn]umber|[Nn]o\.?)[\s:#-]*\d{4,12}\b",
                score=0.9,
            ),
        ],
    ),
    PatternRecognizer(
        supported_entity="PATIENT_ID",
        name="PatientIdRecognizer",
        patterns=[
            Pattern(
                name="patient_id_labeled",
                regex=r"\b(?:[Pp]atient(?:\s+(?:ID|[Ii]d|[Nn]umber|[Nn]o\.?))?|PT|PID)[\s:#-]*\d{4,12}\b",
                score=0.85,
            ),
        ],
    ),
    PatternRecognizer(
        supported_entity="STREET_ADDRESS",
        name="StreetAddressRecognizer",
        patterns=[
            Pattern(
                name="us_street_address",
                regex=(
                    r"\b\d{1,5}\s+(?:[A-Z][A-Za-z]*\.?\s+){1,3}"
                    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|"
                    r"Drive|Dr|Court|Ct|Place|Pl|Way|Terrace|Ter|Circle|Cir)\.?"
                    r"(?:,?\s+(?:Apt|Suite|Ste|Unit)\.?\s*#?\s*\w+)?\b"
                ),
                score=0.75,
            ),
        ],
    ),
    PatternRecognizer(
        supported_entity="DATE_TIME",
        name="DobRecognizer",
        patterns=[
            Pattern(
                name="dob_labeled",
                regex=(
                    r"\b(?:DOB|[Dd]ate of [Bb]irth|[Bb]orn(?:\s+on)?)[\s:]*"
                    r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
                ),
                score=0.9,
            ),
            Pattern(
                name="numeric_date",
                regex=r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
                score=0.6,
            ),
        ],
    ),
]


@dataclass(frozen=True)
class SanitizeResult:
    """Sanitized text plus how many entities were redacted."""

    text: str | None
    redaction_count: int


def _env_spacy_model() -> str:
    return os.environ.get("AUSCULT_SPACY_MODEL", DEFAULT_SPACY_MODEL)


def _env_score_threshold() -> float:
    raw = os.environ.get("AUSCULT_SCORE_THRESHOLD")
    if raw is None:
        return DEFAULT_SCORE_THRESHOLD
    return float(raw)


def _normalize_allowlist(terms: frozenset[str] | set[str] | list[str]) -> frozenset[str]:
    return frozenset(t.strip().lower() for t in terms if t and t.strip())


@lru_cache(maxsize=4)
def _analyzer(model_name: str) -> AnalyzerEngine:
    """Build a Presidio analyzer for the given spaCy model (cached)."""
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
    )
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["en"],
    )
    for recognizer in _CUSTOM_RECOGNIZERS:
        analyzer.registry.add_recognizer(recognizer)
    for name, regex, score in DEFAULT_DENYLIST_PATTERNS:
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="CUSTOM_IDENTIFIER",
                name=f"DenyList_{name}",
                patterns=[Pattern(name=name, regex=regex, score=score)],
            )
        )
    return analyzer


@lru_cache(maxsize=1)
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()  # type: ignore[no-untyped-call]


def reset_analyzer_cache() -> None:
    """Clear cached analyzers (for tests that change the spaCy model)."""
    _analyzer.cache_clear()


class Sanitizer:
    """Replaces PHI with consistent synthetic values.

    One instance should be used per run so that repeated occurrences of
    the same real value map to the same fake value.
    """

    def __init__(
        self,
        *,
        nlp_model: str | None = None,
        score_threshold: float | None = None,
        allowlist: frozenset[str] | set[str] | list[str] | None = None,
        denylist_patterns: list[tuple[str, str, float]] | None = None,
    ) -> None:
        self._faker = Faker()
        self._mapping: dict[tuple[str, str], str] = {}
        self._nlp_model = nlp_model or _env_spacy_model()
        self._score_threshold = (
            _env_score_threshold() if score_threshold is None else score_threshold
        )
        self._allowlist = _normalize_allowlist(
            allowlist if allowlist is not None else DEFAULT_CLINICAL_ALLOWLIST
        )
        self._extra_denylist = denylist_patterns or []
        self.total_redactions: int = 0

    def sanitize(self, text: str | None) -> str | None:
        """Sanitize text; returns only the sanitized string (API-compatible)."""
        return self.sanitize_with_stats(text).text

    def sanitize_with_stats(self, text: str | None) -> SanitizeResult:
        if not text:
            return SanitizeResult(text=text, redaction_count=0)

        # Structured JSON payloads: walk string leaves so Faker commas/quotes
        # cannot corrupt object structure used later in replay comparisons.
        if self._looks_like_json(text):
            return self._sanitize_json(text)

        return self._sanitize_plain(text)

    def _sanitize_plain(self, text: str) -> SanitizeResult:
        results = self._analyze(text)
        if not results:
            return SanitizeResult(text=text, redaction_count=0)

        operators = {
            entity_type: OperatorConfig(
                "custom", {"lambda": partial(self._replacement, entity_type)}
            )
            for entity_type in {r.entity_type for r in results}
        }
        anonymized = _anonymizer().anonymize(
            text=text,
            analyzer_results=results,  # type: ignore[arg-type]
            operators=operators,
        ).text
        count = len(results)
        self.total_redactions += count
        return SanitizeResult(text=anonymized, redaction_count=count)

    def _sanitize_json(self, text: str) -> SanitizeResult:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return self._sanitize_plain(text)

        total = 0

        def walk(value: object) -> object:
            nonlocal total
            if isinstance(value, str):
                result = self._sanitize_plain(value)
                total += result.redaction_count
                # _sanitize_plain already bumped total_redactions
                return result.text
            if isinstance(value, list):
                return [walk(item) for item in value]
            if isinstance(value, dict):
                return {key: walk(item) for key, item in value.items()}
            return value

        sanitized_payload = walk(payload)
        # Re-serialize with compact separators so structure stays stable.
        return SanitizeResult(
            text=json.dumps(sanitized_payload, ensure_ascii=False, separators=(",", ":")),
            redaction_count=total,
        )

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        stripped = text.lstrip()
        if not stripped or stripped[0] not in "{[":
            return False
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return False
        return True

    def _analyze(self, text: str) -> list[RecognizerResult]:
        analyzer = _analyzer(self._nlp_model)
        # Extra denylist patterns are instance-specific; apply via a one-off
        # analyzer copy only when needed (rare). Default denylist is in cache.
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=ENTITY_TYPES,
            score_threshold=self._score_threshold,
        )
        if self._extra_denylist:
            results = list(results) + self._analyze_extra_denylist(text)
            results = self._dedupe_results(results)

        return self._apply_allowlist(text, results)

    def _analyze_extra_denylist(self, text: str) -> list[RecognizerResult]:
        found: list[RecognizerResult] = []
        for name, regex, score in self._extra_denylist:
            for match in re.finditer(regex, text):
                if score >= self._score_threshold:
                    found.append(
                        RecognizerResult(
                            entity_type="CUSTOM_IDENTIFIER",
                            start=match.start(),
                            end=match.end(),
                            score=score,
                            recognition_metadata={"recognizer": name},
                        )
                    )
        return found

    @staticmethod
    def _dedupe_results(results: list[RecognizerResult]) -> list[RecognizerResult]:
        best: dict[tuple[int, int, str], RecognizerResult] = {}
        for result in results:
            key = (result.start, result.end, result.entity_type)
            existing = best.get(key)
            if existing is None or result.score > existing.score:
                best[key] = result
        return list(best.values())

    def _apply_allowlist(
        self, text: str, results: list[RecognizerResult]
    ) -> list[RecognizerResult]:
        if not self._allowlist:
            return results

        filtered: list[RecognizerResult] = []
        for result in results:
            if result.entity_type not in {"PERSON", "LOCATION"}:
                filtered.append(result)
                continue
            span = text[result.start : result.end].strip().lower()
            # Drop if the whole span or any token is an allowlisted clinical term.
            tokens = {span} | set(re.split(r"[\s\-']+", span))
            tokens.discard("")
            if tokens & self._allowlist or span in self._allowlist:
                continue
            filtered.append(result)
        return filtered

    def _replacement(self, entity_type: str, original: str) -> str:
        key = (entity_type, original)
        if key not in self._mapping:
            self._mapping[key] = self._fake_value(entity_type, original)
        return self._mapping[key]

    def _fake_value(self, entity_type: str, original: str) -> str:
        # Regenerate on the (unlikely) collision with the real value so
        # raw PHI can never survive sanitization.
        for _ in range(10):
            fake = self._generate(entity_type)
            if fake.lower() != original.lower():
                return fake
        return f"<{entity_type}>"

    def _generate(self, entity_type: str) -> str:
        faker = self._faker
        match entity_type:
            case "PERSON":
                return faker.name()
            case "PHONE_NUMBER":
                return faker.phone_number()
            case "EMAIL_ADDRESS":
                return faker.email()
            case "DATE_TIME":
                return faker.date()
            case "LOCATION":
                return faker.city()
            case "US_SSN":
                return faker.ssn()
            case "MEDICAL_LICENSE":
                return faker.bothify("??-#######").upper()
            case "MEDICAL_RECORD_NUMBER":
                return faker.numerify("MRN-########")
            case "PATIENT_ID":
                return faker.numerify("PT-########")
            case "STREET_ADDRESS":
                # Avoid commas so flat-text embeddings of addresses stay simple;
                # JSON path sanitizes leaves separately anyway.
                return faker.street_address().replace(",", "")
            case "CUSTOM_IDENTIFIER":
                return faker.bothify("ID-########").upper()
            case _:
                return faker.numerify("ID-########")
