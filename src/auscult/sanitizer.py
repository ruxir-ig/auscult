"""PHI detection and replacement for trace text.

Uses Presidio to find PHI entities and Faker to substitute realistic
synthetic values. A Sanitizer instance keeps a mapping of real -> fake
values so the same entity is replaced consistently within one run.
"""

from functools import lru_cache, partial

from faker import Faker
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
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
]

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


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    """Build the Presidio analyzer once; it is expensive to construct."""
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["en"],
    )
    for recognizer in _CUSTOM_RECOGNIZERS:
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


@lru_cache(maxsize=1)
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


class Sanitizer:
    """Replaces PHI with consistent synthetic values.

    One instance should be used per run so that repeated occurrences of
    the same real value map to the same fake value.
    """

    def __init__(self) -> None:
        self._faker = Faker()
        self._mapping: dict[tuple[str, str], str] = {}

    def sanitize(self, text: str | None) -> str | None:
        if not text:
            return text

        results = _analyzer().analyze(text=text, language="en", entities=ENTITY_TYPES)
        if not results:
            return text

        operators = {
            entity_type: OperatorConfig(
                "custom", {"lambda": partial(self._replacement, entity_type)}
            )
            for entity_type in {r.entity_type for r in results}
        }
        return _anonymizer().anonymize(
            text=text, analyzer_results=results, operators=operators
        ).text

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
                return faker.street_address()
            case _:
                return faker.numerify("ID-########")
