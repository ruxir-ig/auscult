import json

from auscult.sanitizer import Sanitizer


def test_replaces_person_name() -> None:
    result = Sanitizer().sanitize("Patient John Smith reports chest pain.")
    assert "John Smith" not in result
    assert "chest pain" in result


def test_replaces_phone_number() -> None:
    result = Sanitizer().sanitize("Call the patient back at 212-555-0182 tomorrow.")
    assert "212-555-0182" not in result


def test_replaces_email_address() -> None:
    result = Sanitizer().sanitize("Send the lab results to john.smith@example.com please.")
    assert "john.smith@example.com" not in result


def test_replaces_date() -> None:
    result = Sanitizer().sanitize("The patient was admitted on January 5, 2024.")
    assert "January 5, 2024" not in result


def test_replaces_medical_record_number() -> None:
    result = Sanitizer().sanitize("Pull the chart for MRN: 48293012 from records.")
    assert "48293012" not in result


def test_consistent_replacement_within_one_sanitizer() -> None:
    sanitizer = Sanitizer()
    first = sanitizer.sanitize("John Smith has a fever.")
    second = sanitizer.sanitize("John Smith was given ibuprofen.")

    fake_first = first.replace(" has a fever.", "")
    fake_second = second.replace(" was given ibuprofen.", "")
    assert fake_first == fake_second
    assert fake_first != "John Smith"


def test_consistent_replacement_within_one_text() -> None:
    sanitizer = Sanitizer()
    result = sanitizer.sanitize(
        "John Smith reported pain. Later John Smith asked for water."
    )
    assert "John Smith" not in result
    # Both occurrences must map to the same fake name, so after removing
    # the static text only one distinct name remains.
    first_name = result.split(" reported pain.")[0]
    assert result.count(first_name) == 2


def test_raw_phi_not_present_in_sanitized_output() -> None:
    text = (
        "John Smith (MRN: 48293012) called from 212-555-0182 on January 5, 2024 "
        "and asked to email results to john.smith@example.com."
    )
    result = Sanitizer().sanitize(text)
    for phi in [
        "John Smith",
        "48293012",
        "212-555-0182",
        "January 5, 2024",
        "john.smith@example.com",
    ]:
        assert phi not in result


def test_none_and_empty_text_pass_through() -> None:
    sanitizer = Sanitizer()
    assert sanitizer.sanitize(None) is None
    assert sanitizer.sanitize("") == ""


def test_text_without_phi_unchanged() -> None:
    text = "Order a CBC panel and check inflammation markers."
    assert Sanitizer().sanitize(text) == text


def test_replaces_street_address() -> None:
    result = Sanitizer().sanitize("Patient lives at 1428 Elm Street, Apt 4B.")
    assert "1428 Elm Street" not in result


def test_replaces_mrn_variants() -> None:
    sanitizer = Sanitizer()
    for text, phi in [
        ("Chart MRN#4829301 was flagged.", "4829301"),
        ("See MRN 48293012 for history.", "48293012"),
        ("Medical Record Number: 992834771 attached.", "992834771"),
        ("Lookup medical record no. 5582901.", "5582901"),
    ]:
        assert phi not in sanitizer.sanitize(text)


def test_replaces_dob_formats() -> None:
    sanitizer = Sanitizer()
    for text, phi in [
        ("DOB: 01/05/1980 confirmed.", "01/05/1980"),
        ("Date of birth 1980-01-05 on file.", "1980-01-05"),
        ("Patient born on 5-1-80.", "5-1-80"),
        ("DOB 01.05.1980 per intake form.", "01.05.1980"),
    ]:
        assert phi not in sanitizer.sanitize(text)


def test_replaces_patient_id_without_mrn_label() -> None:
    sanitizer = Sanitizer()
    for text, phi in [
        ("Patient ID: 48293012 admitted today.", "48293012"),
        ("Records for PT-5582901 are pending.", "5582901"),
        ("Flagged PID 99283477 for review.", "99283477"),
        ("Patient 48293012 missed the appointment.", "48293012"),
    ]:
        assert phi not in sanitizer.sanitize(text)


def test_clinical_free_text_strips_all_phi() -> None:
    text = (
        "Pt John Smith, DOB 03/14/1962, MRN#7728190, presented to clinic at "
        "920 Maple Avenue with dyspnea. Spouse Mary Smith reachable at "
        "646-555-0144. Follow-up scheduled for 04/02/2024; fax results to "
        "msmith@example.org."
    )
    result = Sanitizer().sanitize(text)
    for phi in [
        "John Smith",
        "03/14/1962",
        "7728190",
        "920 Maple Avenue",
        "Mary Smith",
        "646-555-0144",
        "04/02/2024",
        "msmith@example.org",
    ]:
        assert phi not in result
    # Clinical content should survive.
    assert "dyspnea" in result


def test_clinical_allowlist_keeps_disease_eponyms() -> None:
    text = "Differential includes Parkinson's disease and Addison's disease."
    result = Sanitizer().sanitize(text)
    assert "Parkinson" in result
    assert "Addison" in result


def test_denylist_redacts_badge_and_case_ids() -> None:
    sanitizer = Sanitizer()
    badge = sanitizer.sanitize("Assigned nurse BADGE:A12B34 on floor 3.")
    case = sanitizer.sanitize("Opened CASE#99887766 for follow-up.")
    assert "A12B34" not in badge
    assert "99887766" not in case


def test_score_threshold_is_honored() -> None:
    # Extremely high threshold should suppress low-confidence hits and leave
    # plain clinical text alone; PHI with strong pattern scores may still go.
    high = Sanitizer(score_threshold=0.99)
    assert "CBC" in (high.sanitize("Order a CBC panel.") or "")


def test_json_payload_preserves_structure() -> None:
    payload = {
        "tool": "notify_patient",
        "args": {
            "name": "John Smith",
            "phone": "212-555-0182",
            "address": "1428 Elm Street",
            "note": "Order CBC",
        },
    }
    raw = json.dumps(payload)
    result = Sanitizer().sanitize_with_stats(raw)
    assert result.text is not None
    parsed = json.loads(result.text)
    assert parsed["tool"] == "notify_patient"
    assert parsed["args"]["note"] == "Order CBC"
    assert "John Smith" not in parsed["args"]["name"]
    assert "212-555-0182" not in parsed["args"]["phone"]
    assert "1428 Elm Street" not in parsed["args"]["address"]
    assert result.redaction_count >= 1


def test_sanitize_with_stats_counts_redactions() -> None:
    result = Sanitizer().sanitize_with_stats(
        "Call John Smith at 212-555-0182 about the CBC."
    )
    assert result.text is not None
    assert "John Smith" not in result.text
    assert result.redaction_count >= 1


def test_custom_denylist_patterns() -> None:
    sanitizer = Sanitizer(
        denylist_patterns=[("site_code", r"\bSITE-[A-Z]{3}\d{3}\b", 0.95)]
    )
    result = sanitizer.sanitize("Route specimen to SITE-NYC001 today.")
    assert "SITE-NYC001" not in result


def test_deterministic_seed_produces_stable_replacements() -> None:
    text = "Call John Smith at 212-555-0182 about the CBC."
    first = Sanitizer(seed="run-abc").sanitize(text)
    second = Sanitizer(seed="run-abc").sanitize(text)
    third = Sanitizer(seed="run-xyz").sanitize(text)
    assert first == second
    assert first != third
    assert "John Smith" not in (first or "")


def test_entity_counts_reported_without_raw_spans() -> None:
    result = Sanitizer().sanitize_with_stats(
        "Call John Smith at 212-555-0182 about the CBC."
    )
    assert result.entity_counts
    assert all(isinstance(v, int) and v > 0 for v in result.entity_counts.values())
    # Audit payload must not embed the original PHI spans as keys/values.
    blob = str(result.entity_counts)
    assert "John Smith" not in blob
    assert "212-555-0182" not in blob
