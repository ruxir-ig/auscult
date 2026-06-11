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
