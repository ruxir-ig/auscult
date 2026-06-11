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
