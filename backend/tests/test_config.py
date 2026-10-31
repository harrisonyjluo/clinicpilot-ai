"""Configuration boundaries for optional settings."""

from app.core.config import Settings


def test_blank_default_patient_id_is_treated_as_unset():
    settings = Settings(_env_file=None, default_patient_id="")

    assert settings.default_patient_id is None


def test_whitespace_default_patient_id_is_treated_as_unset():
    settings = Settings(_env_file=None, default_patient_id="   ")

    assert settings.default_patient_id is None


def test_configured_default_patient_id_is_validated():
    settings = Settings(_env_file=None, default_patient_id="P001")

    assert settings.default_patient_id == "P001"


def test_blank_anthropic_base_url_is_treated_as_unset():
    settings = Settings(_env_file=None, anthropic_base_url="")

    assert settings.anthropic_base_url is None
