"""Knowledge base over the provided clinic CSVs.

The knowledge base is read-only. It loads the three public knowledge CSVs,
joins doctors to clinics and treatments to specialties, and builds natural
language retrieval text for the dense-vector index.

``retrieval_text`` is index-only. It is never returned as a clinic fact; only
the original ``payload`` rows are quoted to the model. The short semantic
descriptions below fill gaps in the sparse CSVs (for example "breakouts" for a
dermatology consultation). At larger scale these descriptions should come from
the source data or a Doc2Query/LLM enrichment step, not from query-time rules.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_DIR

DEFAULT_DATA_DIR = PROJECT_DIR / "mock-data"

REQUIRED_COLUMNS = {
    "clinics.csv": {
        "clinic_id",
        "clinic_name",
        "address",
        "phone",
        "operating_hours",
        "cancellation_policy",
        "emergency_instructions",
    },
    "doctors.csv": {
        "doctor_id",
        "clinic_id",
        "full_name",
        "specialty",
        "languages_spoken",
        "consultation_fee_usd",
        "bio",
    },
    "treatments.csv": {
        "treatment_id",
        "treatment_name",
        "category",
        "duration_minutes",
        "price_usd",
        "pre_treatment_instructions",
    },
}

CATEGORY_TO_SPECIALTY = {
    "Dermatology": "Dermatology",
    "Orthopedics": "Orthopedics",
    "Pediatrics": "Pediatrics",
    "Immunization": "General Practice",
    "Preventative Care": "General Practice",
}

# Index-only natural-language context. These sentences describe what a service
# means in ordinary patient language; they must not introduce new clinic facts.
_TREATMENT_CONTEXT = {
    "T001": (
        "It is an annual physical examination, yearly physical examination, "
        "routine physical exam, annual wellness exam, general medical "
        "checkup, primary care check-up, or family doctor appointment for "
        "adults. Adults schedule a yearly physical once a year as a routine "
        "wellness check."
    ),
    "T002": (
        "It is for infants, babies, children, kids, toddlers, and adolescents, "
        "and provides childhood shots, school vaccination requirements, "
        "routine immunizations, and pediatric preventive care."
    ),
    "T003": (
        "It is for skin concerns of the face and body, including rashes, acne, "
        "breakouts, pimples, eczema, moles, skin cancer screening, and cosmetic "
        "skin problems. People describe this as breaking out on the face or "
        "body. A dermatologist evaluates the skin."
    ),
    "T004": (
        "It provides the influenza vaccine, seasonal flu shot, flu jab, and "
        "adult vaccination by injection."
    ),
    "T005": (
        "It is for joint pain, knee pain, back pain, ankle sprains, twisted "
        "ankles, soccer injuries, sports injuries, muscle pain, and bone pain, "
        "with an orthopedic evaluation and physiotherapy or rehabilitation "
        "guidance for people who run or exercise."
    ),
}


@dataclass(frozen=True)
class KnowledgeDocument:
    """One searchable clinic, doctor, or treatment record."""

    kind: str
    id: str
    title: str
    text: str
    payload: dict[str, Any]
    clinic_id: str | None = None
    specialty: str | None = None
    category: str | None = None


def _read_csv(directory: Path, filename: str) -> list[dict[str, str]]:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing knowledge file: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS[filename] - columns
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        return [
            {key.strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def _coerce_numbers(
    rows: list[dict[str, Any]],
    *,
    integer_fields: tuple[str, ...] = (),
    decimal_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in decimal_fields:
            row[field] = float(row[field])
    return rows


def _source_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("clinics.csv", "doctors.csv", "treatments.csv"):
        path = directory / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _saturday_phrase(operating_hours: str) -> str:
    """Describe Saturday hours in natural language at index time."""

    parts = [part.strip() for part in operating_hours.split(",")]
    saturday = next(
        (part for part in parts if part.casefold().startswith("sat")),
        "",
    )
    if not saturday:
        return ""
    if "closed" in saturday.casefold():
        return (
            "The clinic is not open on Saturday or Sunday; it is closed on "
            "weekends."
        )
    hours = saturday.split(":", 1)[-1].strip()
    return (
        f"Our Saturday clinic hours are {hours}, and patients can visit on "
        "Saturday morning. The clinic is open on Saturday."
    )


class KnowledgeBase:
    """Typed records plus joined views. Read-only for the life of the process."""

    def __init__(
        self,
        clinics: list[dict[str, Any]],
        doctors: list[dict[str, Any]],
        treatments: list[dict[str, Any]],
        *,
        source_fingerprint: str = "",
    ) -> None:
        self.clinics = clinics
        self.doctors = doctors
        self.treatments = treatments
        self.source_fingerprint = source_fingerprint
        self._clinics_by_id = {row["clinic_id"]: row for row in clinics}
        self._doctors_by_id = {row["doctor_id"]: row for row in doctors}
        self._treatments_by_id = {row["treatment_id"]: row for row in treatments}
        self.documents = self._build_documents()

    @classmethod
    def load_from_directory(cls, directory: Path) -> KnowledgeBase:
        clinics = _read_csv(directory, "clinics.csv")
        doctors = _coerce_numbers(
            _read_csv(directory, "doctors.csv"),
            decimal_fields=("consultation_fee_usd",),
        )
        treatments = _coerce_numbers(
            _read_csv(directory, "treatments.csv"),
            integer_fields=("duration_minutes",),
            decimal_fields=("price_usd",),
        )
        return cls(
            clinics=clinics,
            doctors=doctors,
            treatments=treatments,
            source_fingerprint=_source_fingerprint(directory),
        )

    @classmethod
    def load_default(cls) -> KnowledgeBase:
        return _load_default()

    def _doctors_for_clinic(self, clinic_id: str) -> list[dict[str, Any]]:
        return self.doctors_for_clinic(clinic_id)

    def doctors_for_clinic(self, clinic_id: str) -> list[dict[str, Any]]:
        return [
            doctor
            for doctor in self.doctors
            if doctor["clinic_id"] == clinic_id
        ]

    def _treatments_for_specialty(
        self,
        specialty: str,
    ) -> list[dict[str, Any]]:
        return [
            treatment
            for treatment in self.treatments
            if CATEGORY_TO_SPECIALTY.get(treatment["category"]) == specialty
        ]

    def _build_documents(self) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []

        for clinic in self.clinics:
            clinic_doctors = self._doctors_for_clinic(clinic["clinic_id"])
            doctor_text = ", ".join(
                f"{doctor['full_name']} ({doctor['specialty']})"
                for doctor in clinic_doctors
            )
            text = " ".join(
                part
                for part in (
                    f"{clinic['clinic_name']} is a medical clinic located at "
                    f"{clinic['address']}.",
                    f"Phone {clinic['phone']}.",
                    f"Opening hours: {clinic['operating_hours']}.",
                    _saturday_phrase(clinic["operating_hours"]),
                    f"Cancellation policy: {clinic['cancellation_policy']}",
                    f"Emergency instructions: {clinic['emergency_instructions']}",
                    (
                        f"Doctors at this clinic include {doctor_text}."
                        if doctor_text
                        else ""
                    ),
                )
                if part
            )
            documents.append(
                KnowledgeDocument(
                    kind="clinic",
                    id=clinic["clinic_id"],
                    title=clinic["clinic_name"],
                    text=text,
                    payload=clinic,
                    clinic_id=clinic["clinic_id"],
                )
            )

        for doctor in self.doctors:
            clinic = self._clinics_by_id.get(doctor["clinic_id"], {})
            treatments = self._treatments_for_specialty(doctor["specialty"])
            treatment_text = ", ".join(
                treatment["treatment_name"] for treatment in treatments
            )
            text = " ".join(
                part
                for part in (
                    f"{doctor['full_name']} is a {doctor['specialty']} doctor "
                    f"at {clinic.get('clinic_name', 'the clinic')}.",
                    doctor["bio"],
                    f"{doctor['full_name']} speaks "
                    f"{doctor['languages_spoken']}.",
                    (
                        f"The consultation fee is "
                        f"${doctor['consultation_fee_usd']:.2f}."
                    ),
                    (
                        f"Services include {treatment_text}."
                        if treatment_text
                        else ""
                    ),
                )
                if part
            )
            documents.append(
                KnowledgeDocument(
                    kind="doctor",
                    id=doctor["doctor_id"],
                    title=doctor["full_name"],
                    text=text,
                    payload=doctor,
                    clinic_id=doctor["clinic_id"],
                    specialty=doctor["specialty"],
                )
            )

        for treatment in self.treatments:
            doctor = self.doctor_for_treatment(treatment["treatment_id"])
            clinic = (
                self.get_clinic(doctor["clinic_id"])
                if doctor is not None
                else None
            )
            service_party = ""
            if doctor is not None and clinic is not None:
                service_party = (
                    f"This service is provided by {doctor['full_name']} at "
                    f"{clinic['clinic_name']}."
                )
            text = " ".join(
                part
                for part in (
                    f"{treatment['treatment_name']} is a "
                    f"{treatment['category']} service lasting "
                    f"{treatment['duration_minutes']} minutes and costing "
                    f"${treatment['price_usd']:.2f}.",
                    _TREATMENT_CONTEXT.get(treatment["treatment_id"], ""),
                    f"Before treatment: {treatment['pre_treatment_instructions']}",
                    service_party,
                )
                if part
            )
            documents.append(
                KnowledgeDocument(
                    kind="treatment",
                    id=treatment["treatment_id"],
                    title=treatment["treatment_name"],
                    text=text,
                    payload=treatment,
                    category=treatment["category"],
                )
            )
        return documents

    def get_clinic(self, clinic_id: str) -> dict[str, Any] | None:
        return self._clinics_by_id.get(clinic_id)

    def get_doctor(self, doctor_id: str) -> dict[str, Any] | None:
        return self._doctors_by_id.get(doctor_id)

    def get_treatment(self, treatment_id: str) -> dict[str, Any] | None:
        return self._treatments_by_id.get(treatment_id)

    def doctors_for_specialty(self, specialty: str) -> list[dict[str, Any]]:
        normalized = specialty.casefold()
        return [
            doctor
            for doctor in self.doctors
            if doctor["specialty"].casefold() == normalized
        ]

    def doctor_for_treatment(
        self,
        treatment_id: str,
        *,
        clinic_id: str | None = None,
    ) -> dict[str, Any] | None:
        treatment = self.get_treatment(treatment_id)
        if treatment is None:
            return None
        specialty = CATEGORY_TO_SPECIALTY.get(treatment["category"])
        if specialty is None:
            return None
        doctors = self.doctors_for_specialty(specialty)
        if clinic_id is not None:
            doctors = [
                doctor
                for doctor in doctors
                if doctor["clinic_id"] == clinic_id
            ]
        return doctors[0] if doctors else None

    def treatments_for_doctor(self, doctor_id: str) -> list[dict[str, Any]]:
        doctor = self.get_doctor(doctor_id)
        if doctor is None:
            return []
        return self._treatments_for_specialty(doctor["specialty"])

    def treatment_for_doctor(self, doctor_id: str) -> dict[str, Any] | None:
        treatments = self.treatments_for_doctor(doctor_id)
        return treatments[0] if treatments else None


@lru_cache(maxsize=1)
def _load_default() -> KnowledgeBase:
    return KnowledgeBase.load_from_directory(DEFAULT_DATA_DIR)
