"""Slot merging, stage transitions, and in-memory conversation isolation."""

from datetime import date, time

from app.agent.state import BookingSlots, BookingStage, ConversationStore


def test_new_slots_list_every_required_field_in_order():
    slots = BookingSlots()

    assert slots.missing_slots() == [
        "doctor_id",
        "treatment_id",
        "booking_date",
        "booking_time",
    ]
    assert slots.is_complete is False


def test_merge_preserves_collected_values_and_fills_new_ones():
    slots = BookingSlots(treatment_id="T003", doctor_id="D103")

    slots.merge(BookingSlots(booking_time=time(10, 0)))

    assert slots.doctor_id == "D103"
    assert slots.treatment_id == "T003"
    assert slots.booking_time == time(10, 0)
    assert slots.missing_slots() == ["booking_date"]


def test_merge_does_not_erase_a_value_with_none():
    slots = BookingSlots(doctor_id="D103", treatment_id="T003", notes="rash")

    slots.merge(BookingSlots(notes=None))

    assert slots.doctor_id == "D103"
    assert slots.treatment_id == "T003"
    assert slots.notes == "rash"


def test_complete_booking_needs_all_four_slots():
    slots = BookingSlots(
        doctor_id="D103",
        treatment_id="T003",
        booking_date=date(2026, 11, 3),
        booking_time=time(10, 0),
    )

    assert slots.is_complete is True
    assert slots.missing_slots() == []


def test_store_returns_the_same_conversation_state():
    store = ConversationStore()

    first = store.get_or_create("demo-1", "P001")
    second = store.get_or_create("demo-1", "P001")

    assert second is first


def test_reset_clears_booking_state_but_keeps_the_conversation_id():
    store = ConversationStore()
    state = store.get_or_create("demo-1", "P001")
    state.stage = BookingStage.CONFIRMING
    state.slots.doctor_id = "D103"
    state.client_request_id = "request-1"

    store.reset("demo-1")

    fresh = store.get_or_create("demo-1", "P001")
    assert fresh.stage is BookingStage.IDLE
    assert fresh.slots == BookingSlots()
    assert fresh.pending is None
    assert fresh.client_request_id is None


def test_changing_patient_in_a_conversation_resets_booking_slots():
    store = ConversationStore()
    state = store.get_or_create("demo-1", "P001")
    state.slots.doctor_id = "D103"

    replacement = store.get_or_create("demo-1", "P002")

    assert replacement is not state
    assert replacement.patient_id == "P002"
    assert replacement.slots == BookingSlots()


def test_reset_clears_the_clinic_preference():
    store = ConversationStore()
    state = store.get_or_create("demo-1", "P001")
    state.clinic_preference = "C001"

    store.reset("demo-1")

    fresh = store.get_or_create("demo-1", "P001")
    assert fresh.clinic_preference is None
