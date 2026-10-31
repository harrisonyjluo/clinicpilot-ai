import type {
  Booking,
  BookingListResponse,
  BookingSummary,
  ChatResponse,
} from "../types";

export function makeBooking(overrides: Partial<Booking> = {}): Booking {
  return {
    booking_id: "B1001",
    patient_id: "P001",
    doctor_id: "D101",
    doctor_name: "Dr. James Carter",
    specialty: "General Practice",
    clinic_id: "C001",
    clinic_name: "Metro Care Health Clinic",
    treatment_id: "T001",
    treatment_name: "Annual Physical Examination",
    duration_minutes: 45,
    price_usd: "150.00",
    booking_date: "2026-11-05",
    booking_time: "09:00:00",
    status: "Confirmed",
    notes: null,
    created_at: "2026-10-20T09:00:00Z",
    ...overrides,
  };
}

export function makeBookingList(
  bookings: Booking[] = [makeBooking()],
): BookingListResponse {
  return {
    patient_id: "P001",
    count: bookings.length,
    bookings,
  };
}

export function makeBookingSummary(
  overrides: Partial<BookingSummary> = {},
): BookingSummary {
  return {
    record_id: "H9003",
    booking_id: "B1001",
    visit_date: "2026-11-20",
    chief_complaint: "itchy rash on the left arm",
    diagnosis: "not stated",
    medication: ["hydrocortisone cream"],
    doctor_instructions: ["keep the area clean", "avoid new lotions"],
    follow_up: "return if the rash spreads",
    red_flags: [],
    confidence: 0.91,
    needs_review: true,
    created_at: "2026-11-01T10:00:00Z",
    ...overrides,
  };
}

export function makeChatReply(
  overrides: Partial<ChatResponse> = {},
): ChatResponse {
  return {
    reply: "I can help with that.",
    intent: "clinic_info",
    conversation_id: "conv-1",
    booking: null,
    tool_calls: [],
    ...overrides,
  };
}
