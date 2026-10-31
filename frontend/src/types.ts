export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
}

export interface Booking {
  booking_id: string;
  patient_id: string;
  doctor_id: string;
  doctor_name: string;
  specialty: string;
  clinic_id: string;
  clinic_name: string;
  treatment_id: string;
  treatment_name: string;
  duration_minutes: number;
  price_usd: string | number;
  booking_date: string;
  booking_time: string;
  status: string;
  notes: string | null;
  created_at: string;
}

export interface BookingListResponse {
  patient_id: string;
  count: number;
  bookings: Booking[];
}

export interface BookingSummary {
  record_id: string;
  booking_id: string;
  visit_date: string;
  chief_complaint: string;
  diagnosis: string;
  medication: string[];
  doctor_instructions: string[];
  follow_up: string | null;
  red_flags: string[];
  confidence: number;
  needs_review: boolean;
  created_at: string;
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
}

export interface ChatRequest {
  message: string;
  patient_id?: string;
  conversation_id?: string;
}

export interface ChatResponse {
  reply: string;
  intent: string;
  conversation_id: string;
  booking: Booking | null;
  tool_calls: ToolCall[];
}

export interface BookingCreatePayload {
  patient_id: string;
  doctor_id: string;
  treatment_id: string;
  booking_date: string;
  booking_time: string;
  notes?: string | null;
  client_request_id?: string;
}

export interface HealthResponse {
  status: string;
  database?: string;
  app?: string;
  version?: string;
}
