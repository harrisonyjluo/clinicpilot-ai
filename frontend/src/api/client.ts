import type {
  Booking,
  BookingCreatePayload,
  BookingListResponse,
  BookingSummary,
  ChatRequest,
  ChatResponse,
  HealthResponse,
} from "../types";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/$/, "");

interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
    details?: unknown;
  };
}

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly details?: unknown;

  constructor(
    message: string,
    status = 0,
    code?: string,
    details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: unknown }).name === "AbortError"
  );
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  let payload: ApiErrorPayload | null = null;
  try {
    payload = (await response.json()) as ApiErrorPayload;
  } catch {
    payload = null;
  }
  return new ApiError(
    payload?.error?.message ??
      `Request failed with status ${response.status}.`,
    response.status,
    payload?.error?.code,
    payload?.error?.details,
  );
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init.headers,
      },
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    throw new ApiError(
      "Could not reach the backend. Is the API running?",
      0,
    );
  }

  if (!response.ok) {
    throw await errorFromResponse(response);
  }

  return (await response.json()) as T;
}

export function sendChat(
  payload: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

interface ChatStreamEvent {
  type?: string;
  delta?: string;
  response?: ChatResponse;
  error?: { code?: string; message?: string };
}

function parseStreamFrame(frame: string): ChatStreamEvent | null {
  const data = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n");
  if (!data) {
    return null;
  }
  try {
    return JSON.parse(data) as ChatStreamEvent;
  } catch {
    throw new ApiError(
      "The chat stream returned an unreadable event.",
      200,
      "CHAT_STREAM_INVALID",
    );
  }
}

/**
 * POST the chat payload and consume the SSE reply incrementally.
 *
 * Resolves with the authoritative `done` payload so callers keep the same
 * conversation/booking handling as the non-streaming endpoint.
 */
export async function streamChat(
  payload: ChatRequest,
  onDelta: (delta: string) => void,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    throw new ApiError(
      "Could not reach the backend. Is the API running?",
      0,
    );
  }

  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  if (!response.body) {
    throw new ApiError(
      "The chat stream ended before the reply completed.",
      200,
      "CHAT_STREAM_INCOMPLETE",
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: ChatResponse | null = null;

  const consume = (frame: string): boolean => {
    const event = parseStreamFrame(frame);
    if (!event) {
      return false;
    }
    if (event.type === "delta") {
      if (typeof event.delta === "string" && event.delta) {
        onDelta(event.delta);
      }
      return false;
    }
    if (event.type === "error") {
      throw new ApiError(
        event.error?.message ?? "The chat stream failed.",
        200,
        event.error?.code,
      );
    }
    if (event.type === "done" && event.response) {
      final = event.response;
      return true;
    }
    return false;
  };

  try {
    let finished = false;
    while (!finished) {
      const chunk = await reader.read();
      if (chunk.value) {
        buffer += decoder.decode(chunk.value, { stream: true });
      }
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (!finished && boundary >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        finished = consume(frame);
        boundary = buffer.indexOf("\n\n");
      }
      if (chunk.done) {
        buffer += decoder.decode();
        if (buffer.trim()) {
          consume(buffer);
        }
        break;
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!final) {
    throw new ApiError(
      "The chat stream ended before the reply completed.",
      200,
      "CHAT_STREAM_INCOMPLETE",
    );
  }
  return final;
}

export function getBookings(
  patientId: string,
  signal?: AbortSignal,
): Promise<BookingListResponse> {
  return request<BookingListResponse>(
    `/api/bookings?patient_id=${encodeURIComponent(patientId)}`,
    { method: "GET", signal },
  );
}

export function createBooking(
  payload: BookingCreatePayload,
  signal?: AbortSignal,
): Promise<Booking> {
  return request<Booking>("/api/bookings", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export function getBookingSummary(
  bookingId: string,
  signal?: AbortSignal,
): Promise<BookingSummary> {
  return request<BookingSummary>(
    `/api/bookings/${encodeURIComponent(bookingId)}/summary`,
    { method: "GET", signal },
  );
}

export function health(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", {
    method: "GET",
    signal,
  });
}
