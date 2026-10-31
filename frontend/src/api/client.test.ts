import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  getBookingSummary,
  getBookings,
  sendChat,
  streamChat,
} from "./client";
import {
  makeBookingList,
  makeBookingSummary,
  makeChatReply,
} from "../test/fixtures";

const fetchMock = vi.fn();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

function streamingResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return {
    ok: true,
    status: 200,
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    }),
  } as unknown as Response;
}

function sse(event: unknown): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("API client", () => {
  it("requests bookings for the given patient", async () => {
    fetchMock.mockResolvedValue(jsonResponse(makeBookingList()));

    const result = await getBookings("P001");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/bookings?patient_id=P001",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
    expect(result.bookings[0].booking_id).toBe("B1001");
  });

  it("posts the chat payload with an optional conversation id", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(makeChatReply({ conversation_id: "conv-9" })),
    );

    await sendChat({
      message: "When is my next appointment?",
      patient_id: "P001",
      conversation_id: "conv-9",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/chat",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          message: "When is my next appointment?",
          patient_id: "P001",
          conversation_id: "conv-9",
        }),
      }),
    );
  });

  it("requests the structured summary for a booking", async () => {
    fetchMock.mockResolvedValue(jsonResponse(makeBookingSummary()));

    const result = await getBookingSummary("B1001");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/bookings/B1001/summary",
      expect.objectContaining({ method: "GET" }),
    );
    expect(result.chief_complaint).toBe("itchy rash on the left arm");
    expect(result.doctor_instructions).toEqual([
      "keep the area clean",
      "avoid new lotions",
    ]);
  });

  it("streams deltas and resolves with the terminal chat response", async () => {
    const reply = makeChatReply({ reply: "Hello world" });
    const deltaOne = sse({ type: "delta", delta: "Hello " });
    const deltaTwo = sse({ type: "delta", delta: "world" });
    const done = sse({ type: "done", response: reply });
    fetchMock.mockResolvedValue(
      streamingResponse([
        deltaOne.slice(0, 20),
        deltaOne.slice(20) + deltaTwo,
        done,
      ]),
    );
    const deltas: string[] = [];

    const result = await streamChat(
      { message: "Hello", patient_id: "P001" },
      (delta) => deltas.push(delta),
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/chat/stream",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          message: "Hello",
          patient_id: "P001",
        }),
      }),
    );
    expect(deltas).toEqual(["Hello ", "world"]);
    expect(result.reply).toBe("Hello world");
  });

  it("turns a mid-stream error event into an ApiError", async () => {
    fetchMock.mockResolvedValue(
      streamingResponse([
        sse({
          type: "error",
          error: {
            code: "AI_UPSTREAM_UNAVAILABLE",
            message: "The AI provider could not be reached.",
          },
        }),
      ]),
    );

    await expect(
      streamChat({ message: "Hello", patient_id: "P001" }, () => undefined),
    ).rejects.toMatchObject({
      name: "ApiError",
      code: "AI_UPSTREAM_UNAVAILABLE",
      message: "The AI provider could not be reached.",
    });
  });

  it("rejects a stream that ends without a done event", async () => {
    fetchMock.mockResolvedValue(
      streamingResponse([sse({ type: "delta", delta: "Partial" })]),
    );

    await expect(
      streamChat({ message: "Hello", patient_id: "P001" }, () => undefined),
    ).rejects.toMatchObject({
      name: "ApiError",
      code: "CHAT_STREAM_INCOMPLETE",
    });
  });

  it("turns a structured backend error into an ApiError", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          error: {
            code: "AI_NOT_CONFIGURED",
            message: "The AI service is not configured.",
            details: { hint: "Set ANTHROPIC_API_KEY" },
          },
        },
        503,
      ),
    );

    await expect(sendChat({ message: "Hello" })).rejects.toMatchObject({
      name: "ApiError",
      status: 503,
      code: "AI_NOT_CONFIGURED",
      message: "The AI service is not configured.",
    });
    await expect(sendChat({ message: "Hello" })).rejects.toBeInstanceOf(
      ApiError,
    );
  });

  it("reports an unreachable backend as an ApiError", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(getBookings("P001")).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
      message: expect.stringMatching(/could not reach/i),
    });
  });
});
