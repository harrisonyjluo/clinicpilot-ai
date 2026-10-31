import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, streamChat } from "../api/client";
import { makeBooking, makeChatReply } from "../test/fixtures";
import { ChatPanel } from "./ChatPanel";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    streamChat: vi.fn(),
    getBookings: vi.fn(),
    createBooking: vi.fn(),
    health: vi.fn(),
  };
});

const streamChatMock = vi.mocked(streamChat);

function mockStreamingReply(
  reply: ReturnType<typeof makeChatReply>,
): void {
  streamChatMock.mockImplementation(async (_payload, onDelta) => {
    onDelta(reply.reply);
    return reply;
  });
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("ChatPanel", () => {
  it("shows the configured patient context and the informational-demo disclaimer", () => {
    render(<ChatPanel patientId="P001" />);

    expect(screen.getByText(/Patient P001/)).toBeInTheDocument();
    expect(screen.getByText(/I can answer clinic questions/)).toBeInTheDocument();
    expect(screen.queryByText(/John Doe/i)).not.toBeInTheDocument();
    expect(
      screen.getAllByText(/not medical advice/i).length,
    ).toBeGreaterThan(0);
    expect(screen.getByLabelText("Message")).toBeInTheDocument();
  });

  it("uses the supplied patient id instead of a built-in demo patient", async () => {
    const user = userEvent.setup();
    mockStreamingReply(makeChatReply({ reply: "Hello." }));
    render(<ChatPanel patientId="P777" />);

    expect(screen.getByText(/Patient P777/)).toBeInTheDocument();
    expect(screen.queryByText(/P001/)).not.toBeInTheDocument();
    expect(screen.queryByText(/John Doe/i)).not.toBeInTheDocument();

    await user.type(screen.getByLabelText("Message"), "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(streamChatMock).toHaveBeenCalledWith(
      { message: "Hello", patient_id: "P777" },
      expect.any(Function),
      expect.any(AbortSignal),
    );
  });

  it("sends a message and appends the assistant reply", async () => {
    const user = userEvent.setup();
    mockStreamingReply(
      makeChatReply({
        reply: "Our Saturday hours are 9:00 AM to 1:00 PM.",
      }),
    );
    render(<ChatPanel patientId="P001" />);

    await user.type(
      screen.getByLabelText("Message"),
      "What are your Saturday hours?",
    );
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(streamChatMock).toHaveBeenCalledWith(
      {
        message: "What are your Saturday hours?",
        patient_id: "P001",
      },
      expect.any(Function),
      expect.any(AbortSignal),
    );
    expect(
      await screen.findByText(/Our Saturday hours are 9:00 AM to 1:00 PM\./),
    ).toBeInTheDocument();
  });

  it("renders streamed deltas into one assistant message", async () => {
    const user = userEvent.setup();
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    streamChatMock.mockImplementation(async (_payload, onDelta) => {
      onDelta("Streamed");
      await gate;
      onDelta(" reply");
      return makeChatReply({ reply: "Streamed reply" });
    });
    render(<ChatPanel patientId="P001" />);

    await user.type(screen.getByLabelText("Message"), "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("Streamed")).toBeInTheDocument();
    release();

    expect(await screen.findByText("Streamed reply")).toBeInTheDocument();
    expect(screen.getAllByText("Streamed reply")).toHaveLength(1);
  });

  it("sends a message when Enter is pressed", async () => {
    const user = userEvent.setup();
    mockStreamingReply(makeChatReply({ reply: "Hello back." }));
    render(<ChatPanel patientId="P001" />);

    await user.type(screen.getByLabelText("Message"), "Hello{Enter}");

    expect(streamChatMock).toHaveBeenCalledWith(
      { message: "Hello", patient_id: "P001" },
      expect.any(Function),
      expect.any(AbortSignal),
    );
    expect(await screen.findByText("Hello back.")).toBeInTheDocument();
  });

  it("reuses the conversation id on follow-up turns", async () => {
    const user = userEvent.setup();
    streamChatMock
      .mockImplementationOnce(async (_payload, onDelta) => {
        const reply = makeChatReply({
          reply: "Hello John.",
          conversation_id: "conv-42",
        });
        onDelta(reply.reply);
        return reply;
      })
      .mockImplementationOnce(async (_payload, onDelta) => {
        const reply = makeChatReply({
          reply: "Your next appointment is B1001.",
          conversation_id: "conv-42",
        });
        onDelta(reply.reply);
        return reply;
      });
    render(<ChatPanel patientId="P001" />);

    const input = screen.getByLabelText("Message");
    await user.type(input, "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));
    await screen.findByText("Hello John.");

    await user.type(input, "When is my next appointment?");
    await user.click(screen.getByRole("button", { name: /send/i }));
    await screen.findByText("Your next appointment is B1001.");

    expect(streamChatMock).toHaveBeenNthCalledWith(
      2,
      {
        message: "When is my next appointment?",
        patient_id: "P001",
        conversation_id: "conv-42",
      },
      expect.any(Function),
      expect.any(AbortSignal),
    );
  });

  it("notifies the bookings panel when a booking is created", async () => {
    const user = userEvent.setup();
    const onBookingCreated = vi.fn();
    mockStreamingReply(
      makeChatReply({
        reply: "Your appointment is confirmed.",
        booking: makeBooking({ booking_id: "B1005" }),
      }),
    );
    render(
      <ChatPanel
        patientId="P001"
        onBookingCreated={onBookingCreated}
      />,
    );

    await user.type(
      screen.getByLabelText("Message"),
      "Book a dermatology appointment.",
    );
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText(/B1005/)).toBeInTheDocument();
    expect(onBookingCreated).toHaveBeenCalledTimes(1);
  });

  it("shows the backend error, keeps the user message, and allows retry", async () => {
    const user = userEvent.setup();
    streamChatMock.mockRejectedValue(
      new ApiError(
        "The AI service is not configured.",
        503,
        "AI_NOT_CONFIGURED",
      ),
    );
    render(<ChatPanel patientId="P001" />);

    const input = screen.getByLabelText("Message");
    await user.type(input, "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(
      await screen.findByRole("alert"),
    ).toHaveTextContent(/The AI service is not configured\./);
    expect(
      screen.getByText("Hello", { selector: "p" }),
    ).toBeInTheDocument();
    await waitFor(() => expect(input).toHaveValue("Hello"));
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
  });

  it("does not send a whitespace-only message", async () => {
    const user = userEvent.setup();
    render(<ChatPanel patientId="P001" />);

    await user.type(screen.getByLabelText("Message"), "   ");

    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
    expect(streamChatMock).not.toHaveBeenCalled();
  });
});
