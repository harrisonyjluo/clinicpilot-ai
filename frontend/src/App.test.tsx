import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getBookings, streamChat } from "./api/client";
import { makeBooking, makeBookingList, makeChatReply } from "./test/fixtures";
import { App } from "./App";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return {
    ...actual,
    streamChat: vi.fn(),
    getBookings: vi.fn(),
    createBooking: vi.fn(),
    health: vi.fn(),
  };
});

const getBookingsMock = vi.mocked(getBookings);
const streamChatMock = vi.mocked(streamChat);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("App", () => {
  it("shows a newly created booking without a page reload", async () => {
    const user = userEvent.setup();
    getBookingsMock
      .mockResolvedValueOnce(makeBookingList([makeBooking()]))
      .mockResolvedValueOnce(
        makeBookingList([
          makeBooking(),
          makeBooking({ booking_id: "B1005" }),
        ]),
      );
    streamChatMock.mockImplementation(async (_payload, onDelta) => {
      const reply = makeChatReply({
        reply: "Your appointment is confirmed.",
        booking: makeBooking({ booking_id: "B1005" }),
      });
      onDelta(reply.reply);
      return reply;
    });

    render(<App patientId="P001" />);
    await screen.findByText("B1001");

    await user.type(
      screen.getByLabelText("Message"),
      "Book a dermatology appointment.",
    );
    await user.click(screen.getByRole("button", { name: /send/i }));

    await user.click(screen.getByRole("tab", { name: "Bookings" }));
    const table = await screen.findByRole("table");
    expect(await within(table).findByText("B1005")).toBeInTheDocument();
    expect(getBookingsMock).toHaveBeenCalledTimes(2);
  });

  it("passes the configured patient id to both panels", async () => {
    getBookingsMock.mockResolvedValue(makeBookingList([]));

    render(<App patientId="P777" />);

    expect(await screen.findAllByText(/Patient P777/)).toHaveLength(2);
    expect(getBookingsMock).toHaveBeenCalledWith(
      "P777",
      expect.any(AbortSignal),
    );
    expect(screen.queryByText(/P001/)).not.toBeInTheDocument();
    expect(screen.queryByText(/John Doe/i)).not.toBeInTheDocument();
  });

  it("uses neutral product copy without implementation jargon", () => {
    getBookingsMock.mockResolvedValue(makeBookingList([]));

    render(<App patientId="P001" />);

    expect(screen.getByText(/ask, plan, and book/i)).toBeInTheDocument();
    expect(screen.queryByText(/phase\s*\d/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/react ui/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/csv/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/live from the api/i)).not.toBeInTheDocument();
  });

  it("switches between the chat and bookings tabs, showing one panel at a time", async () => {
    const user = userEvent.setup();
    getBookingsMock.mockResolvedValue(makeBookingList([]));

    render(<App patientId="P001" />);

    const chatTab = screen.getByRole("tab", { name: "Chat" });
    const bookingsTab = screen.getByRole("tab", { name: "Bookings" });
    const chatPanel = screen.getByRole("tabpanel", { name: "Chat" });
    const bookingsPanel = document.getElementById("panel-bookings");

    expect(chatTab).toHaveAttribute("aria-selected", "true");
    expect(bookingsTab).toHaveAttribute("aria-selected", "false");
    expect(chatPanel).not.toHaveAttribute("hidden");
    expect(bookingsPanel).toHaveAttribute("role", "tabpanel");
    expect(bookingsPanel).toHaveAttribute("hidden");

    await user.click(bookingsTab);

    expect(bookingsTab).toHaveAttribute("aria-selected", "true");
    expect(chatTab).toHaveAttribute("aria-selected", "false");
    expect(
      screen.getByRole("tabpanel", { name: "Bookings" }),
    ).not.toHaveAttribute("hidden");
    expect(document.getElementById("panel-chat")).toHaveAttribute("hidden");
  });
});
