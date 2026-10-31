import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, getBookingSummary, getBookings } from "../api/client";
import {
  makeBooking,
  makeBookingList,
  makeBookingSummary,
} from "../test/fixtures";
import { BookingsPanel } from "./BookingsPanel";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    streamChat: vi.fn(),
    getBookings: vi.fn(),
    getBookingSummary: vi.fn(),
    createBooking: vi.fn(),
    health: vi.fn(),
  };
});

const getBookingsMock = vi.mocked(getBookings);
const getBookingSummaryMock = vi.mocked(getBookingSummary);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("BookingsPanel", () => {
  it("loads and renders the patient's bookings", async () => {
    getBookingsMock.mockResolvedValue(
      makeBookingList([
        makeBooking(),
        makeBooking({
          booking_id: "B1004",
          doctor_name: "Dr. Sofia Ramirez",
          treatment_name: "Orthopedic Consultation",
          booking_date: "2026-11-15",
          booking_time: "11:00:00",
        }),
      ]),
    );

    render(<BookingsPanel patientId="P001" />);

    expect(await screen.findByText("B1001")).toBeInTheDocument();
    expect(screen.getByText("B1004")).toBeInTheDocument();
    expect(screen.getByText("Dr. James Carter")).toBeInTheDocument();
    expect(screen.getByText("Appointments for P001")).toBeInTheDocument();
    expect(screen.getByText(/Patient P001/)).toBeInTheDocument();
    expect(screen.getByText(/last updated/i)).toBeInTheDocument();
    expect(getBookingsMock).toHaveBeenCalledWith(
      "P001",
      expect.any(AbortSignal),
    );
  });

  it("uses the supplied patient id for the request and header", async () => {
    getBookingsMock.mockResolvedValue(makeBookingList([]));

    render(<BookingsPanel patientId="P777" />);

    expect(await screen.findByText(/Patient P777/)).toBeInTheDocument();
    expect(getBookingsMock).toHaveBeenCalledWith(
      "P777",
      expect.any(AbortSignal),
    );
    expect(screen.queryByText(/P001/)).not.toBeInTheDocument();
  });

  it("refreshes when the refresh button is clicked", async () => {
    const user = userEvent.setup();
    getBookingsMock
      .mockResolvedValueOnce(makeBookingList([makeBooking()]))
      .mockResolvedValueOnce(
        makeBookingList([
          makeBooking(),
          makeBooking({ booking_id: "B1005" }),
        ]),
      );

    render(<BookingsPanel patientId="P001" />);
    await screen.findByText("B1001");

    await user.click(screen.getByRole("button", { name: /refresh/i }));

    expect(await screen.findByText("B1005")).toBeInTheDocument();
    expect(getBookingsMock).toHaveBeenCalledTimes(2);
  });

  it("refreshes when the parent bumps the refresh token", async () => {
    getBookingsMock
      .mockResolvedValueOnce(makeBookingList([makeBooking()]))
      .mockResolvedValueOnce(
        makeBookingList([
          makeBooking(),
          makeBooking({ booking_id: "B1005" }),
        ]),
      );

    const { rerender } = render(
      <BookingsPanel patientId="P001" refreshToken={0} />,
    );
    await screen.findByText("B1001");

    rerender(<BookingsPanel patientId="P001" refreshToken={1} />);

    expect(await screen.findByText("B1005")).toBeInTheDocument();
    expect(getBookingsMock).toHaveBeenCalledTimes(2);
  });

  it("exposes the booking list as a keyboard-scrollable region", async () => {
    getBookingsMock.mockResolvedValue(makeBookingList([makeBooking()]));

    render(<BookingsPanel patientId="P001" />);

    const region = await screen.findByRole("region", {
      name: /bookings table/i,
    });
    expect(region).toHaveAttribute("tabindex", "0");
    expect(within(region).getByRole("table")).toBeInTheDocument();
  });

  it("shows an empty state instead of a blank panel", async () => {
    getBookingsMock.mockResolvedValue(makeBookingList([]));

    render(<BookingsPanel patientId="P001" />);

    expect(await screen.findByText(/no bookings/i)).toBeInTheDocument();
  });

  it("surfaces a backend error", async () => {
    getBookingsMock.mockRejectedValue(
      new ApiError("Could not reach the backend.", 0),
    );

    render(<BookingsPanel patientId="P001" />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /could not reach the backend/i,
    );
  });

  it("renders a summary button for every booking row", async () => {
    getBookingsMock.mockResolvedValue(
      makeBookingList([
        makeBooking(),
        makeBooking({ booking_id: "B1004" }),
      ]),
    );

    render(<BookingsPanel patientId="P001" />);

    expect(
      await screen.findByRole("button", {
        name: /view summary for booking B1001/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: /view summary for booking B1004/i,
      }),
    ).toBeInTheDocument();
    expect(getBookingSummaryMock).not.toHaveBeenCalled();
  });

  it("loads and displays the four required summary fields", async () => {
    const user = userEvent.setup();
    getBookingsMock.mockResolvedValue(makeBookingList([makeBooking()]));
    getBookingSummaryMock.mockResolvedValue(makeBookingSummary());

    render(<BookingsPanel patientId="P001" />);
    await user.click(
      await screen.findByRole("button", {
        name: /view summary for booking B1001/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", {
      name: /visit summary for B1001/i,
    });
    expect(within(dialog).getByText("Chief complaint")).toBeInTheDocument();
    expect(
      within(dialog).getByText("itchy rash on the left arm"),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("Diagnosis")).toBeInTheDocument();
    expect(within(dialog).getByText("not stated")).toBeInTheDocument();
    expect(within(dialog).getByText("Medication")).toBeInTheDocument();
    expect(
      within(dialog).getByText("hydrocortisone cream"),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText("Doctor instructions"),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("keep the area clean")).toBeInTheDocument();
    expect(
      within(dialog).getByText("Clinician review recommended."),
    ).toBeInTheDocument();
    expect(getBookingSummaryMock).toHaveBeenCalledWith(
      "B1001",
      expect.any(AbortSignal),
    );
  });

  it("identifies the linked treatment history record", async () => {
    const user = userEvent.setup();
    getBookingsMock.mockResolvedValue(makeBookingList([makeBooking()]));
    getBookingSummaryMock.mockResolvedValue(makeBookingSummary());

    render(<BookingsPanel patientId="P001" />);
    await user.click(
      await screen.findByRole("button", {
        name: /view summary for booking B1001/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", {
      name: /visit summary for B1001/i,
    });
    expect(
      within(dialog).getByText(/treatment history record H9003/i),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/visit 2026-11-20/i)).toBeInTheDocument();
  });

  it("shows a loading state while the summary is fetching", async () => {
    const user = userEvent.setup();
    getBookingsMock.mockResolvedValue(makeBookingList([makeBooking()]));
    let resolveSummary:
      | ((value: ReturnType<typeof makeBookingSummary>) => void)
      | undefined;
    getBookingSummaryMock.mockReturnValue(
      new Promise((resolve) => {
        resolveSummary = resolve;
      }),
    );

    render(<BookingsPanel patientId="P001" />);
    await user.click(
      await screen.findByRole("button", {
        name: /view summary for booking B1001/i,
      }),
    );

    expect(
      await screen.findByText(/loading summary/i),
    ).toBeInTheDocument();

    resolveSummary?.(makeBookingSummary());
    expect(
      await screen.findByText("itchy rash on the left arm"),
    ).toBeInTheDocument();
  });

  it("shows a no-summary message for a booking without one", async () => {
    const user = userEvent.setup();
    getBookingsMock.mockResolvedValue(makeBookingList([makeBooking()]));
    getBookingSummaryMock.mockRejectedValue(
      new ApiError(
        "No summary exists for booking B1001.",
        404,
        "BOOKING_SUMMARY_NOT_FOUND",
      ),
    );

    render(<BookingsPanel patientId="P001" />);
    await user.click(
      await screen.findByRole("button", {
        name: /view summary for booking B1001/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", {
      name: /visit summary for B1001/i,
    });
    expect(
      await within(dialog).findByText(/no summary available/i),
    ).toBeInTheDocument();
  });

  it("surfaces a summary loading error inside the dialog", async () => {
    const user = userEvent.setup();
    getBookingsMock.mockResolvedValue(makeBookingList([makeBooking()]));
    getBookingSummaryMock.mockRejectedValue(
      new ApiError("The summary service is unavailable.", 503),
    );

    render(<BookingsPanel patientId="P001" />);
    await user.click(
      await screen.findByRole("button", {
        name: /view summary for booking B1001/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", {
      name: /visit summary for B1001/i,
    });
    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      /summary service is unavailable/i,
    );
  });

  it("closes the summary dialog", async () => {
    const user = userEvent.setup();
    getBookingsMock.mockResolvedValue(makeBookingList([makeBooking()]));
    getBookingSummaryMock.mockResolvedValue(makeBookingSummary());

    render(<BookingsPanel patientId="P001" />);
    await user.click(
      await screen.findByRole("button", {
        name: /view summary for booking B1001/i,
      }),
    );
    await screen.findByRole("dialog", {
      name: /visit summary for B1001/i,
    });

    await user.click(screen.getByRole("button", { name: /^close$/i }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
