import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  getBookingSummary,
  getBookings,
  isAbortError,
} from "../api/client";
import type { Booking, BookingSummary } from "../types";
import { BookingSummaryDialog } from "./BookingSummaryDialog";
import type { BookingSummaryStatus } from "./BookingSummaryDialog";
import { BookingTable } from "./BookingTable";

interface BookingsPanelProps {
  patientId: string;
  refreshToken?: number;
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Could not load bookings. Please try again.";
}

export function BookingsPanel({
  patientId,
  refreshToken = 0,
}: BookingsPanelProps) {
  const [bookings, setBookings] = useState<Booking[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [summaryBooking, setSummaryBooking] = useState<Booking | null>(null);
  const [summary, setSummary] = useState<BookingSummary | null>(null);
  const [summaryStatus, setSummaryStatus] =
    useState<BookingSummaryStatus>("loading");
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const summaryAbortRef = useRef<AbortController | null>(null);

  const load = useCallback(
    async (signal: AbortSignal) => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await getBookings(patientId, signal);
        if (signal.aborted) {
          return;
        }
        setBookings(response.bookings);
        setLastUpdated(new Date());
      } catch (caught) {
        if (isAbortError(caught) || signal.aborted) {
          return;
        }
        setError(errorMessage(caught));
      } finally {
        if (!signal.aborted) {
          setIsLoading(false);
        }
      }
    },
    [patientId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load, refreshToken, reloadKey]);

  useEffect(() => {
    return () => summaryAbortRef.current?.abort();
  }, []);

  const closeSummary = useCallback(() => {
    summaryAbortRef.current?.abort();
    summaryAbortRef.current = null;
    setSummaryBooking(null);
    setSummary(null);
    setSummaryError(null);
  }, []);

  const openSummary = useCallback(async (booking: Booking) => {
    summaryAbortRef.current?.abort();
    const controller = new AbortController();
    summaryAbortRef.current = controller;

    setSummaryBooking(booking);
    setSummary(null);
    setSummaryError(null);
    setSummaryStatus("loading");

    try {
      const result = await getBookingSummary(
        booking.booking_id,
        controller.signal,
      );
      if (controller.signal.aborted) {
        return;
      }
      setSummary(result);
      setSummaryStatus("ready");
    } catch (caught) {
      if (isAbortError(caught) || controller.signal.aborted) {
        return;
      }
      if (
        caught instanceof ApiError &&
        caught.code === "BOOKING_SUMMARY_NOT_FOUND"
      ) {
        setSummaryStatus("missing");
        return;
      }
      setSummaryError(errorMessage(caught));
      setSummaryStatus("error");
    }
  }, []);

  const hasBookings = bookings.length > 0;

  return (
    <section className="panel bookings-panel" aria-labelledby="bookings-heading">
      <header className="panel-header">
        <div>
          <h2 id="bookings-heading">Bookings</h2>
          <p className="panel-subtitle">
            Patient {patientId} · your appointments
          </p>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={() => setReloadKey((current) => current + 1)}
          disabled={isLoading}
        >
          Refresh
        </button>
      </header>

      <div className="bookings-meta">
        <span>{bookings.length} booking(s)</span>
        <span>
          {lastUpdated
            ? `Last updated ${lastUpdated.toLocaleTimeString()}`
            : "Not loaded yet"}
        </span>
      </div>

      {isLoading && !lastUpdated ? (
        <p className="muted" role="status">
          Loading bookings…
        </p>
      ) : null}

      {error ? (
        <div className="error-banner" role="alert">
          {error}
        </div>
      ) : null}

      {hasBookings ? (
        <BookingTable
          bookings={bookings}
          patientId={patientId}
          onViewSummary={openSummary}
        />
      ) : null}

      {!isLoading && !error && !hasBookings ? (
        <p className="empty-state">No bookings found.</p>
      ) : null}

      <BookingSummaryDialog
        booking={summaryBooking}
        status={summaryStatus}
        summary={summary}
        error={summaryError}
        onClose={closeSummary}
      />
    </section>
  );
}
