import { useEffect, useRef } from "react";

import type { Booking, BookingSummary } from "../types";

export type BookingSummaryStatus =
  | "loading"
  | "ready"
  | "missing"
  | "error";

interface BookingSummaryDialogProps {
  booking: Booking | null;
  status: BookingSummaryStatus;
  summary: BookingSummary | null;
  error: string | null;
  onClose: () => void;
}

function SummaryList({
  title,
  items,
  emptyText,
}: {
  title: string;
  items: string[];
  emptyText: string;
}) {
  return (
    <section className="summary-section">
      <h3>{title}</h3>
      {items.length > 0 ? (
        <ul>
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="muted">{emptyText}</p>
      )}
    </section>
  );
}

export function BookingSummaryDialog({
  booking,
  status,
  summary,
  error,
  onClose,
}: BookingSummaryDialogProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!booking) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    closeRef.current?.focus();
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [booking, onClose]);

  if (!booking) {
    return null;
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        className="summary-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="summary-dialog-title"
      >
        <header className="summary-dialog-header">
          <div>
            <h2 id="summary-dialog-title">
              Visit summary for {booking.booking_id}
            </h2>
            <p className="panel-subtitle">
              {booking.treatment_name} with {booking.doctor_name} on{" "}
              {booking.booking_date}
            </p>
            {status === "ready" && summary ? (
              <p className="summary-record">
                Treatment history record {summary.record_id} · visit{" "}
                {summary.visit_date}
              </p>
            ) : null}
          </div>
          <button
            ref={closeRef}
            type="button"
            className="secondary-button"
            onClick={onClose}
          >
            Close
          </button>
        </header>

        {status === "loading" ? (
          <p className="muted" role="status">
            Loading summary…
          </p>
        ) : null}

        {status === "missing" ? (
          <p className="empty-state">
            No summary available for this booking. Summaries are generated
            automatically when an appointment is booked through chat.
          </p>
        ) : null}

        {status === "error" ? (
          <div className="error-banner" role="alert">
            {error}
          </div>
        ) : null}

        {status === "ready" && summary ? (
          <div className="summary-content">
            <dl className="summary-grid">
              <div>
                <dt>Chief complaint</dt>
                <dd>{summary.chief_complaint}</dd>
              </div>
              <div>
                <dt>Diagnosis</dt>
                <dd>{summary.diagnosis}</dd>
              </div>
            </dl>

            <SummaryList
              title="Medication"
              items={summary.medication}
              emptyText="None recorded."
            />
            <SummaryList
              title="Doctor instructions"
              items={summary.doctor_instructions}
              emptyText="None recorded."
            />

            {summary.follow_up ? (
              <section className="summary-section">
                <h3>Follow-up</h3>
                <p>{summary.follow_up}</p>
              </section>
            ) : null}

            {summary.red_flags.length > 0 ? (
              <section className="summary-section summary-flags">
                <h3>Red flags</h3>
                <ul>
                  {summary.red_flags.map((flag) => (
                    <li key={flag}>{flag}</li>
                  ))}
                </ul>
              </section>
            ) : null}

            {summary.needs_review ? (
              <p className="summary-review">Clinician review recommended.</p>
            ) : null}

            <p className="summary-disclaimer">
              AI-generated summary. It requires clinician review and is not
              medical advice.
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}
