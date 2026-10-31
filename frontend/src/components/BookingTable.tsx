import type { Booking } from "../types";

interface BookingTableProps {
  bookings: Booking[];
  patientId: string;
  onViewSummary: (booking: Booking) => void;
}

function formatTime(value: string): string {
  return value.slice(0, 5);
}

function formatPrice(value: string | number): string {
  const amount = Number(value);
  return Number.isFinite(amount) ? `$${amount.toFixed(2)}` : String(value);
}

export function BookingTable({
  bookings,
  patientId,
  onViewSummary,
}: BookingTableProps) {
  return (
    <div
      className="table-wrap"
      role="region"
      aria-label="Bookings table"
      tabIndex={0}
    >
      <table>
        <caption>Appointments for {patientId}</caption>
        <thead>
          <tr>
            <th scope="col">Booking</th>
            <th scope="col">Doctor</th>
            <th scope="col">Treatment</th>
            <th scope="col">Date</th>
            <th scope="col">Time</th>
            <th scope="col">Status</th>
            <th scope="col">Notes</th>
            <th scope="col">Summary</th>
          </tr>
        </thead>
        <tbody>
          {bookings.map((booking) => (
            <tr key={booking.booking_id}>
              <td>{booking.booking_id}</td>
              <td>
                <strong>{booking.doctor_name}</strong>
                <span className="cell-detail">{booking.specialty}</span>
              </td>
              <td>
                <strong>{booking.treatment_name}</strong>
                <span className="cell-detail">
                  {booking.duration_minutes} min ·{" "}
                  {formatPrice(booking.price_usd)}
                </span>
              </td>
              <td>{booking.booking_date}</td>
              <td>{formatTime(booking.booking_time)}</td>
              <td>
                <span className="status">{booking.status}</span>
              </td>
              <td>{booking.notes ?? "—"}</td>
              <td>
                <button
                  type="button"
                  className="summary-button"
                  onClick={() => onViewSummary(booking)}
                  aria-label={`View summary for booking ${booking.booking_id}`}
                >
                  Summary
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
