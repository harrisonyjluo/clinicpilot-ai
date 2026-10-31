import { useState } from "react";

import { BookingsPanel } from "./components/BookingsPanel";
import { ChatPanel } from "./components/ChatPanel";

interface AppProps {
  patientId: string;
}

type AppTab = "chat" | "bookings";

const TABS: ReadonlyArray<{ id: AppTab; label: string }> = [
  { id: "chat", label: "Chat" },
  { id: "bookings", label: "Bookings" },
];

export function App({ patientId }: AppProps) {
  const [refreshToken, setRefreshToken] = useState(0);
  const [activeTab, setActiveTab] = useState<AppTab>("chat");

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">Clinic assistant</p>
          <h1>ClinicPilot AI</h1>
        </div>
        <p className="tagline">Ask, plan, and book in one place.</p>
      </header>

      <div className="tab-list" role="tablist" aria-label="ClinicPilot views">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            className="tab-button"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <main className="layout">
        <section
          className="tab-panel"
          role="tabpanel"
          id="panel-chat"
          aria-labelledby="tab-chat"
          hidden={activeTab !== "chat"}
        >
          <ChatPanel
            patientId={patientId}
            onBookingCreated={() => {
              setRefreshToken((current) => current + 1);
            }}
          />
        </section>

        <section
          className="tab-panel"
          role="tabpanel"
          id="panel-bookings"
          aria-labelledby="tab-bookings"
          hidden={activeTab !== "bookings"}
        >
          <BookingsPanel patientId={patientId} refreshToken={refreshToken} />
        </section>
      </main>

      <footer className="app-footer">
        Answers are grounded in the clinic's services and policies. This is
        general information, not medical advice.
      </footer>
    </div>
  );
}
