import React from "react";
import ReactDOM from "react-dom/client";

import { App } from "./App";
import "./styles.css";

const patientId = import.meta.env.VITE_PATIENT_ID?.trim();

function MissingConfiguration() {
  return (
    <main className="app">
      <h1>ClinicPilot AI</h1>
      <div className="error-banner" role="alert">
        VITE_PATIENT_ID is not configured. Copy <code>.env.example</code> to{" "}
        <code>.env</code> in <code>frontend/</code> and set the patient to load.
      </div>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {patientId ? <App patientId={patientId} /> : <MissingConfiguration />}
  </React.StrictMode>,
);
