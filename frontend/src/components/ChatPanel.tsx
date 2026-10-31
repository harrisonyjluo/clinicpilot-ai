import { useEffect, useRef, useState } from "react";

import { isAbortError, streamChat } from "../api/client";
import type { ChatMessage, ChatRequest } from "../types";
import { MessageInput } from "./MessageInput";
import { MessageList } from "./MessageList";

interface ChatPanelProps {
  patientId: string;
  onBookingCreated?: () => void;
}

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text:
    "Hi — I can answer clinic questions, recommend treatments, check " +
    "your bookings, and book an appointment. I provide general information, " +
    "not medical advice.",
};

let nextMessageId = 0;

function createMessage(role: ChatMessage["role"], text: string): ChatMessage {
  nextMessageId += 1;
  return { id: `message-${nextMessageId}`, role, text };
}

function formatTime(value: string): string {
  return value.slice(0, 5);
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
}

export function ChatPanel({ patientId, onBookingCreated }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const streamedMessageRef = useRef<string | null>(null);

  useEffect(() => {
    return () => controllerRef.current?.abort();
  }, []);

  async function handleSend() {
    const message = input.trim();
    if (!message || isSending) {
      return;
    }

    const controller = new AbortController();
    controllerRef.current?.abort();
    controllerRef.current = controller;

    setMessages((current) => [
      ...current,
      createMessage("user", message),
    ]);
    setInput("");
    setError(null);
    setIsSending(true);

    const payload: ChatRequest = {
      message,
      patient_id: patientId,
    };
    if (conversationId) {
      payload.conversation_id = conversationId;
    }

    streamedMessageRef.current = null;

    const handleDelta = (delta: string) => {
      if (!delta) {
        return;
      }
      if (streamedMessageRef.current === null) {
        streamedMessageRef.current = createMessage("assistant", "").id;
      }
      const id = streamedMessageRef.current;
      setMessages((current) => {
        if (!current.some((entry) => entry.id === id)) {
          return [...current, { id, role: "assistant", text: delta }];
        }
        return current.map((entry) =>
          entry.id === id ? { ...entry, text: entry.text + delta } : entry,
        );
      });
    };

    try {
      const response = await streamChat(
        payload,
        handleDelta,
        controller.signal,
      );
      setConversationId(response.conversation_id);

      // The terminal event is authoritative; replace any partial render.
      const replyId =
        streamedMessageRef.current ?? createMessage("assistant", "").id;
      setMessages((current) => {
        if (!current.some((entry) => entry.id === replyId)) {
          return [
            ...current,
            { id: replyId, role: "assistant", text: response.reply },
          ];
        }
        return current.map((entry) =>
          entry.id === replyId
            ? { ...entry, text: response.reply }
            : entry,
        );
      });

      if (response.booking) {
        const booking = response.booking;
        setMessages((current) => [
          ...current,
          createMessage(
            "assistant",
            `Booking confirmed: ${booking.booking_id} — ` +
              `${booking.treatment_name} with ${booking.doctor_name} on ` +
              `${booking.booking_date} at ${formatTime(booking.booking_time)}.`,
          ),
        ]);
        onBookingCreated?.();
      }
    } catch (caught) {
      if (isAbortError(caught)) {
        return;
      }
      setError(errorMessage(caught));
      setInput(message);
    } finally {
      streamedMessageRef.current = null;
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setIsSending(false);
      }
    }
  }

  return (
    <section className="panel chat-panel" aria-labelledby="chat-heading">
      <header className="panel-header">
        <div>
          <h2 id="chat-heading">Chat</h2>
          <p className="panel-subtitle">Patient {patientId}</p>
        </div>
        <span className="pill">AI assistant</span>
      </header>

      <p className="disclaimer">
        Informational demo only — not medical advice. For emergencies, call
        911 or go to the nearest emergency room.
      </p>

      <MessageList messages={messages} isSending={isSending} />

      {error ? (
        <div className="error-banner" role="alert">
          {error}
        </div>
      ) : null}

      <MessageInput
        value={input}
        onChange={setInput}
        onSubmit={() => {
          void handleSend();
        }}
        disabled={isSending}
      />
    </section>
  );
}
