import { useLayoutEffect, useRef } from "react";

import type { ChatMessage } from "../types";

interface MessageListProps {
  messages: ChatMessage[];
  isSending: boolean;
}

export function MessageList({ messages, isSending }: MessageListProps) {
  const logRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const log = logRef.current;
    if (!log) {
      return;
    }
    log.scrollTop = log.scrollHeight;
  }, [messages, isSending]);

  return (
    <div
      ref={logRef}
      className="messages"
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
    >
      {messages.map((message) => (
        <article
          key={message.id}
          className={`message message-${message.role}`}
        >
          <span className="message-role">
            {message.role === "user" ? "You" : "ClinicPilot"}
          </span>
          <p>{message.text}</p>
        </article>
      ))}
      {isSending ? (
        <p className="typing" role="status">
          ClinicPilot is typing…
        </p>
      ) : null}
    </div>
  );
}
