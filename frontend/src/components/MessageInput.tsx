import { useLayoutEffect, useRef, type KeyboardEvent } from "react";

const MAX_INPUT_HEIGHT_PX = 160;

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
}

export function MessageInput({
  value,
  onChange,
  onSubmit,
  disabled,
}: MessageInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(
      textarea.scrollHeight,
      MAX_INPUT_HEIGHT_PX,
    )}px`;
  }, [value]);

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }
    // Enter is part of an IME composition (for example while typing Chinese);
    // it must confirm the composition, not send the message.
    if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) {
      return;
    }
    event.preventDefault();
    if (!disabled && value.trim()) {
      onSubmit();
    }
  }

  return (
    <form
      className="message-input"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <label htmlFor="chat-message">Message</label>
      <textarea
        ref={textareaRef}
        id="chat-message"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about clinics, treatments, or bookings…"
        rows={1}
        maxLength={2000}
        disabled={disabled}
      />
      <div className="message-input-actions">
        <span className="message-input-hint">
          Enter to send · Shift+Enter for a new line
        </span>
        <button type="submit" disabled={disabled || !value.trim()}>
          Send
        </button>
      </div>
    </form>
  );
}
