import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ChatMessage } from "../types";
import { MessageList } from "./MessageList";

const welcome: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text: "Hello",
};

describe("MessageList", () => {
  it("scrolls to the newest message whenever the conversation changes", () => {
    const { rerender } = render(
      <MessageList messages={[welcome]} isSending={false} />,
    );
    const log = screen.getByRole("log");

    Object.defineProperty(log, "scrollHeight", {
      configurable: true,
      value: 240,
    });
    rerender(
      <MessageList
        messages={[
          welcome,
          { id: "reply", role: "assistant", text: "How can I help?" },
        ]}
        isSending={false}
      />,
    );

    expect(log.scrollTop).toBe(240);
  });

  it("scrolls again when the typing indicator appears", () => {
    const messages: ChatMessage[] = [
      welcome,
      { id: "user", role: "user", text: "Hello" },
    ];
    const { rerender } = render(
      <MessageList messages={messages} isSending={false} />,
    );
    const log = screen.getByRole("log");

    Object.defineProperty(log, "scrollHeight", {
      configurable: true,
      value: 360,
    });
    rerender(<MessageList messages={messages} isSending />);

    expect(log.scrollTop).toBe(360);
  });
});
