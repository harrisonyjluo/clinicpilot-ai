import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { MessageInput } from "./MessageInput";

function makeProps(overrides: Partial<Parameters<typeof MessageInput>[0]> = {}) {
  return {
    value: "",
    onChange: vi.fn(),
    onSubmit: vi.fn(),
    disabled: false,
    ...overrides,
  };
}

describe("MessageInput", () => {
  it("sends the message when Enter is pressed", () => {
    const props = makeProps({ value: "Hello" });
    render(<MessageInput {...props} />);

    fireEvent.keyDown(screen.getByLabelText("Message"), { key: "Enter" });

    expect(props.onSubmit).toHaveBeenCalledTimes(1);
  });

  it("inserts a newline instead of sending when Shift+Enter is pressed", () => {
    const props = makeProps({ value: "Hello" });
    render(<MessageInput {...props} />);

    fireEvent.keyDown(screen.getByLabelText("Message"), {
      key: "Enter",
      shiftKey: true,
    });

    expect(props.onSubmit).not.toHaveBeenCalled();
  });

  it("does not send while an IME composition is active", () => {
    const props = makeProps({ value: "你好" });
    render(<MessageInput {...props} />);

    fireEvent.keyDown(screen.getByLabelText("Message"), {
      key: "Enter",
      keyCode: 229,
      isComposing: true,
    });

    expect(props.onSubmit).not.toHaveBeenCalled();
  });

  it("does not send a whitespace-only message", () => {
    const props = makeProps({ value: "   " });
    render(<MessageInput {...props} />);

    fireEvent.keyDown(screen.getByLabelText("Message"), { key: "Enter" });

    expect(props.onSubmit).not.toHaveBeenCalled();
  });

  it("grows the textarea with its content up to the maximum height", () => {
    const props = makeProps({ value: "short" });
    const { rerender } = render(<MessageInput {...props} />);
    const textarea = screen.getByLabelText("Message") as HTMLTextAreaElement;

    Object.defineProperty(textarea, "scrollHeight", {
      configurable: true,
      value: 120,
    });
    rerender(
      <MessageInput
        {...props}
        value={"line one\nline two\nline three"}
      />,
    );

    expect(textarea.style.height).toBe("120px");

    Object.defineProperty(textarea, "scrollHeight", {
      configurable: true,
      value: 400,
    });
    rerender(
      <MessageInput
        {...props}
        value={"one\ntwo\nthree\nfour\nfive\nsix\nseven\neight"}
      />,
    );

    expect(textarea.style.height).toBe("160px");
  });
});
