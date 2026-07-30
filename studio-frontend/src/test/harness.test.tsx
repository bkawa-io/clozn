import { useEffect, useState } from "react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { render, screen } from "./render";
import { createFetchController } from "./fetch";

function DeferredGreeting() {
  const [started, setStarted] = useState(false);
  const [message, setMessage] = useState("NOT STARTED");

  useEffect(() => {
    if (!started) return;
    const controller = new AbortController();
    setMessage("LOADING");
    void fetch("/greeting", { signal: controller.signal })
      .then((response) => response.json())
      .then((body: { message?: unknown }) => {
        setMessage(typeof body.message === "string" ? body.message : "INVALID");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setMessage("FAILED");
      });
    return () => controller.abort();
  }, [started]);

  return (
    <section>
      <button type="button" onClick={() => setStarted(true)}>LOAD GREETING</button>
      <output aria-live="polite">{message}</output>
    </section>
  );
}

test("runs component effects against a deterministically deferred fetch", async () => {
  const transport = createFetchController();
  vi.stubGlobal("fetch", transport.fetch);
  const user = userEvent.setup();

  render(<DeferredGreeting />);
  expect(screen.getByText("NOT STARTED")).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "LOAD GREETING" }));
  const request = await transport.nextRequest();
  expect(request.input).toBe("/greeting");
  expect(request.signal?.aborted).toBe(false);
  expect(screen.getByText("LOADING")).toBeInTheDocument();

  transport.respondJson(request, { message: "HELLO FROM TEST" });
  expect(await screen.findByText("HELLO FROM TEST")).toBeInTheDocument();
});

test("cleans up an in-flight component effect on unmount", async () => {
  const transport = createFetchController();
  vi.stubGlobal("fetch", transport.fetch);
  const user = userEvent.setup();

  const view = render(<DeferredGreeting />);
  await user.click(screen.getByRole("button", { name: "LOAD GREETING" }));
  const request = await transport.nextRequest();
  const rejection = expect(request.response.promise).rejects.toMatchObject({
    name: "AbortError",
  });

  view.unmount();

  expect(request.signal?.aborted).toBe(true);
  expect(screen.queryByRole("button", { name: "LOAD GREETING" })).not.toBeInTheDocument();
  await rejection;
});
