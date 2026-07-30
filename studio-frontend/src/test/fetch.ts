import { vi } from "vitest";

export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
}

export function deferred<T>(): Deferred<T> {
  let resolve!: Deferred<T>["resolve"];
  let reject!: Deferred<T>["reject"];
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

export interface PendingFetch {
  input: RequestInfo | URL;
  init?: RequestInit;
  signal?: AbortSignal;
  response: Deferred<Response>;
}

export function createFetchController() {
  const requests: PendingFetch[] = [];
  const available: PendingFetch[] = [];
  const waiters: Array<(request: PendingFetch) => void> = [];

  const fetch = vi.fn((
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const response = deferred<Response>();
    const signal = init?.signal
      ?? (input instanceof Request ? input.signal : undefined)
      ?? undefined;
    const request = { input, init, signal, response };
    requests.push(request);

    const waiter = waiters.shift();
    if (waiter) waiter(request);
    else available.push(request);

    if (signal) {
      const rejectForAbort = () => {
        response.reject(new DOMException("The operation was aborted.", "AbortError"));
      };
      if (signal.aborted) rejectForAbort();
      else {
        signal.addEventListener("abort", rejectForAbort, { once: true });
        void response.promise.then(
          () => signal.removeEventListener("abort", rejectForAbort),
          () => signal.removeEventListener("abort", rejectForAbort),
        );
      }
    }
    return response.promise;
  });

  function nextRequest(): Promise<PendingFetch> {
    const request = available.shift();
    if (request) return Promise.resolve(request);
    return new Promise((resolve) => waiters.push(resolve));
  }

  function respondJson(
    request: PendingFetch,
    body: unknown,
    init: ResponseInit = {},
  ) {
    const headers = new Headers(init.headers);
    if (!headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }
    request.response.resolve(new Response(JSON.stringify(body), {
      ...init,
      headers,
    }));
  }

  return {
    fetch,
    requests,
    nextRequest,
    respondJson,
  };
}
