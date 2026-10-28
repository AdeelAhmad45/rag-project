import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import test from "node:test";
import { createApp } from "./server.js";

function listen(app) {
  const server = createServer(app);
  server.listen(0, "127.0.0.1");
  return once(server, "listening").then(() => {
    const { port } = server.address();
    return { server, url: `http://127.0.0.1:${port}` };
  });
}

test("POST /api/chat validates missing question", async () => {
  const { server, url } = await listen(createApp({ fetchImpl: async () => {
    throw new Error("should not be called");
  } }));
  try {
    const response = await fetch(`${url}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(response.status, 400);
  } finally {
    server.close();
  }
});

test("POST /api/chat returns answer and sources", async () => {
  const fetchImpl = async () =>
    new Response(
      JSON.stringify({
        answer: "Seattle, Washington",
        sources: [{ source: "handbook.txt", page: null, snippet: "headquarters" }],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );

  const { server, url } = await listen(createApp({ fetchImpl, ragUrl: "http://python-rag:8000" }));
  try {
    const response = await fetch(`${url}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "Where is headquarters?" }),
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.answer, "Seattle, Washington");
    assert.equal(body.sources[0].source, "handbook.txt");
  } finally {
    server.close();
  }
});

test("POST /api/chat maps timeouts to 504", async () => {
  const fetchImpl = async (_url, options) => {
    await new Promise((_, reject) => {
      options.signal.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      });
    });
  };

  const { server, url } = await listen(createApp({ fetchImpl, timeoutMs: 20 }));
  try {
    const response = await fetch(`${url}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "What is Atlas Insights?" }),
    });
    assert.equal(response.status, 504);
  } finally {
    server.close();
  }
});

test("GET /health returns ok", async () => {
  const { server, url } = await listen(createApp());
  try {
    const response = await fetch(`${url}/health`);
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.status, "ok");
  } finally {
    server.close();
  }
});
