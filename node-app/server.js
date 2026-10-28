import express from "express";
import dotenv from "dotenv";
import { pathToFileURL } from "node:url";

dotenv.config();

const DEFAULT_TIMEOUT_MS = Number.parseInt(process.env.RAG_TIMEOUT_MS || "30000", 10);
const MAX_QUESTION_LENGTH = 4000;

export function createApp({
  fetchImpl = fetch,
  ragUrl = process.env.PYTHON_RAG_URL || "http://localhost:8000",
  timeoutMs = DEFAULT_TIMEOUT_MS,
} = {}) {
  const app = express();
  app.use(express.json({ limit: "32kb" }));

  app.get("/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  app.post("/api/chat", async (req, res) => {
    const question = extractQuestion(req.body);
    if (!question) {
      return res.status(400).json({
        error: "A non-empty 'question' or 'message' string is required.",
      });
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      return res.status(400).json({
        error: `Question must be at most ${MAX_QUESTION_LENGTH} characters.`,
      });
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetchImpl(`${ragUrl.replace(/\/$/, "")}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      const payload = await readJson(response);
      if (!response.ok) {
        const detail = payload?.detail || payload?.error || "RAG service failed";
        const status = response.status >= 400 && response.status < 600 ? response.status : 502;
        return res.status(status).json({ error: detail });
      }

      return res.json({
        answer: payload.answer,
        sources: payload.sources || [],
      });
    } catch (error) {
      if (error?.name === "AbortError") {
        return res.status(504).json({ error: "Timed out waiting for the RAG service." });
      }
      return res.status(502).json({ error: "Unable to reach the RAG service." });
    } finally {
      clearTimeout(timer);
    }
  });

  return app;
}

function extractQuestion(body) {
  if (!body || typeof body !== "object") {
    return null;
  }
  const value = body.question ?? body.message;
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length ? trimmed : null;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

const isMainModule = import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMainModule) {
  const port = Number.parseInt(process.env.NODE_PORT || process.env.PORT || "3001", 10);
  const app = createApp();
  app.listen(port, () => {
    console.log(`Node RAG gateway listening on ${port}`);
  });
}

export default createApp;
