import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openrouter import OpenRouter


HOST = "127.0.0.1"
PORT = 8001
MODEL = "openai/gpt-5.4"
DOCX_PATH = "MAE301 Project Guideline v1.1.docx"
AGENT_PATH = "AGENT_STEP7.md"
SKILL_PATH = "SKILL_STEP7.md"

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAE301 Guideline Assistant</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef3f1;
      --panel: #fbfdfc;
      --ink: #182126;
      --muted: #50616a;
      --line: #cdd8d3;
      --accent: #0a5c74;
      --accent-2: #b06a2b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #d8ebe5 0, transparent 26%),
        radial-gradient(circle at bottom right, #f0e2d1 0, transparent 28%),
        var(--bg);
      color: var(--ink);
    }
    main {
      max-width: 760px;
      margin: 48px auto;
      padding: 0 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 18px 50px rgba(20, 31, 38, 0.08);
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 5vw, 3.2rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    p {
      margin: 0 0 20px;
      color: var(--muted);
      line-height: 1.5;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      resize: vertical;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 14px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
    }
    .primary {
      background: var(--accent);
      color: #fff;
    }
    .secondary {
      background: var(--accent-2);
      color: #fff;
    }
    .ghost {
      background: #e7eeeb;
      color: var(--ink);
    }
    #status {
      margin-top: 14px;
      color: var(--muted);
      min-height: 24px;
    }
    pre {
      margin: 16px 0 0;
      padding: 16px;
      border-radius: 14px;
      background: #1d2328;
      color: #f7f7f2;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 180px;
      overflow-x: auto;
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>MAE301 Guide</h1>
      <p>Ask by voice or text about deliverables, dates, tools, or repo structure.</p>
      <textarea id="prompt" placeholder="Try: when is phase 2 due?"></textarea>
      <div class="row">
        <button class="secondary" id="voiceBtn" type="button">Start Voice Input</button>
        <button class="primary" id="sendBtn" type="button">Ask</button>
        <button class="ghost" id="clearBtn" type="button">Clear</button>
      </div>
      <div id="status">Idle.</div>
      <pre id="output"></pre>
    </section>
  </main>
  <script>
    const promptEl = document.getElementById("prompt");
    const outputEl = document.getElementById("output");
    const statusEl = document.getElementById("status");
    const voiceBtn = document.getElementById("voiceBtn");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let listening = false;

    if (SpeechRecognition) {
      recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = true;
      recognition.continuous = false;

      recognition.onstart = () => {
        listening = true;
        voiceBtn.textContent = "Listening...";
        statusEl.textContent = "Listening for speech...";
      };

      recognition.onresult = (event) => {
        let transcript = "";
        for (const result of event.results) {
          transcript += result[0].transcript;
        }
        promptEl.value = transcript.trim();
      };

      recognition.onerror = (event) => {
        statusEl.textContent = "Voice error: " + event.error;
      };

      recognition.onend = () => {
        listening = false;
        voiceBtn.textContent = "Start Voice Input";
        statusEl.textContent = promptEl.value.trim() ? "Voice captured." : "Idle.";
      };
    } else {
      voiceBtn.disabled = true;
      statusEl.textContent = "Voice input is not supported in this browser. Use Chrome or Edge.";
    }

    voiceBtn.addEventListener("click", () => {
      if (!recognition) return;
      if (listening) {
        recognition.stop();
        return;
      }
      promptEl.value = "";
      recognition.start();
    });

    clearBtn.addEventListener("click", () => {
      promptEl.value = "";
      outputEl.textContent = "";
      statusEl.textContent = "Idle.";
    });

    sendBtn.addEventListener("click", async () => {
      const message = promptEl.value.trim();
      if (!message) {
        statusEl.textContent = "Enter or speak a question first.";
        return;
      }

      outputEl.textContent = "";
      statusEl.textContent = "Checking guideline...";

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Request failed.");
        }

        outputEl.textContent = data.output || "";
        statusEl.textContent = "Done.";
      } catch (error) {
        statusEl.textContent = "Error.";
        outputEl.textContent = String(error);
      }
    });
  </script>
</body>
</html>
"""


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        with open("openrouter.md", "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return ""


def load_docx_paragraphs(path: str) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    for node in root.findall(".//w:p", ns):
        parts = [text.text for text in node.findall(".//w:t", ns) if text.text]
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)

    return paragraphs


def chunk_paragraphs(paragraphs: list[str], max_chars: int = 1200) -> list[str]:
    chunks = []
    current = []
    current_len = 0

    for paragraph in paragraphs:
        extra = len(paragraph) + 1
        if current and current_len + extra > max_chars:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += extra

    if current:
        chunks.append("\n".join(current))

    return chunks


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def select_relevant_chunks(chunks: list[str], question: str, limit: int = 3) -> list[str]:
    query_terms = tokenize(question)
    scored = []

    for index, chunk in enumerate(chunks):
        chunk_terms = tokenize(chunk)
        overlap = len(query_terms & chunk_terms)
        scored.append((overlap, -index, chunk))

    scored.sort(reverse=True)
    selected = [chunk for score, _, chunk in scored if score > 0][:limit]

    if selected:
        return selected

    return chunks[:limit]


def answer_question(question: str) -> str:
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing and openrouter.md was not found.")

    agent_md = load_text(AGENT_PATH)
    skill_md = load_text(SKILL_PATH)
    paragraphs = load_docx_paragraphs(DOCX_PATH)
    chunks = chunk_paragraphs(paragraphs)
    context = "\n\n---\n\n".join(select_relevant_chunks(chunks, question))

    messages = [
        {
            "role": "system",
            "content": f"{agent_md}\n\n{skill_md}\n\nGuideline context:\n{context}",
        },
        {"role": "user", "content": question},
    ]

    with OpenRouter(api_key=api_key) as client:
        response = client.chat.send(model=MODEL, messages=messages)
        return response.choices[0].message.content.strip()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return

        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/ask":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)

        try:
            data = json.loads(raw_body.decode("utf-8"))
            message = str(data.get("message", "")).strip()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON."})
            return

        if not message:
            self._send_json(400, {"error": "Message is required."})
            return

        try:
            output = answer_question(message)
            self._send_json(200, {"output": output})
        except Exception as error:
            self._send_json(500, {"error": str(error)})

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving http://{HOST}:{PORT}")
    server.serve_forever()
