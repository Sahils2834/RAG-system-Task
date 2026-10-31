
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const EXAMPLES = [
  "What is retrieval-augmented generation?",
  "How do embeddings help semantic search?",
  "How does ChromaDB retrieve documents?",
];

const DOCUMENTS = [
  "RAG Research Paper",
  "Gemini API Guide",
  "Embeddings Guide",
  "ChromaDB Guide",
];

function getFileName(path) {
  return String(path || "Unknown source")
    .split(/[\\/]/)
    .pop();
}

function formatLatency(seconds) {
  if (!Number.isFinite(seconds)) return "";
  return seconds < 1
    ? `${Math.round(seconds * 1000)} ms`
    : `${seconds.toFixed(2)} seconds`;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, loading]);

  async function sendQuestion(value = question) {
    const text = value.trim();

    if (!text || loading) return;

    const history = messages
      .filter(
        (message) =>
          message.role === "user" ||
          message.role === "assistant"
      )
      .map(({ role, content }) => ({ role, content }))
      .slice(-10);

    setQuestion("");
    setError("");
    setLoading(true);

    setMessages((previous) => [
      ...previous,
      { role: "user", content: text },
    ]);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: text,
          history,
        }),
      });

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The server returned an invalid response. Check the backend terminal."
        );
      }

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Request failed (${response.status})`
        );
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            data.answer || "The model returned an empty response.",
          sources: Array.isArray(data.sources)
            ? data.sources
            : [],
          latency: data.latency_seconds,
        },
      ]);
    } catch (err) {
      setError(
        err.message ||
        "Could not connect to the API. Check that the backend is running."
      );
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendQuestion();
  }

  function startNewConversation() {
    if (loading) return;

    setMessages([]);
    setQuestion("");
    setError("");
    textareaRef.current?.focus();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">D</div>

          <div className="brand-copy">
            <strong>DocuMind</strong>
            <span>DOCUMENT INTELLIGENCE</span>
          </div>
        </div>

        <button
          className="new-chat"
          type="button"
          onClick={startNewConversation}
          disabled={loading}
        >
          <span className="new-chat-icon">＋</span>
          New conversation
        </button>

        <div className="sidebar-label">YOUR KNOWLEDGE BASE</div>

        <div className="document-list">
          {DOCUMENTS.map((document) => (
            <div className="document-item" key={document}>
              <span className="file-icon">PDF</span>
              <span>{document}</span>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <span className="status-dot" />
          <span>Local document index</span>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              YOUR PERSONAL RESEARCH ASSISTANT
            </div>
            <h1>Document Chat</h1>
          </div>

          <div className="model-badge">
            <span className="status-dot" />
            RAG + Gemini
          </div>
        </header>

        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">✳</div>

              <div className="eyebrow">WELCOME TO DOCUMIND</div>

              <h2>
                Your documents.
                <br />
                Clearer answers.
              </h2>

              <p className="welcome-description">
                Ask questions about your indexed PDFs. DocuMind
                retrieves relevant passages and uses them to generate
                answers with source references.
              </p>

              <div className="example-grid">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    className="example-card"
                    type="button"
                    onClick={() => sendQuestion(example)}
                    disabled={loading}
                  >
                    <span>{example}</span>
                    <span className="example-arrow">↗</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message, index) => (
                <article
                  key={`${index}-${message.role}`}
                  className={`message ${message.role}`}
                >
                  <div className="avatar">
                    {message.role === "user" ? "Y" : "D"}
                  </div>

                  <div className="message-body">
                    <div className="message-label">
                      {message.role === "user"
                        ? "YOU"
                        : "DOCUMIND"}
                    </div>

                    <div className="message-text">
                      {message.role === "assistant" ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      ) : (
                        message.content
                      )}
                    </div>

                    {message.sources?.length > 0 && (
                      <div className="sources">
                        <div className="sources-heading">
                          <span>SOURCES USED</span>
                          <span className="source-count">
                            {message.sources.length}
                          </span>
                        </div>

                        <div className="source-list">
                          {message.sources.map((source) => (
                            <details
                              className="source-card"
                              key={source.id}
                            >
                              <summary className="source-summary">
                                <div className="source-title">
                                  <span className="source-tag">
                                    {source.id}
                                  </span>

                                  <div className="source-file-info">
                                    <strong>
                                      {getFileName(source.source)}
                                    </strong>
                                    <span>
                                      Page {source.page ?? "unknown"}
                                    </span>
                                  </div>
                                </div>

                                <span
                                  className="source-chevron"
                                  aria-hidden="true"
                                >
                                  ⌄
                                </span>
                              </summary>

                              <div className="source-content">
                                <p>{source.text}</p>
                              </div>
                            </details>
                          ))}
                        </div>
                      </div>
                    )}

                    {message.role === "assistant" &&
                      Number.isFinite(message.latency) && (
                        <div className="latency">
                          <span className="latency-dot" />
                          Answer generated in{" "}
                          {formatLatency(message.latency)}
                        </div>
                      )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="loading-state">
                  <div className="avatar">D</div>

                  <div className="loading-content">
                    <div className="message-label">DOCUMIND</div>

                    <div className="loading-indicator">
                      <div className="typing">
                        <span />
                        <span />
                        <span />
                      </div>
                      <span>Searching your documents...</span>
                    </div>

                    <p>
                      Retrieving relevant passages and preparing an answer.
                    </p>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <div className="error-icon">!</div>

              <div className="error-content">
                <strong>Unable to generate an answer</strong>
                <p>{error}</p>

                <button
                  type="button"
                  onClick={() => {
                    setError("");
                    textareaRef.current?.focus();
                  }}
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </section>

        <footer className="composer-area">
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask something about your documents..."
              rows={2}
              maxLength={4000}
              disabled={loading}
              aria-label="Question about your documents"
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault();
                  handleSubmit(event);
                }
              }}
            />

            <button
              className="send-button"
              type="submit"
              disabled={loading || !question.trim()}
              aria-label="Send question"
              title="Send message"
            >
              {loading ? (
                <span className="send-spinner" />
              ) : (
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M12 19V5M5 12l7-7 7 7"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          </form>

          <p className="composer-note">
            AI-generated answers may contain errors. Verify important
            claims against the original documents.
          </p>
        </footer>
      </main>
    </div>
  );
}
