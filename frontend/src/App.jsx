
import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const EXAMPLES = [
  "What is retrieval-augmented generation?",
  "How do embeddings help semantic search?",
  "How does ChromaDB retrieve documents?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function sendQuestion(value = question) {
    const text = value.trim();

    if (!text || loading) return;

    const history = messages
      .filter((message) => message.role === "user" ||
        message.role === "assistant")
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          history,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || `Request failed (${response.status})`
        );
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
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
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendQuestion();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">D</div>
          <div>
            <strong>DocuMind</strong>
            <span>DOCUMENT INTELLIGENCE</span>
          </div>
        </div>

        <button
          className="new-chat"
          onClick={() => {
            setMessages([]);
            setQuestion("");
            setError("");
          }}
        >
          <span>＋</span> New conversation
        </button>

        <div className="sidebar-label">YOUR KNOWLEDGE BASE</div>

        <div className="document-item">
          <span className="file-icon">PDF</span>
          <span>RAG Research Paper</span>
        </div>
        <div className="document-item">
          <span className="file-icon">PDF</span>
          <span>Gemini API Guide</span>
        </div>
        <div className="document-item">
          <span className="file-icon">PDF</span>
          <span>Embeddings Guide</span>
        </div>
        <div className="document-item">
          <span className="file-icon">PDF</span>
          <span>ChromaDB Guide</span>
        </div>

        <div className="sidebar-footer">
          <span className="status-dot" />
          Local document index
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <div className="eyebrow">YOUR PERSONAL RESEARCH ASSISTANT</div>
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
              <h2>Your documents.<br />Clearer answers.</h2>
              <p>
                Ask questions about your indexed PDFs. Answers use
                retrieved document passages and include source references.
              </p>

              <div className="example-grid">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    className="example-card"
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
                      {message.role === "user" ? "YOU" : "DOCUMIND"}
                    </div>
                    <div className="message-text">
                      {message.content}
                    </div>

                    {message.sources?.length > 0 && (
                      <div className="sources">
                        <div className="sources-heading">
                          SOURCES USED
                        </div>
                        {message.sources.map((source) => (
                          <div className="source-card" key={source.id}>
                            <div className="source-topline">
                              <span className="source-tag">
                                [{source.id}]
                              </span>
                              <strong>{source.source.split(/[\\/]/).pop()}</strong>
                            </div>
                            <span className="source-page">
                              Page {source.page ?? "unknown"}
                            </span>
                            <p>{source.text}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {Number.isFinite(message.latency) && (
                      <div className="latency">
                        Answer generated in {message.latency}s
                      </div>
                    )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="loading-state">
                  <div className="avatar">D</div>
                  <div>
                    <div className="message-label">DOCUMIND</div>
                    <div className="typing">
                      <span />
                      <span />
                      <span />
                    </div>
                    <p>Searching documents and preparing an answer...</p>
                  </div>
                </div>
              )}

              <div id="chat-bottom" />
            </div>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <strong>Something went wrong</strong>
              <p>{error}</p>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
        </section>

        <footer className="composer-area">
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask something about your documents..."
              rows={2}
              maxLength={4000}
              disabled={loading}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey
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
            >
              {loading ? "…" : "↑"}
            </button>
          </form>
          <p className="composer-note">
            AI-generated answers may contain errors. Verify important claims
            against the original documents.
          </p>
        </footer>
      </main>
    </div>
  );
}
