"use client";

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

type Evidence = {
  source?: string;
  source_title?: string;
  source_type?: string;
  content?: string;
};

type Message = {
  role: 'user' | 'assistant';
  content: string;
  kpis?: any;
  evidence?: Evidence[];
};

/* ── Avatar Components ────────────────────────────────── */

function BotAvatar() {
  return (
    <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-900/30">
      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714a2.25 2.25 0 0 0 .659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-1.756-1.089a2.25 2.25 0 0 0-1.386-.361H8.142a2.25 2.25 0 0 0-1.386.361L5 14.5m14 0v4a2.25 2.25 0 0 1-2.25 2.25H7.25A2.25 2.25 0 0 1 5 18.5v-4m3.5 2.25h.008v.008H8.5v-.008Zm5 0h.008v.008h-.008v-.008Z" />
      </svg>
    </div>
  );
}

function UserAvatar() {
  return (
    <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-900/30">
      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
      </svg>
    </div>
  );
}

/* ── Typing / Loading Indicator ───────────────────────── */

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 py-2">
      <BotAvatar />
      <div className="bg-slate-900 border border-slate-800 rounded-2xl px-5 py-3 flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '0ms' }} />
        <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '150ms' }} />
        <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  );
}

/* ── Custom Markdown Components for headings ──────────── */

const markdownComponents = {
  h1: ({ children, ...props }: any) => (
    <h1 className="text-2xl font-bold text-cyan-300 mt-4 mb-2 border-b border-slate-700/50 pb-1" {...props}>{children}</h1>
  ),
  h2: ({ children, ...props }: any) => (
    <h2 className="text-xl font-bold text-cyan-200 mt-3 mb-2" {...props}>{children}</h2>
  ),
  h3: ({ children, ...props }: any) => (
    <h3 className="text-lg font-semibold text-slate-100 mt-3 mb-1" {...props}>{children}</h3>
  ),
  h4: ({ children, ...props }: any) => (
    <h4 className="text-base font-semibold text-slate-200 mt-2 mb-1" {...props}>{children}</h4>
  ),
  p: ({ children, ...props }: any) => (
    <p className="text-[15px] leading-relaxed text-slate-200 mb-2" {...props}>{children}</p>
  ),
  ul: ({ children, ...props }: any) => (
    <ul className="list-disc list-inside space-y-1 text-slate-300 mb-2 ml-2" {...props}>{children}</ul>
  ),
  ol: ({ children, ...props }: any) => (
    <ol className="list-decimal list-inside space-y-1 text-slate-300 mb-2 ml-2" {...props}>{children}</ol>
  ),
  li: ({ children, ...props }: any) => (
    <li className="text-[15px] text-slate-300" {...props}>{children}</li>
  ),
  strong: ({ children, ...props }: any) => (
    <strong className="font-bold text-slate-100" {...props}>{children}</strong>
  ),
  code: ({ children, ...props }: any) => (
    <code className="bg-slate-800 text-cyan-300 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>{children}</code>
  ),
};

/* ── Main Chat App ────────────────────────────────────── */

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isWaitingForFirstToken, setIsWaitingForFirstToken] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isWaitingForFirstToken]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userMsg = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMsg }]);
    setIsStreaming(true);
    setIsWaitingForFirstToken(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg }),
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';
      let addedAssistant = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (!dataStr) continue;

            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.text) {
                if (!addedAssistant) {
                  setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);
                  addedAssistant = true;
                  setIsWaitingForFirstToken(false);
                }
                assistantContent += parsed.text;
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].content = assistantContent;
                  return newMsgs;
                });
              } else if (parsed.kpis) {
                if (!addedAssistant) {
                  setMessages((prev) => [...prev, { role: 'assistant', content: assistantContent }]);
                  addedAssistant = true;
                  setIsWaitingForFirstToken(false);
                }
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].kpis = parsed.kpis;
                  newMsgs[newMsgs.length - 1].evidence = parsed.evidence;
                  return newMsgs;
                });
              }
            } catch (err) {
              console.error('Error parsing SSE data', err);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setIsWaitingForFirstToken(false);
      setMessages((prev) => [...prev, { role: 'assistant', content: 'An error occurred while connecting to the server.' }]);
    } finally {
      setIsStreaming(false);
      setIsWaitingForFirstToken(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Main Chat Area — now max-w-7xl for much wider layout */}
      <div className="flex-1 flex flex-col max-w-7xl w-full mx-auto h-full px-6 md:px-10 py-4">

        <header className="mb-6 py-4 border-b border-slate-800/50">
          <h1 className="text-4xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent tracking-tight">
            RAGent AI
          </h1>
          <p className="text-slate-400 text-base mt-1">High-performance Retrieval-Augmented Generation</p>
        </header>

        <div className="flex-1 overflow-y-auto space-y-6 pb-32" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
          {messages.length === 0 && !isWaitingForFirstToken && (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-4">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center border border-cyan-500/20">
                <svg className="w-10 h-10 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <p className="text-xl">What would you like to know about your game data?</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {/* Bot avatar on the left */}
              {msg.role === 'assistant' && <BotAvatar />}

              <div className="flex flex-col max-w-[85%]">
                <div
                  className={`rounded-2xl px-6 py-4 ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                      : 'bg-slate-900 border border-slate-800 shadow-xl'
                  }`}
                >
                  {msg.role === 'assistant' ? (
                    <div className="max-w-none">
                      <ReactMarkdown components={markdownComponents}>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-[16px] leading-relaxed">{msg.content}</p>
                  )}
                </div>

                {/* KPIs & Evidence Display */}
                {msg.role === 'assistant' && msg.kpis && (
                  <div className="mt-4 w-full">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
                        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Engine Latency</p>
                        <p className="text-lg font-bold text-slate-200">{(msg.kpis.engine_latency_ms / 1000).toFixed(2)} s</p>
                      </div>
                      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
                        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">LLM Latency</p>
                        <p className="text-lg font-bold text-slate-200">{msg.kpis.llm_latency_ms ? (msg.kpis.llm_latency_ms / 1000).toFixed(2) + ' s' : '-'}</p>
                      </div>
                      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
                        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Confidence</p>
                        <p className="text-lg font-bold text-cyan-400">{(msg.kpis.confidence_score * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
                        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Sources Used</p>
                        <p className="text-lg font-bold text-slate-200">{msg.kpis.retrieved_chunks}</p>
                      </div>
                    </div>

                    {msg.evidence && msg.evidence.length > 0 && (
                      <div className="mt-3 space-y-2">
                        <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-2">Sources (Evidence)</p>
                        {msg.evidence.slice(0, 3).map((ev, i) => (
                          <div key={i} className="text-sm bg-slate-900/30 border border-slate-800/80 rounded-lg p-3 flex flex-col hover:bg-slate-800/50 transition-colors">
                            <span className="font-semibold text-cyan-300/90 truncate">{ev.source_title || ev.source || "Unknown Source"}</span>
                            <span className="text-slate-500 text-xs mt-1 line-clamp-2">{ev.content || ''}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* User avatar on the right */}
              {msg.role === 'user' && <UserAvatar />}
            </div>
          ))}

          {/* Animated loading indicator while waiting for first token */}
          {isWaitingForFirstToken && <TypingIndicator />}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-slate-950 via-slate-950 to-transparent pointer-events-none">
          <div className="max-w-7xl mx-auto pointer-events-auto px-6 md:px-10">
            <form onSubmit={handleSubmit} className="relative group shadow-2xl shadow-cyan-900/5">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about the game database..."
                disabled={isStreaming}
                className="w-full bg-slate-900 border border-slate-800 rounded-2xl py-4 pl-6 pr-14 text-[16px] text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all backdrop-blur-xl disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={isStreaming || !input.trim()}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
