"use client";

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

type Evidence = {
  source?: string;
  source_title?: string;
  source_type?: string;
  content?: string;
};

// Mirrors the kpis dict built in engine/execution_engine_streaming.py.
// Rendered as received -- nothing here is recomputed in the UI.
type Kpis = {
  engine_latency_ms: number;
  llm_ran: boolean;
  llm_latency_ms: number | null;
  quality_status: string;
  confidence_score: number;
  answer_capability: string;
  retrieved_chunks: number;
  task_success: boolean;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  cost_usd: number | null;
  finish_reason: string | null;
  answer_truncated: boolean;
  answer_model?: string | null;
};

// Mirrors the StreamingStage dataclass in
// engine/execution_engine_streaming.py -- one entry per pipeline step,
// emitted at "started" and again at "completed"/"skipped"/"cancelled".
type Stage = {
  name: string;
  status: string; // started | completed | skipped | cancelled | failed
  duration_ms: number | null;
  data?: Record<string, unknown>;
};

// Mirrors agent_decisions as built in execution_engine_streaming.py.
// Every field is optional -- the panel renders whatever the backend
// actually populated for this query rather than assuming a fixed shape.
type QualitySnapshot = {
  status?: string;
  confidence_score?: number;
  has_temporal_signal?: boolean;
  max_relevance?: number;
  entity_grounded?: boolean;
  evidence_count?: number;
};

type WebSearchDecision = {
  should_search_web?: boolean;
  reason?: string;
  confidence?: number;
  source?: string; // "llm" | "deterministic_fallback"
};

type OutputValidation = {
  is_valid?: boolean;
  issues?: string[];
  has_required_section?: boolean;
  cited_sources?: string[];
  unmatched_citations?: string[];
  citation_count?: number;
};

type AgentDecisions = {
  task?: string;
  intent_signals?: string[];
  routing_reason?: string;
  retrieval_strategy?: {
    limit?: number;
    use_query_decomposition?: boolean;
    use_window_expansion?: boolean;
    allow_web_fallback?: boolean;
  };
  merge_state?: string;
  quality?: QualitySnapshot;
  quality_pre_web?: QualitySnapshot;
  web_search_decision?: WebSearchDecision;
  answer_capability?: string;
  output_validation?: OutputValidation;
};

type Message = {
  role: 'user' | 'assistant';
  content: string;
  kpis?: Kpis;
  evidence?: Evidence[];
  failed?: boolean;
  stages?: Stage[];
  agent_decisions?: AgentDecisions;
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

/* ── Per-answer KPI panel ─────────────────────────────── */

function Metric({ label, value, tone = 'default', className = '' }: { label: string; value: string; tone?: 'default' | 'good' | 'warn' | 'bad' | 'accent'; className?: string }) {
  const toneClass = {
    default: 'text-slate-200',
    accent: 'text-cyan-400',
    good: 'text-emerald-400',
    warn: 'text-amber-400',
    bad: 'text-red-400',
  }[tone];
  return (
    <div className={`bg-slate-900/50 border border-slate-800 rounded-xl p-3 ${className}`}>
      <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</p>
      <p className={`text-lg font-bold ${toneClass}`}>{value}</p>
    </div>
  );
}

const ms = (v: number | null | undefined) => (v == null ? '—' : `${(v / 1000).toFixed(2)} s`);

// ms() rounds everything to seconds, which turns a 12ms routing stage
// into "0.01 s" -- too coarse to tell a fast stage from a stalled one.
const msShort = (v: number | null | undefined) => {
  if (v == null) return '—';
  return v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`;
};

// Tone only -- the label itself is whatever the backend sent, so a new
// capability or gate status still renders rather than vanishing.
function capabilityTone(v: string): 'good' | 'warn' | 'bad' | 'default' {
  if (v === 'full') return 'good';
  if (v === 'partial') return 'warn';
  if (v === 'insufficient') return 'bad';
  return 'default';
}

function qualityTone(v: string): 'good' | 'warn' | 'bad' | 'default' {
  if (v === 'quality_ok') return 'good';
  if (v === 'quality_weak') return 'warn';
  if (v === 'quality_empty') return 'bad';
  return 'default';
}

function KpiPanel({ kpis }: { kpis: Kpis }) {
  const tokens =
    kpis.prompt_tokens == null && kpis.completion_tokens == null
      ? '—'
      : `${kpis.prompt_tokens ?? 0} → ${kpis.completion_tokens ?? 0}`;
  const cost = kpis.cost_usd == null ? '—' : `$${kpis.cost_usd.toFixed(6)}`;

  return (
    <div className="mt-4 w-full">
      {kpis.answer_truncated && (
        <div className="mb-3 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-300">
          {kpis.finish_reason === 'length'
            ? 'This answer hit the model’s output limit and is incomplete.'
            : 'The model closed the stream before finishing — this answer is incomplete.'}
        </div>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Metric label="Engine Latency" value={ms(kpis.engine_latency_ms)} />
        <Metric label="LLM Latency" value={ms(kpis.llm_latency_ms)} />
        <Metric label="Confidence" value={`${(kpis.confidence_score * 100).toFixed(1)}%`} tone="accent" />
        <Metric label="Sources Used" value={String(kpis.retrieved_chunks)} />
        <Metric label="Capability" value={kpis.answer_capability} tone={capabilityTone(kpis.answer_capability)} />
        <Metric label="Quality Gate" value={kpis.quality_status} tone={qualityTone(kpis.quality_status)} />
        <Metric label="Tokens (in → out)" value={tokens} />
        <Metric label="Est. Cost" value={cost} />
        <Metric label="Answer Model" value={kpis.answer_model ?? '—'} className="col-span-2" />
      </div>
    </div>
  );
}

/* ── Pipeline Stage Progress ──────────────────────────── */

// Display names only. Which stages exist is the backend's call -- an
// unrecognized stage name still renders via the fallback, same reasoning
// as PROVIDER_LABELS above.
const STAGE_LABELS: Record<string, string> = {
  routing: 'Routing',
  strategy: 'Strategy Selection',
  retrieval: 'Retrieval',
  capability: 'Capability Assessment',
  context_assembly: 'Context Assembly',
  prompt_construction: 'Prompt Construction',
  generation: 'Generation',
};

function stageLabel(name: string): string {
  return STAGE_LABELS[name] ?? name.replace(/_/g, ' ');
}

function stageDetail(stage: Stage): string | undefined {
  const d = stage.data ?? {};
  switch (stage.name) {
    case 'routing': {
      const signals = Array.isArray(d.signals) ? (d.signals as string[]) : [];
      return d.task ? `${d.task}${signals.length ? ' · ' + signals.join(', ') : ''}` : undefined;
    }
    case 'strategy': {
      const config = (d.config ?? {}) as Record<string, unknown>;
      if (config.limit == null) return undefined;
      return `limit ${config.limit}${config.allow_web_fallback ? ' · web fallback' : ''}`;
    }
    case 'retrieval':
      if (d.chunks_found == null) return undefined;
      return `${d.chunks_found} chunks · ${d.merge_state} · ${d.quality}`;
    case 'capability':
      return typeof d.capability === 'string' ? d.capability : undefined;
    case 'context_assembly':
      return d.chunks_assembled != null ? `${d.chunks_assembled} chunks` : undefined;
    case 'prompt_construction':
      return d.prompt_length != null ? `${d.prompt_length} chars` : undefined;
    case 'generation':
      if (stage.status === 'skipped') return typeof d.reason === 'string' ? d.reason : undefined;
      return d.tokens_generated != null ? `${d.tokens_generated} tokens` : undefined;
    default:
      return undefined;
  }
}

function StageGlyph({ status }: { status: string }) {
  if (status === 'started') {
    return <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse flex-shrink-0" />;
  }
  if (status === 'completed') {
    return <span className="text-emerald-400 flex-shrink-0">✓</span>;
  }
  if (status === 'skipped') {
    return <span className="text-slate-500 flex-shrink-0">↷</span>;
  }
  if (status === 'cancelled' || status === 'failed') {
    return <span className="text-red-400 flex-shrink-0">⨯</span>;
  }
  return <span className="text-slate-600 flex-shrink-0">•</span>;
}

function StageProgress({ stages, live = false }: { stages: Stage[]; live?: boolean }) {
  if (stages.length === 0) return null;
  return (
    <div className="space-y-1.5">
      {stages.map((stage) => {
        const detail = stageDetail(stage);
        return (
          <div
            key={stage.name}
            className="flex items-center gap-2.5 bg-slate-900/50 border border-slate-800 rounded-xl px-3 py-2 text-sm"
          >
            <StageGlyph status={stage.status} />
            <span className="text-slate-200 font-medium">{stageLabel(stage.name)}</span>
            {detail && <span className="text-slate-500 truncate">· {detail}</span>}
            <span className="ml-auto text-slate-500 text-xs flex-shrink-0">{msShort(stage.duration_ms)}</span>
          </div>
        );
      })}
      {live && <TypingIndicatorDots />}
    </div>
  );
}

function TypingIndicatorDots() {
  return (
    <div className="flex items-center gap-1.5 px-3 py-1">
      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '0ms' }} />
      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '150ms' }} />
      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  );
}

/* ── Agent Decisions Panel ─────────────────────────────── */

function AgentDecisionsPanel({ decisions }: { decisions: AgentDecisions }) {
  const hasAnything =
    decisions.task || decisions.retrieval_strategy || decisions.quality ||
    decisions.web_search_decision || decisions.output_validation;
  if (!hasAnything) return null;

  return (
    <div className="mt-2 space-y-3 text-sm">
      {decisions.task && (
        <div>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-1">Routing</p>
          <p className="text-slate-300">
            <span className="text-slate-100 font-medium">{decisions.task}</span>
            {decisions.intent_signals && decisions.intent_signals.length > 0 && (
              <span className="text-slate-500"> · {decisions.intent_signals.join(', ')}</span>
            )}
          </p>
          {decisions.routing_reason && <p className="text-slate-500 text-xs mt-0.5">{decisions.routing_reason}</p>}
        </div>
      )}

      {decisions.retrieval_strategy && (
        <div>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-1">Retrieval Strategy</p>
          <p className="text-slate-300">
            limit {decisions.retrieval_strategy.limit}
            {decisions.retrieval_strategy.use_query_decomposition && ' · query decomposition'}
            {decisions.retrieval_strategy.use_window_expansion && ' · window expansion'}
            {decisions.retrieval_strategy.allow_web_fallback && ' · web fallback allowed'}
          </p>
        </div>
      )}

      {decisions.quality && (
        <div>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-1">Quality Gate</p>
          <p className="text-slate-300">
            <span className={qualityTone(decisions.quality.status ?? '') === 'bad' ? 'text-red-400' : qualityTone(decisions.quality.status ?? '') === 'warn' ? 'text-amber-400' : 'text-emerald-400'}>
              {decisions.quality.status}
            </span>
            {decisions.quality.confidence_score != null && ` · ${(decisions.quality.confidence_score * 100).toFixed(1)}% confidence`}
            {decisions.quality.evidence_count != null && ` · ${decisions.quality.evidence_count} evidence`}
            {decisions.quality.entity_grounded === false && ' · not entity-grounded'}
          </p>
          {decisions.quality_pre_web && (
            <p className="text-slate-500 text-xs mt-0.5">
              Before web rescue: {decisions.quality_pre_web.status}
              {decisions.quality_pre_web.confidence_score != null && ` (${(decisions.quality_pre_web.confidence_score * 100).toFixed(1)}%)`}
            </p>
          )}
        </div>
      )}

      {decisions.web_search_decision && (
        <div>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-1">Web Search Decision</p>
          <p className="text-slate-300">
            {decisions.web_search_decision.should_search_web ? 'Searched the web' : 'Did not search the web'}
            {decisions.web_search_decision.confidence != null && ` · ${(decisions.web_search_decision.confidence * 100).toFixed(0)}% confidence`}
            {decisions.web_search_decision.source && (
              <span className="text-slate-500"> · {decisions.web_search_decision.source}</span>
            )}
          </p>
          {decisions.web_search_decision.reason && (
            <p className="text-slate-500 text-xs mt-0.5">{decisions.web_search_decision.reason}</p>
          )}
        </div>
      )}

      {decisions.output_validation && (
        <div>
          <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-1">Output Validation</p>
          <p className={decisions.output_validation.is_valid ? 'text-emerald-400' : 'text-amber-400'}>
            {decisions.output_validation.is_valid ? 'Valid' : 'Issues found'}
            {decisions.output_validation.citation_count != null && ` · ${decisions.output_validation.citation_count} citations`}
          </p>
          {decisions.output_validation.issues && decisions.output_validation.issues.length > 0 && (
            <ul className="text-slate-500 text-xs mt-0.5 list-disc list-inside">
              {decisions.output_validation.issues.map((issue, i) => <li key={i}>{issue}</li>)}
            </ul>
          )}
          {decisions.output_validation.unmatched_citations && decisions.output_validation.unmatched_citations.length > 0 && (
            <p className="text-slate-500 text-xs mt-0.5">
              Unmatched citations: {decisions.output_validation.unmatched_citations.join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function PipelinePanel({ stages, decisions }: { stages: Stage[]; decisions?: AgentDecisions }) {
  if (stages.length === 0) return null;
  const totalMs = stages.reduce((sum, s) => sum + (s.duration_ms ?? 0), 0);
  return (
    <details className="mt-3 group">
      <summary className="cursor-pointer text-xs text-slate-500 font-semibold uppercase tracking-wider select-none hover:text-slate-400">
        Pipeline · {stages.length} stages · {msShort(totalMs)}
      </summary>
      <div className="mt-2 space-y-3">
        <StageProgress stages={stages} />
        {decisions && <AgentDecisionsPanel decisions={decisions} />}
      </div>
    </details>
  );
}

/* ── Usage Dashboard ──────────────────────────────────── */

type ProviderUsage = {
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  fallback_events: number;
  limits?: { rpd?: number; rpm?: number };
  percent_of_rpd?: number | null;
};

type SurfaceUsage = {
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
};

type UsageData = {
  date: string;
  degraded: boolean;
  by_provider: Record<string, ProviderUsage>;
  by_surface: Record<string, SurfaceUsage>;
};

// Display names only. Which providers exist is the backend's call —
// /api/usage returns them — so this map is a lookup, never the source of
// truth: an unrecognized key still renders, capitalized.
const PROVIDER_LABELS: Record<string, string> = {
  gemini: 'Gemini',
  groq: 'Groq',
  cloudflare: 'Cloudflare',
  voyage: 'Voyage',
  hfspace: 'HF Space',
};

function providerLabel(key: string): string {
  return PROVIDER_LABELS[key] ?? key.charAt(0).toUpperCase() + key.slice(1);
}

function UsageCard({ name, usage }: { name: string; usage?: ProviderUsage }) {
  const pct = usage?.percent_of_rpd ?? 0;
  const barColor = pct >= 90 ? 'bg-red-500' : pct >= 60 ? 'bg-amber-500' : 'bg-cyan-500';
  const tokens = (usage?.prompt_tokens ?? 0) + (usage?.completion_tokens ?? 0);
  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
      <p className="text-sm font-semibold text-slate-200 uppercase tracking-wider">{name}</p>
      <p className="text-2xl font-bold text-slate-100 mt-1">
        {usage?.requests ?? 0}
        {usage?.limits?.rpd ? (
          <span className="text-sm font-normal text-slate-500"> / {usage.limits.rpd} req/day</span>
        ) : (
          <span className="text-sm font-normal text-slate-500"> req</span>
        )}
      </p>
      {usage?.limits?.rpd ? (
        <div className="w-full h-2 bg-slate-800 rounded-full mt-2 overflow-hidden">
          <div
            className={`h-full ${barColor} transition-all`}
            style={{ width: `${Math.min(100, pct)}%` }}
          />
        </div>
      ) : null}
      <p className="text-xs text-slate-500 mt-2">
        {/* No progress bar without a declared rpd: providers capped on
            something else (Cloudflare bills Neurons) would otherwise get
            a bar drawn against an invented limit. Tokens are shown
            instead, which is the figure their cap derives from. */}
        {tokens > 0 ? `${tokens.toLocaleString()} tokens · ` : ''}
        {usage?.fallback_events ?? 0} fallback event(s)
      </p>
    </div>
  );
}

function UsageDashboard({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<UsageData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
    fetch(`${apiUrl}/api/usage`)
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch(() => setError('Could not load usage data.'));

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4">
      <div className="bg-slate-950 border border-slate-800 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-slate-100">Usage Today{data ? ` — ${data.date}` : ''}</h2>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium transition-colors"
          >
            Close
          </button>
        </div>

        {error && <p className="text-red-400 text-sm">{error}</p>}
        {!data && !error && <p className="text-slate-500 text-sm">Loading...</p>}

        {data && (
          <>
            {data.degraded && (
              <p className="text-amber-400 text-xs mb-3">
                Degraded: showing in-process counts only (Qdrant unreachable).
              </p>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Driven by whatever /api/usage returns, in its order.
                  Hardcoding the provider list here meant a new backend
                  provider (the Cloudflare reranker) was counted but
                  invisible — the UI was deciding which providers exist,
                  which is the backend's job. */}
              {Object.entries(data.by_provider ?? {}).map(([key, usage]) => (
                <UsageCard key={key} name={providerLabel(key)} usage={usage} />
              ))}
            </div>

            {Object.keys(data.by_surface || {}).length > 0 && (
              <div className="mt-6">
                <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-2">By surface</p>
                <table className="w-full text-sm text-left">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-800">
                      <th className="py-1 font-medium">Surface</th>
                      <th className="py-1 font-medium text-right">Requests</th>
                      <th className="py-1 font-medium text-right">Prompt tok</th>
                      <th className="py-1 font-medium text-right">Completion tok</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.by_surface).map(([surface, s]) => (
                      <tr key={surface} className="border-b border-slate-900 text-slate-300">
                        <td className="py-1.5">{surface}</td>
                        <td className="py-1.5 text-right">{s.requests}</td>
                        <td className="py-1.5 text-right">{s.prompt_tokens}</td>
                        <td className="py-1.5 text-right">{s.completion_tokens}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ── Main Chat App ────────────────────────────────────── */

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isWaitingForFirstToken, setIsWaitingForFirstToken] = useState(false);
  const [showUsage, setShowUsage] = useState(false);
  const [activeStages, setActiveStages] = useState<Stage[]>([]);
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

      // Stages start arriving before the assistant message exists --
      // routing completes long before the first token -- so they can't
      // ride along on updateAssistant. Mirrored into state via
      // upsertStage so the live view re-renders as each one lands.
      const stageList: Stage[] = [];
      const upsertStage = (s: Stage) => {
        const i = stageList.findIndex((x) => x.name === s.name);
        if (i === -1) stageList.push(s);
        else stageList[i] = { ...stageList[i], ...s };
        setActiveStages([...stageList]);
      };

      const ensureAssistant = () => {
        if (addedAssistant) return;
        setMessages((prev) => [...prev, { role: 'assistant', content: assistantContent }]);
        addedAssistant = true;
        setIsWaitingForFirstToken(false);
      };

      // Mutate the assistant message in place without touching earlier
      // ones -- the streamed message is always the last in the list.
      const updateAssistant = (patch: Partial<Message>) => {
        setMessages((prev) =>
          prev.map((m, i) => (i === prev.length - 1 ? { ...m, ...patch } : m)),
        );
      };

      const handleFrame = (frame: string) => {
        let event = 'message';
        const dataLines: string[] = [];
        for (const rawLine of frame.split('\n')) {
          const line = rawLine.replace(/\r$/, '');
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
        }
        if (dataLines.length === 0) return;

        let parsed;
        try {
          parsed = JSON.parse(dataLines.join('\n'));
        } catch (err) {
          console.error('Error parsing SSE data', err);
          return;
        }

        if (event === 'token') {
          ensureAssistant();
          assistantContent += parsed.text;
          updateAssistant({ content: assistantContent });
        } else if (event === 'done') {
          ensureAssistant();
          // final_answer is the backend's authoritative copy of the
          // answer. Preferring it over the locally accumulated tokens
          // keeps the rendered text identical to what the engine
          // validated and traced, instead of a reconstruction that a
          // single dropped frame would leave subtly wrong.
          updateAssistant({
            content: parsed.final_answer ?? assistantContent,
            kpis: parsed.kpis,
            evidence: parsed.evidence,
            stages: [...stageList],
            agent_decisions: parsed.agent_decisions,
          });
        } else if (event === 'error') {
          ensureAssistant();
          updateAssistant({
            content: `Request failed: ${parsed.error}`,
            failed: true,
            stages: [...stageList],
          });
        } else if (event === 'stage') {
          upsertStage(parsed);
        }
      };

      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // SSE frames end at a blank line, and a network read can stop
        // anywhere -- including the middle of one. Everything after the
        // last separator is held back until the rest arrives. Parsing
        // each read independently dropped every frame that straddled a
        // boundary, which is why the ~5.6KB done event carrying the KPIs
        // and evidence never rendered while the small token frames did.
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() ?? '';
        for (const frame of frames) handleFrame(frame);
      }
      // A final frame with no trailing blank line still counts.
      buffer += decoder.decode();
      if (buffer.trim()) handleFrame(buffer);
    } catch (err) {
      console.error(err);
      setIsWaitingForFirstToken(false);
      setMessages((prev) => [...prev, { role: 'assistant', content: 'An error occurred while connecting to the server.' }]);
    } finally {
      setIsStreaming(false);
      setIsWaitingForFirstToken(false);
      setActiveStages([]);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Main Chat Area — now max-w-7xl for much wider layout */}
      <div className="flex-1 flex flex-col max-w-7xl w-full mx-auto h-full px-6 md:px-10 py-4">

        <header className="mb-6 py-4 border-b border-slate-800/50 flex items-start justify-between">
          <div>
            <h1 className="text-4xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent tracking-tight">
              RAGent AI
            </h1>
            <p className="text-slate-400 text-base mt-1">High-performance Retrieval-Augmented Generation</p>
          </div>
          <button
            onClick={() => setShowUsage(true)}
            className="px-4 py-2 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 text-sm font-medium transition-colors mt-2"
          >
            Usage
          </button>
        </header>

        {showUsage && <UsageDashboard onClose={() => setShowUsage(false)} />}

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
                      : msg.failed
                        ? 'bg-slate-900 border border-red-500/40 shadow-xl'
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
                  <>
                    <KpiPanel kpis={msg.kpis} />
                    {msg.stages && <PipelinePanel stages={msg.stages} decisions={msg.agent_decisions} />}

                    {msg.evidence && msg.evidence.length > 0 && (
                      <div className="mt-3 space-y-2">
                        <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-2">
                          Sources (Evidence) — showing {Math.min(3, msg.evidence.length)} of {msg.evidence.length}
                        </p>
                        {msg.evidence.slice(0, 3).map((ev, i) => (
                          <div key={i} className="text-sm bg-slate-900/30 border border-slate-800/80 rounded-lg p-3 flex flex-col hover:bg-slate-800/50 transition-colors">
                            <span className="font-semibold text-cyan-300/90 truncate">{ev.source_title || ev.source || "Unknown Source"}</span>
                            <span className="text-slate-500 text-xs mt-1 line-clamp-2">{ev.content || ''}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* User avatar on the right */}
              {msg.role === 'user' && <UserAvatar />}
            </div>
          ))}

          {/* Live pipeline progress -- stays mounted while tokens stream,
              since the "generation" stage only completes after the
              stream ends. Falls back to the bouncing dots for the brief
              gap before the first stage event arrives. */}
          {isStreaming && activeStages.length > 0 ? (
            <div className="flex items-start gap-3 py-2">
              <BotAvatar />
              <div className="flex-1 max-w-[85%]">
                <StageProgress stages={activeStages} live />
              </div>
            </div>
          ) : (
            isWaitingForFirstToken && <TypingIndicator />
          )}

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
