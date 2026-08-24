// Pure formatting/lookup helpers extracted from app/page.tsx so they can
// be unit tested directly instead of only through a full component mount.

// Mirrors the StreamingStage dataclass in
// engine/execution_engine_streaming.py -- one entry per pipeline step,
// emitted at "started" and again at "completed"/"skipped"/"cancelled".
export type Stage = {
  name: string;
  status: string; // started | completed | skipped | cancelled | failed
  duration_ms: number | null;
  data?: Record<string, unknown>;
};

export const ms = (v: number | null | undefined) => (v == null ? '—' : `${(v / 1000).toFixed(2)} s`);

// ms() rounds everything to seconds, which turns a 12ms routing stage
// into "0.01 s" -- too coarse to tell a fast stage from a stalled one.
export const msShort = (v: number | null | undefined) => {
  if (v == null) return '—';
  return v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`;
};

// Tone only -- the label itself is whatever the backend sent, so a new
// capability or gate status still renders rather than vanishing.
export function capabilityTone(v: string): 'good' | 'warn' | 'bad' | 'default' {
  if (v === 'full') return 'good';
  if (v === 'partial') return 'warn';
  if (v === 'insufficient') return 'bad';
  return 'default';
}

export function qualityTone(v: string): 'good' | 'warn' | 'bad' | 'default' {
  if (v === 'quality_ok') return 'good';
  if (v === 'quality_weak') return 'warn';
  if (v === 'quality_empty') return 'bad';
  return 'default';
}

// Display names only. Which stages exist is the backend's call -- an
// unrecognized stage name still renders via the fallback, same reasoning
// as PROVIDER_LABELS below.
export const STAGE_LABELS: Record<string, string> = {
  query_rewrite: 'Query Rewrite',
  routing: 'Routing',
  strategy: 'Strategy Selection',
  retrieval: 'Retrieval',
  capability: 'Capability Assessment',
  context_assembly: 'Context Assembly',
  prompt_construction: 'Prompt Construction',
  generation: 'Generation',
};

export function stageLabel(name: string): string {
  return STAGE_LABELS[name] ?? name.replace(/_/g, ' ');
}

export function stageDetail(stage: Stage): string | undefined {
  const d = stage.data ?? {};
  switch (stage.name) {
    case 'query_rewrite':
      if (d.source == null) return undefined;
      return d.rewritten ? `rewritten · ${d.source}` : `unchanged · ${d.source}`;
    case 'routing': {
      const signals = Array.isArray(d.signals) ? (d.signals as string[]) : [];
      return d.task ? `${d.task}${signals.length ? ' · ' + signals.join(', ') : ''}` : undefined;
    }
    case 'strategy': {
      const config = (d.config ?? {}) as Record<string, unknown>;
      if (config.limit == null) return undefined;
      return `limit ${config.limit}${config.allow_web_fallback ? ' · web fallback' : ''}`;
    }
    case 'retrieval': {
      if (d.chunks_found == null) return undefined;
      const dropped = typeof d.chunks_dropped_as_noise === 'number' && d.chunks_dropped_as_noise > 0
        ? ` · ${d.chunks_dropped_as_noise} dropped as noise`
        : '';
      return `${d.chunks_found} chunks · ${d.merge_state} · ${d.quality}${dropped}`;
    }
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

// Display names only. Which providers exist is the backend's call —
// /api/usage returns them — so this map is a lookup, never the source of
// truth: an unrecognized key still renders, capitalized.
export const PROVIDER_LABELS: Record<string, string> = {
  gemini: 'Gemini',
  groq: 'Groq',
  cloudflare: 'Cloudflare',
  voyage: 'Voyage',
  hfspace: 'HF Space',
};

export function providerLabel(key: string): string {
  return PROVIDER_LABELS[key] ?? key.charAt(0).toUpperCase() + key.slice(1);
}
