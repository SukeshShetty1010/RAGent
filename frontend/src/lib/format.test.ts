import { describe, it, expect } from 'vitest';
import {
  ms,
  msShort,
  capabilityTone,
  qualityTone,
  stageLabel,
  stageDetail,
  providerLabel,
  type Stage,
} from './format';

describe('ms', () => {
  it.each([
    [null, '—'],
    [undefined, '—'],
    [0, '0.00 s'],
    [1500, '1.50 s'],
    [125000, '125.00 s'],
  ])('ms(%p) -> %p', (input, expected) => {
    expect(ms(input as number | null | undefined)).toBe(expected);
  });
});

describe('msShort', () => {
  it.each([
    [null, '—'],
    [undefined, '—'],
    [0, '0 ms'],
    [12, '12 ms'],
    [999, '999 ms'],
    [1000, '1.00 s'],
    [12345, '12.35 s'],
  ])('msShort(%p) -> %p', (input, expected) => {
    expect(msShort(input as number | null | undefined)).toBe(expected);
  });
});

describe('capabilityTone', () => {
  it.each([
    ['full', 'good'],
    ['partial', 'warn'],
    ['insufficient', 'bad'],
    ['unknown_future_value', 'default'],
  ])('capabilityTone(%p) -> %p', (input, expected) => {
    expect(capabilityTone(input)).toBe(expected);
  });
});

describe('qualityTone', () => {
  it.each([
    ['quality_ok', 'good'],
    ['quality_weak', 'warn'],
    ['quality_empty', 'bad'],
    ['unknown_future_value', 'default'],
  ])('qualityTone(%p) -> %p', (input, expected) => {
    expect(qualityTone(input)).toBe(expected);
  });
});

describe('stageLabel', () => {
  it('returns the known display name for a mapped stage', () => {
    expect(stageLabel('query_rewrite')).toBe('Query Rewrite');
    expect(stageLabel('context_assembly')).toBe('Context Assembly');
  });

  it('falls back to an underscore-replaced name for an unrecognized stage', () => {
    expect(stageLabel('future_stage_name')).toBe('future stage name');
  });
});

describe('providerLabel', () => {
  it('returns the known display name for a mapped provider', () => {
    expect(providerLabel('gemini')).toBe('Gemini');
    expect(providerLabel('hfspace')).toBe('HF Space');
  });

  it('falls back to a capitalized key for an unrecognized provider', () => {
    expect(providerLabel('newprovider')).toBe('Newprovider');
  });
});

describe('stageDetail', () => {
  const base = (overrides: Partial<Stage>): Stage => ({
    name: 'query_rewrite',
    status: 'completed',
    duration_ms: 10,
    data: {},
    ...overrides,
  });

  it('query_rewrite: undefined when no source', () => {
    expect(stageDetail(base({ name: 'query_rewrite', data: {} }))).toBeUndefined();
  });

  it('query_rewrite: rewritten', () => {
    expect(
      stageDetail(base({ name: 'query_rewrite', data: { rewritten: true, source: 'llm' } })),
    ).toBe('rewritten · llm');
  });

  it('query_rewrite: unchanged', () => {
    expect(
      stageDetail(
        base({ name: 'query_rewrite', data: { rewritten: false, source: 'skipped_no_history' } }),
      ),
    ).toBe('unchanged · skipped_no_history');
  });

  it('routing: task with signals', () => {
    expect(
      stageDetail(
        base({ name: 'routing', data: { task: 'COMPARISON', signals: ['vs', 'compare'] } }),
      ),
    ).toBe('COMPARISON · vs, compare');
  });

  it('routing: task without signals', () => {
    expect(stageDetail(base({ name: 'routing', data: { task: 'FACTUAL', signals: [] } }))).toBe(
      'FACTUAL',
    );
  });

  it('routing: undefined when no task', () => {
    expect(stageDetail(base({ name: 'routing', data: {} }))).toBeUndefined();
  });

  it('strategy: with limit and web fallback', () => {
    expect(
      stageDetail(
        base({ name: 'strategy', data: { config: { limit: 10, allow_web_fallback: true } } }),
      ),
    ).toBe('limit 10 · web fallback');
  });

  it('strategy: undefined when no limit', () => {
    expect(stageDetail(base({ name: 'strategy', data: { config: {} } }))).toBeUndefined();
  });

  it('retrieval: with dropped noise', () => {
    expect(
      stageDetail(
        base({
          name: 'retrieval',
          data: { chunks_found: 12, merge_state: 'corpus_only', quality: 'quality_ok', chunks_dropped_as_noise: 3 },
        }),
      ),
    ).toBe('12 chunks · corpus_only · quality_ok · 3 dropped as noise');
  });

  it('retrieval: without dropped noise', () => {
    expect(
      stageDetail(
        base({
          name: 'retrieval',
          data: { chunks_found: 5, merge_state: 'corpus_only', quality: 'quality_ok', chunks_dropped_as_noise: 0 },
        }),
      ),
    ).toBe('5 chunks · corpus_only · quality_ok');
  });

  it('capability: string value', () => {
    expect(stageDetail(base({ name: 'capability', data: { capability: 'full' } }))).toBe('full');
  });

  it('context_assembly: with chunks', () => {
    expect(
      stageDetail(base({ name: 'context_assembly', data: { chunks_assembled: 7 } })),
    ).toBe('7 chunks');
  });

  it('prompt_construction: with length', () => {
    expect(
      stageDetail(base({ name: 'prompt_construction', data: { prompt_length: 4200 } })),
    ).toBe('4200 chars');
  });

  it('generation: skipped with reason', () => {
    expect(
      stageDetail(
        base({ name: 'generation', status: 'skipped', data: { reason: 'insufficient_evidence' } }),
      ),
    ).toBe('insufficient_evidence');
  });

  it('generation: completed with tokens', () => {
    expect(
      stageDetail(base({ name: 'generation', status: 'completed', data: { tokens_generated: 512 } })),
    ).toBe('512 tokens');
  });

  it('default: unrecognized stage name returns undefined', () => {
    expect(stageDetail(base({ name: 'unknown_future_stage', data: { anything: 1 } }))).toBeUndefined();
  });
});
