import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatApp from './page';

// Builds a mock fetch Response whose body is a ReadableStream emitting the
// given SSE frames (each frame gets the blank-line terminator appended),
// split across two chunks to also exercise the buffer-boundary path.
function mockSSEResponse(frames: string[]) {
  const text = frames.map((f) => `${f}\n\n`).join('');
  const bytes = new TextEncoder().encode(text);
  const mid = Math.floor(bytes.length / 2);
  const chunk1 = bytes.slice(0, mid);
  const chunk2 = bytes.slice(mid);

  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(chunk1);
      controller.enqueue(chunk2);
      controller.close();
    },
  });

  return new Response(body);
}

describe('ChatApp', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders streamed tokens and the final KPI/stage panel', async () => {
    const frames = [
      'event: stage\ndata: {"name":"routing","status":"completed","duration_ms":5,"data":{"task":"FACTUAL"}}',
      'event: token\ndata: {"text":"Hello"}',
      'event: token\ndata: {"text":" world"}',
      'event: done\ndata: {"final_answer":"Hello world","kpis":{"engine_latency_ms":100,"llm_ran":true,"llm_latency_ms":50,"quality_status":"quality_ok","confidence_score":0.9,"answer_capability":"full","retrieved_chunks":3,"task_success":true,"prompt_tokens":10,"completion_tokens":5,"cost_usd":0.0001,"finish_reason":"stop","answer_truncated":false},"evidence":[],"agent_decisions":{}}',
    ];
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(mockSSEResponse(frames));

    const user = userEvent.setup();
    render(<ChatApp />);

    const input = screen.getByPlaceholderText(/ask a question/i);
    await user.type(input, 'What is Far Cry 5?{Enter}');

    await waitFor(() => {
      expect(screen.getByText('Hello world')).toBeInTheDocument();
    });

    // KPI panel renders backend-provided values as-is.
    expect(screen.getByText('full')).toBeInTheDocument();
    expect(screen.getByText('quality_ok')).toBeInTheDocument();
  });

  it('shows the truncated-answer banner when finish_reason is not "stop"', async () => {
    const frames = [
      'event: token\ndata: {"text":"partial"}',
      'event: done\ndata: {"final_answer":"partial","kpis":{"engine_latency_ms":100,"llm_ran":true,"llm_latency_ms":50,"quality_status":"quality_ok","confidence_score":0.5,"answer_capability":"partial","retrieved_chunks":2,"task_success":true,"prompt_tokens":10,"completion_tokens":5,"cost_usd":0.0001,"finish_reason":"length","answer_truncated":true},"evidence":[],"agent_decisions":{}}',
    ];
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(mockSSEResponse(frames));

    const user = userEvent.setup();
    render(<ChatApp />);

    const input = screen.getByPlaceholderText(/ask a question/i);
    await user.type(input, 'Another question{Enter}');

    await waitFor(() => {
      expect(screen.getByText(/hit the model.s output limit/i)).toBeInTheDocument();
    });
  });

  it('renders a failed message on an error event', async () => {
    const frames = ['event: error\ndata: {"error":"boom"}'];
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(mockSSEResponse(frames));

    const user = userEvent.setup();
    render(<ChatApp />);

    const input = screen.getByPlaceholderText(/ask a question/i);
    await user.type(input, 'Trigger an error{Enter}');

    await waitFor(() => {
      expect(screen.getByText(/request failed: boom/i)).toBeInTheDocument();
    });
  });
});
