import { describe, it, expect } from 'vitest';
import { splitSSEBuffer, parseSSEFrame } from './sse';

describe('splitSSEBuffer', () => {
  it('splits a single complete frame, leaving no rest', () => {
    const { frames, rest } = splitSSEBuffer('event: token\ndata: {"text":"hi"}\n\n');
    expect(frames).toEqual(['event: token\ndata: {"text":"hi"}']);
    expect(rest).toBe('');
  });

  it('splits multiple frames from one buffer', () => {
    const buffer =
      'event: token\ndata: {"text":"a"}\n\n' + 'event: token\ndata: {"text":"b"}\n\n';
    const { frames, rest } = splitSSEBuffer(buffer);
    expect(frames).toEqual(['event: token\ndata: {"text":"a"}', 'event: token\ndata: {"text":"b"}']);
    expect(rest).toBe('');
  });

  it('holds back a frame split across chunk boundaries', () => {
    // Simulates a network read stopping mid-frame: no trailing blank
    // line yet, so the whole thing is "rest" awaiting the next chunk.
    const { frames, rest } = splitSSEBuffer('event: done\ndata: {"final_ans');
    expect(frames).toEqual([]);
    expect(rest).toBe('event: done\ndata: {"final_ans');
  });

  it('reassembles a frame split across two buffer chunks', () => {
    const chunk1 = 'event: done\ndata: {"final_ans';
    const { frames: frames1, rest: rest1 } = splitSSEBuffer(chunk1);
    expect(frames1).toEqual([]);

    const chunk2 = rest1 + 'wer":"ok"}\n\n';
    const { frames: frames2, rest: rest2 } = splitSSEBuffer(chunk2);
    expect(frames2).toEqual(['event: done\ndata: {"final_answer":"ok"}']);
    expect(rest2).toBe('');
  });

  it('handles CRLF separators', () => {
    const { frames, rest } = splitSSEBuffer('event: token\r\ndata: {"text":"a"}\r\n\r\n');
    expect(frames).toEqual(['event: token\r\ndata: {"text":"a"}']);
    expect(rest).toBe('');
  });
});

describe('parseSSEFrame', () => {
  it('parses event and data fields', () => {
    expect(parseSSEFrame('event: token\ndata: {"text":"hi"}')).toEqual({
      event: 'token',
      data: '{"text":"hi"}',
    });
  });

  it('defaults event to "message" when no event: line present', () => {
    expect(parseSSEFrame('data: {"text":"hi"}')).toEqual({
      event: 'message',
      data: '{"text":"hi"}',
    });
  });

  it('joins multiple data: lines with newlines', () => {
    expect(parseSSEFrame('event: done\ndata: line1\ndata: line2')).toEqual({
      event: 'done',
      data: 'line1\nline2',
    });
  });

  it('returns null for a malformed frame with no data: line', () => {
    expect(parseSSEFrame('event: token')).toBeNull();
  });

  it('strips trailing \\r from CRLF-separated lines', () => {
    expect(parseSSEFrame('event: token\r\ndata: {"text":"a"}')).toEqual({
      event: 'token',
      data: '{"text":"a"}',
    });
  });
});
