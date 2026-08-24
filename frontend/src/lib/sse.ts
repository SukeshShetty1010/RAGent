// Pure SSE buffer-splitting / frame-parsing logic extracted from
// app/page.tsx's handleSubmit so it's testable without a fetch/ReadableStream
// mock. Frame dispatch into React state stays in the component -- only the
// text-processing part moves here.

export type SSEFrame = { event: string; data: string };

// SSE frames end at a blank line, and a network read can stop anywhere --
// including the middle of one. Everything after the last separator is held
// back until the rest arrives (returned as `rest`, to be prepended to the
// next chunk). Parsing each read independently dropped every frame that
// straddled a boundary.
export function splitSSEBuffer(buffer: string): { frames: string[]; rest: string } {
  const frames = buffer.split(/\r?\n\r?\n/);
  const rest = frames.pop() ?? '';
  return { frames, rest };
}

export function parseSSEFrame(frame: string): SSEFrame | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const rawLine of frame.split('\n')) {
    const line = rawLine.replace(/\r$/, '');
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join('\n') };
}
