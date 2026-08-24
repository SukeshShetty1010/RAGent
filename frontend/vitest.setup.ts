import '@testing-library/jest-dom/vitest';

// jsdom doesn't implement scrollIntoView; page.tsx calls it on every
// message-list update, so leaving it undefined crashes every render.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
