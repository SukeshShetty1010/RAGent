# RAGent Frontend

This is the Next.js web application for **RAGent** — the Capability-Aware Agentic RAG for Gaming Intelligence.

## Getting Started

First, ensure your Python FastAPI backend is running on port 8000 (see the root `README.md` for instructions).

Then, run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Configuration

The application uses an environment variable to connect to the backend API.

- `NEXT_PUBLIC_API_URL`: The URL of the FastAPI backend. If not set, it defaults to `http://localhost:8000`.

To change this, create a `.env.local` file in the `frontend` directory:

```env
NEXT_PUBLIC_API_URL=http://your-production-url.com
```

## Structure

- `src/app/page.tsx`: The main chat interface that handles SSE (Server-Sent Events) from the backend.
- `src/app/globals.css` / `index.css`: Tailwind styling and global configurations.
