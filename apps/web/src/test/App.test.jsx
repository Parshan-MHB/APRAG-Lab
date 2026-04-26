import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { App } from '../App.jsx';

const project = {
  id: 'p1',
  name: 'Local Benchmark Lab',
  source_count: 1,
  chunk_count: 2,
  run_count: 1,
  processing_status: 'processed',
  active_version: { version_number: 1 },
};

const sources = [
  {
    id: 's1',
    filename: 'auth.txt',
    source_type: 'text',
    status: 'processed',
    knowledge_base_version_id: 'version1234',
    local_path: 'projects/p1/sources/original/auth.txt',
  },
];

const sourceContent = {
  source: sources[0],
  blocks: [
    {
      id: 'b1',
      block_type: 'text',
      text: 'Authentication uses API Gateway and Token Store.',
      page_number: null,
      timestamp_start: null,
      frame_path: null,
      metadata: {},
    },
  ],
};

const runSummary = {
  id: 'r1',
  question: 'What does authentication use?',
  created_at: '2026-04-25T12:00:00Z',
  recommendation: { recommended_flow: 'traditional' },
};

const runDetail = {
  ...runSummary,
  job_status: 'succeeded',
  user_feedback: '',
  recommendation: {
    recommended_flow: 'traditional',
    reasons: ['Traditional RAG is the simplest sufficient flow for this question.'],
  },
  comparison: {
    best_pipeline: 'traditional',
    tradeoff_notes: ['All viable flows answered from similar retrieved evidence; prefer the simplest sufficient pipeline.'],
    pipelines: {
      traditional: {
        quality_label: 'usable',
        evidence_score: 0.72,
        citations_count: 1,
        visual_evidence_used: false,
        graph_evidence_used: false,
      },
      agentic: {
        quality_label: 'usable',
        evidence_score: 0.7,
        citations_count: 1,
        visual_evidence_used: false,
        graph_evidence_used: false,
      },
      hybrid_graph: {
        quality_label: 'usable',
        evidence_score: 0.7,
        citations_count: 1,
        visual_evidence_used: false,
        graph_evidence_used: true,
      },
    },
  },
  results: ['traditional', 'agentic', 'hybrid_graph'].map((pipeline) => ({
    pipeline_type: pipeline,
    answer: `Based on evidence for ${pipeline}.`,
    citations: [{ chunk_id: `${pipeline}-c1`, label: 'auth.txt#chunk-1', score: 0.8, metadata: {} }],
    trace: [{ step: 'retrieve', detail: 'Retrieved evidence.' }],
    metrics: {
      latency_seconds: 0.1,
      citations_count: 1,
      grounding_score: 0.5,
      warnings_count: 0,
    },
    techniques: ['retrieval'],
    warnings: [],
  })),
};

const runEvents = [
  { id: 'e1', event_type: 'create_run', status: 'running', message: 'started', created_at: '2026-04-25T12:00:00Z' },
  { id: 'e2', event_type: 'persist_outputs', status: 'succeeded', message: 'done', created_at: '2026-04-25T12:00:01Z' },
];

const settings = {
  llm: { provider: 'deterministic', available: true },
  vlm: { provider: 'placeholder', available: false, warning: 'Real VLM adapter not configured.' },
  embeddings: { provider: 'deterministic_lexical', available: true },
  transcription: { provider: 'placeholder', available: false },
  overrides: {},
  recommended_models: {
    profile: 'lite',
    llm_model: 'qwen3:8b',
    vlm_model: 'qwen3-vl:4b',
    embedding_model: 'bge-m3',
    models_to_pull: ['qwen3:8b', 'qwen3-vl:4b', 'bge-m3'],
  },
  ollama_model_management: {
    provider: 'ollama',
    runtime: 'host',
    reachable: true,
    configured_models: ['qwen3:8b', 'qwen3-vl:4b', 'bge-m3'],
    installed_models: ['qwen3-vl:4b', 'bge-m3'],
    missing_models: ['qwen3:8b'],
    safe_to_pull: true,
    pull_commands: ['ollama pull qwen3:8b'],
    warning: '',
  },
};

const dependencies = {
  python: { provider: 'container', available: true, version: '3.12' },
  ffmpeg: { provider: 'system-path', available: false, warning: 'Video frame/audio extraction is unavailable until FFmpeg is installed in the container.' },
  ollama: { provider: 'host-service', available: false, warning: 'Host Ollama is unavailable; install/start Ollama on the laptop and keep it listening on port 11434.' },
  disk: { provider: 'local-volume', available: true, free_bytes: 1000 },
};

const adapters = {
  vector_stores: { chroma: { provider: 'chroma' }, qdrant: { provider: 'qdrant' } },
  metadata_stores: { sqlite: { provider: 'sqlite' }, postgres: { provider: 'postgres' } },
  graph_stores: { sqlite_graph: { provider: 'sqlite_graph' }, networkx: { provider: 'networkx' } },
};

const reliability = {
  defaults: { top_k_chunks: 8, reranked_chunks: 5 },
  processing_states: ['processing', 'completed', 'failed', 'waiting_for_local_models', 'partial_success'],
  disk: { warning: false },
};

const providerHealth = {
  real_mode: true,
  ready: false,
  required_missing: ['qwen3:8b'],
  ollama: { runtime: 'host', models: { 'qwen3:8b': { installed: false } } },
};

const resourceProfile = {
  selected_profile: 'lite',
  safe_to_pull: true,
  runtime: 'host_ollama',
  warnings: ['Auto uses the lite host-Ollama profile; choose standard or high_quality only after confirming the laptop can run those models.'],
  resources: { free_disk_gib: 48, memory_gib: 16 },
  models_to_pull: ['qwen3:8b', 'qwen3-vl:4b', 'bge-m3'],
  pull_commands: ['ollama pull qwen3:8b', 'ollama pull qwen3-vl:4b', 'ollama pull bge-m3'],
};

const databaseDashboards = {
  warning: 'Dashboards expose local development data and should stay bound to localhost.',
  dashboards: [
    { id: 'qdrant', label: 'Qdrant Dashboard', url: 'http://localhost:6333/dashboard' },
    { id: 'chroma', label: 'Chroma API Docs', url: 'http://localhost:8001/docs' },
    { id: 'sqlite', label: 'SQLite Query Console', url: 'http://localhost:8000/api/database/sqlite-dashboard' },
    { id: 'redis', label: 'Redis Commander', url: 'http://localhost:8083' },
  ],
};

const observability = {
  enabled: true,
  otel_available: true,
  otlp_endpoints: ['http://jaeger:4318/v1/traces', 'http://phoenix:6006/v1/traces'],
  dashboards: [
    { id: 'jaeger', label: 'Jaeger Traces', url: 'http://localhost:16686' },
    { id: 'phoenix', label: 'Phoenix AI Observability', url: 'http://localhost:6006' },
    { id: 'langsmith', label: 'LangSmith Project', url: 'https://smith.langchain.com' },
  ],
  langsmith: { enabled: false, api_key_configured: false, project: 'APRAG-Lab' },
  phoenix: { enabled: true, collector_endpoint: 'http://phoenix:6006/v1/traces' },
};

function ok(json) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(json) });
}

beforeEach(() => {
  global.fetch = vi.fn((url, options = {}) => {
    if (url.endsWith('/health')) return ok({ status: 'ok' });
    if (url.endsWith('/api/projects') && !options.method) return ok([project]);
    if (url.endsWith('/api/projects') && options.method === 'POST') return ok(project);
    if (url.endsWith('/api/projects/p1')) return ok(project);
    if (url.endsWith('/api/projects/p1/sources')) return ok(sources);
    if (url.endsWith('/api/sources/s1/content')) return ok(sourceContent);
    if (url.endsWith('/api/projects/p1/runs') && !options.method) return ok([runSummary]);
    if (url.endsWith('/api/projects/p1/runs') && options.method === 'POST') return ok(runDetail);
    if (url.endsWith('/api/runs/r1')) return ok(runDetail);
    if (url.endsWith('/api/runs/r1/feedback')) return ok({ ...runDetail, user_feedback: JSON.parse(options.body).user_feedback });
    if (url.endsWith('/api/runs/r1/events')) return ok(runEvents);
    if (url.endsWith('/api/runs/r1/cancel')) return ok({ status: 'cancelled' });
    if (url.endsWith('/api/jobs/model-job-1')) return ok({ id: 'model-job-1', status: 'succeeded' });
    if (url.endsWith('/api/jobs/model-job-1/events')) return ok([{ id: 'm1', event_type: 'model_pull_complete', status: 'succeeded', message: 'done' }]);
    if (url.endsWith('/api/settings/models/pull')) return ok({ id: 'model-job-1', status: 'queued', job_type: 'model_pull' });
    if (url.endsWith('/api/settings/models') && options.method === 'PATCH') return ok({ ...settings, overrides: JSON.parse(options.body) });
    if (url.endsWith('/api/settings/models')) return ok(settings);
    if (url.endsWith('/api/settings/dependencies')) return ok(dependencies);
    if (url.endsWith('/api/settings/adapters')) return ok(adapters);
    if (url.endsWith('/api/settings/database-dashboards')) return ok(databaseDashboards);
    if (url.endsWith('/api/settings/observability')) return ok(observability);
    if (url.endsWith('/api/settings/reliability')) return ok(reliability);
    if (url.endsWith('/api/settings/provider-health')) return ok(providerHealth);
    if (url.endsWith('/api/settings/resource-profile')) return ok(resourceProfile);
    return ok({});
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('APRAG-Lab UI', () => {
  it('renders the required product workspace areas and accessible tabs', async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByRole('heading', { name: 'APRAG-Lab' })).toBeInTheDocument());
    expect(screen.getByText('Upload And Processing')).toBeInTheDocument();
    expect(screen.getByText('Project Sources')).toBeInTheDocument();
    expect(screen.getByText('Documents')).toBeInTheDocument();
    expect(screen.getByText('0/1 selected')).toBeInTheDocument();
    expect(screen.getByText('Source Viewer')).toBeInTheDocument();
    expect(screen.getByText('Benchmark Run')).toBeInTheDocument();
    expect(screen.getByText('Run History')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Traditional/i })).toHaveAttribute('aria-selected', 'true');
  });

  it('shows source detail and settings dependency state', async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('auth.txt');
    await user.click(screen.getByRole('button', { name: 'View' }));

    expect(screen.getByText('projects/p1/sources/original/auth.txt')).toBeInTheDocument();
    expect(screen.getByText('Extracted Evidence Blocks')).toBeInTheDocument();
    expect(screen.getByText('Authentication uses API Gateway and Token Store.')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Actions' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('link', { name: 'Qdrant Dashboard' })).toHaveAttribute('target', '_blank');
    expect(screen.getByRole('link', { name: 'Chroma API Docs' })).toHaveAttribute('href', 'http://localhost:8001/docs');
    expect(screen.getByRole('link', { name: 'SQLite Query Console' })).toHaveAttribute('href', 'http://localhost:8000/api/database/sqlite-dashboard');
    expect(screen.getByRole('link', { name: 'Redis Commander' })).toHaveAttribute('href', 'http://localhost:8083');

    await user.click(screen.getByRole('tab', { name: 'System info' }));
    expect(screen.getAllByText('placeholder').length).toBeGreaterThan(0);
    expect(screen.getByText('Real VLM adapter not configured.')).toBeInTheDocument();
    await user.click(screen.getByText('Model Recommendations'));
    expect(screen.getByText('recommended models')).toBeInTheDocument();
    expect(screen.getByText('llm: qwen3:8b')).toBeInTheDocument();
    expect(screen.getByText('overrides')).toBeInTheDocument();
    await user.click(screen.getByText('Reliability And Adapters'));
    expect(screen.getByText('8 top-k chunks')).toBeInTheDocument();
    expect(screen.getByText('chroma, qdrant')).toBeInTheDocument();
  });

  it('saves real model configuration from Settings UI', async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('Model Configuration');
    await user.click(screen.getByText('Model Configuration'));
    const llm = await screen.findByLabelText('LLM model');
    await user.clear(llm);
    await user.type(llm, 'qwen3:32b');
    await user.selectOptions(screen.getByLabelText('Vector store'), 'qdrant');
    await user.click(screen.getByRole('button', { name: /Save model settings/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/settings/models',
      expect.objectContaining({ method: 'PATCH' }),
    ));
    const call = global.fetch.mock.calls.find(([url, options]) => url.endsWith('/api/settings/models') && options?.method === 'PATCH');
    expect(JSON.parse(call[1].body)).toMatchObject({ llm_model: 'qwen3:32b', vector_store: 'qdrant' });
  });

  it('queues missing Ollama model pulls from Settings UI', async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('host Ollama models');
    expect(screen.getByText('missing: qwen3:8b')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Pull missing models/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/settings/models/pull',
      expect.objectContaining({ method: 'POST' }),
    ));
    const call = global.fetch.mock.calls.find(([url, options]) => url.endsWith('/api/settings/models/pull') && options?.method === 'POST');
    expect(JSON.parse(call[1].body)).toMatchObject({ models: ['qwen3:8b'], project_id: 'p1' });
  });

  it('opens history without rerunning and displays comparison exports', async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('What does authentication use?');
    await user.click(screen.getByRole('button', { name: /What does authentication use/i }));

    expect(await screen.findByText('Recommendation')).toBeInTheDocument();
    expect(screen.getByText('RAG Flow Queue')).toBeInTheDocument();
    expect(screen.getByText('3/3 RAG flows complete')).toBeInTheDocument();
    expect(screen.getByText('persist_outputs')).toBeInTheDocument();
    expect(screen.getByText('Quality Labels')).toBeInTheDocument();
    expect(screen.getByText('Side-by-side Answer Summaries')).toBeInTheDocument();
    expect(screen.getByText('Source Coverage')).toBeInTheDocument();
    expect(screen.getByText('JSON export')).toHaveAttribute('href', 'http://localhost:8000/api/runs/r1/export.json');
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/runs/r1',
      expect.objectContaining({ headers: expect.any(Headers) }),
    );
  });

  it('saves user feedback for an opened run', async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('What does authentication use?');
    await user.click(screen.getByRole('button', { name: /What does authentication use/i }));
    await user.type(await screen.findByLabelText('Run feedback'), 'Hybrid was easiest to trust.');
    await user.click(screen.getByRole('button', { name: /Save feedback/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/runs/r1/feedback',
      expect.objectContaining({ method: 'POST' }),
    ));
  });

  it('shows cancellable queued run progress and calls cancel endpoint', async () => {
    const user = userEvent.setup();
    const queuedRun = { ...runDetail, job_status: 'queued' };
    global.fetch = vi.fn((url, options = {}) => {
      if (url.endsWith('/health')) return ok({ status: 'ok' });
      if (url.endsWith('/api/projects') && !options.method) return ok([project]);
      if (url.endsWith('/api/projects/p1')) return ok(project);
      if (url.endsWith('/api/projects/p1/sources')) return ok(sources);
      if (url.endsWith('/api/projects/p1/runs') && !options.method) return ok([runSummary]);
      if (url.endsWith('/api/runs/r1')) return ok(queuedRun);
      if (url.endsWith('/api/runs/r1/events')) return ok([{ id: 'e1', event_type: 'queued', status: 'queued', message: 'queued' }]);
      if (url.endsWith('/api/runs/r1/cancel')) return ok({ status: 'cancelled' });
      if (url.endsWith('/api/settings/models')) return ok(settings);
      if (url.endsWith('/api/settings/dependencies')) return ok(dependencies);
      if (url.endsWith('/api/settings/adapters')) return ok(adapters);
      if (url.endsWith('/api/settings/reliability')) return ok(reliability);
      if (url.endsWith('/api/settings/provider-health')) return ok(providerHealth);
      if (url.endsWith('/api/settings/resource-profile')) return ok(resourceProfile);
      return ok({});
    });
    render(<App />);

    await screen.findByText('What does authentication use?');
    await user.click(screen.getByRole('button', { name: /What does authentication use/i }));
    expect(await screen.findByRole('button', { name: /Cancel/i })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Cancel/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/runs/r1/cancel',
      expect.objectContaining({ method: 'POST' }),
    ));
  });

  it('submits runs with selected sources and follow-up mode', async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByLabelText('Question');
    await user.click(screen.getByRole('checkbox'));
    await user.type(screen.getByLabelText('Question'), 'What does authentication use?');
    await user.click(screen.getByLabelText('Follow-up'));
    await user.selectOptions(screen.getByLabelText('Parent run'), 'r1');
    await user.click(screen.getByRole('button', { name: /Run all RAG flows/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/projects/p1/runs',
      expect.objectContaining({ method: 'POST' }),
    ));
    const call = global.fetch.mock.calls.find(([url, options]) => url.endsWith('/api/projects/p1/runs') && options?.method === 'POST');
    expect(JSON.parse(call[1].body)).toMatchObject({
      selected_sources: ['s1'],
      run_mode: 'follow_up',
      parent_run_id: 'r1',
    });
  });

  it('shows controlled API error state', async () => {
    global.fetch = vi.fn((url) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({ ok: false, json: () => Promise.resolve({ detail: 'API down' }) });
      }
      return ok({});
    });

    render(<App />);

    expect(await screen.findByRole('alert')).toHaveTextContent('API down');
  });

  it('keeps every empty tab explanatory before a run is open', async () => {
    global.fetch = vi.fn((url, options = {}) => {
      if (url.endsWith('/health')) return ok({ status: 'ok' });
      if (url.endsWith('/api/projects') && !options.method) return ok([{ ...project, run_count: 0 }]);
      if (url.endsWith('/api/projects/p1')) return ok({ ...project, run_count: 0 });
      if (url.endsWith('/api/projects/p1/sources')) return ok([]);
      if (url.endsWith('/api/projects/p1/runs')) return ok([]);
      if (url.endsWith('/api/settings/models')) return ok(settings);
      if (url.endsWith('/api/settings/dependencies')) return ok(dependencies);
      if (url.endsWith('/api/settings/adapters')) return ok(adapters);
      if (url.endsWith('/api/settings/reliability')) return ok(reliability);
      if (url.endsWith('/api/settings/provider-health')) return ok(providerHealth);
      if (url.endsWith('/api/settings/resource-profile')) return ok(resourceProfile);
      return ok({});
    });
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText('Run or reopen a benchmark to compare Traditional, Agentic, and Hybrid Graph results.');
    for (const tab of ['Agentic', 'Hybrid Graph', 'Comparison']) {
      await user.click(screen.getByRole('tab', { name: new RegExp(tab, 'i') }));
      expect(screen.getByText('Run or reopen a benchmark to compare Traditional, Agentic, and Hybrid Graph results.')).toBeInTheDocument();
    }
  });
});
