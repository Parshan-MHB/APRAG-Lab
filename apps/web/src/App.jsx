import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  ChevronRight,
  CheckCircle2,
  Database,
  FileText,
  FolderOpen,
  History,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Settings,
  XCircle,
  Upload,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const CLIENT_LOG_TO_API = import.meta.env.VITE_CLIENT_LOG_TO_API === 'true';
const CLIENT_LOG_CONSOLE = import.meta.env.MODE !== 'test' || import.meta.env.VITE_CLIENT_LOG_IN_TEST === 'true';
const PIPELINES = [
  { id: 'traditional', label: 'Traditional' },
  { id: 'agentic', label: 'Agentic' },
  { id: 'hybrid_graph', label: 'Hybrid Graph' },
  { id: 'comparison', label: 'Comparison' },
];

const SOURCE_ACTIVE_STATES = new Set(['queued', 'processing', 'running']);
const SOURCE_DONE_STATES = new Set(['processed', 'failed', 'partial_success']);
const TERMINAL_JOB_STATES = new Set(['succeeded', 'failed', 'cancelled', 'partial_success']);
const RAG_STEPS = [
  'queued',
  'queue_worker_started',
  'create_run',
  'start_traditional_pipeline',
  'start_agentic_pipeline',
  'start_hybrid_graph_pipeline',
  'collect_results',
  'compute_comparison',
  'persist_outputs',
];

function summarizeSources(sources) {
  const counts = sources.reduce((acc, source) => {
    const status = source.status || 'unknown';
    acc[status] = (acc[status] || 0) + 1;
    return acc;
  }, {});
  const total = sources.length;
  const processed = counts.processed || 0;
  const failed = counts.failed || 0;
  const done = sources.filter((source) => SOURCE_DONE_STATES.has(source.status)).length;
  const active = sources.some((source) => SOURCE_ACTIVE_STATES.has(source.status));
  return { total, processed, failed, done, active, counts };
}

function formatVersion(project, sources) {
  if (project?.active_version?.version_number) return `v${project.active_version.version_number}`;
  if (project?.active_version_id) return `pending ${project.active_version_id.slice(0, 8)}`;
  if (sources.some((source) => SOURCE_ACTIVE_STATES.has(source.status))) return 'processing';
  return '-';
}

function isProviderRecord(value) {
  return value && typeof value === 'object' && ('provider' in value || 'available' in value || 'warning' in value);
}

function formatSettingValue(value) {
  if (value === null || value === undefined || value === '') return 'none';
  if (Array.isArray(value)) return value.length ? value.join(', ') : 'none';
  if (typeof value === 'object') return Object.entries(value).map(([key, item]) => `${key}: ${item}`).join(', ') || 'none';
  return String(value);
}

function sourceGroupLabel(sourceType = 'unknown') {
  const normalized = sourceType.toLowerCase();
  if (['pdf', 'docx', 'markdown', 'text'].includes(normalized)) return 'Documents';
  if (['image', 'png', 'jpg', 'jpeg', 'webp', 'gif'].includes(normalized)) return 'Images';
  if (['audio', 'wav', 'mp3', 'm4a', 'ogg'].includes(normalized)) return 'Audio';
  if (['video', 'mp4', 'mov', 'mkv', 'webm'].includes(normalized)) return 'Video';
  if (['csv', 'json', 'table', 'data'].includes(normalized)) return 'Data';
  return 'Other';
}

function groupSources(sources) {
  return sources.reduce((groups, source) => {
    const label = sourceGroupLabel(source.source_type);
    if (!groups[label]) groups[label] = [];
    groups[label].push(source);
    return groups;
  }, {});
}

function requestId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function logClientEvent(level, eventType, payload = {}) {
  const entry = {
    created_at: new Date().toISOString(),
    level,
    event_type: eventType,
    payload,
  };
  const writer = level === 'error' ? console.error : level === 'warn' ? console.warn : console.info;
  if (CLIENT_LOG_CONSOLE) writer('[ragbench]', entry);
  if (CLIENT_LOG_TO_API && eventType !== 'frontend_log_forward_failed') {
    fetch(`${API_BASE}/api/diagnostics/frontend-event`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level, event_type: eventType, payload }),
      keepalive: true,
    }).catch((error) => {
      console.warn('[ragbench]', {
        created_at: new Date().toISOString(),
        level: 'warn',
        event_type: 'frontend_log_forward_failed',
        payload: { message: error.message },
      });
    });
  }
}

async function api(path, options = {}) {
  const id = requestId();
  const method = options.method || 'GET';
  const started = performance.now();
  const headers = new Headers(options.headers || {});
  headers.set('X-Request-ID', id);
  logClientEvent('info', 'api_request_started', { request_id: id, method, path });
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
    const elapsed_ms = Math.round(performance.now() - started);
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      logClientEvent('error', 'api_request_failed', {
        request_id: id,
        method,
        path,
        status: response.status,
        elapsed_ms,
        detail: body.detail || response.statusText,
      });
      throw new Error(body.detail || response.statusText);
    }
    logClientEvent('info', 'api_request_completed', { request_id: id, method, path, status: response.status, elapsed_ms });
    return response.json();
  } catch (error) {
    if (!error.message?.includes('HTTP')) {
      logClientEvent('error', 'api_request_error', { request_id: id, method, path, message: error.message });
    }
    throw error;
  }
}

function StatusBanner({ error, busy, status }) {
  if (error) {
    return (
      <div className="status-banner error" role="alert">
        <AlertTriangle size={18} />
        <span>{error}</span>
      </div>
    );
  }
  if (busy) {
    return (
      <div className="status-banner loading" aria-live="polite">
        <Loader2 size={18} className="spin" />
        <span>Working on the current request...</span>
      </div>
    );
  }
  return (
    <div className="status-banner ready" aria-live="polite">
      <CheckCircle2 size={18} />
      <span>{status}</span>
    </div>
  );
}

function SummaryStrip({ project, sources }) {
  const cells = [
    ['Project', project?.name || 'Loading'],
    ['Sources', project?.source_count ?? 0],
    ['Chunks', project?.chunk_count ?? 0],
    ['Runs', project?.run_count ?? 0],
    ['Version', formatVersion(project, sources)],
    ['Status', project?.processing_status || 'starting'],
  ];
  return (
    <section className="summary-strip" aria-label="Knowledge base summary">
      {cells.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}

function ExtractionProgress({ project, sources, events }) {
  const summary = summarizeSources(sources);
  if (!sources.length) {
    return <div className="empty-state">Extraction queue progress appears after upload.</div>;
  }
  const status = summary.active ? 'processing' : summary.failed ? 'partial_success' : project?.processing_status || 'processed';
  return (
    <div className="progress-card" aria-live="polite">
      <div className="result-header compact">
        <div>
          <h3>Extraction Queue</h3>
          <span>{status}</span>
        </div>
        <strong>{summary.processed}/{summary.total} documents processed</strong>
      </div>
      <div className="progress-bar" aria-label={`Extraction ${summary.done} of ${summary.total} documents complete`}>
        <span style={{ width: `${summary.total ? Math.round((summary.done / summary.total) * 100) : 0}%` }} />
      </div>
      <div className="progress-counts">
        {Object.entries(summary.counts).map(([statusName, count]) => (
          <span key={statusName}>{statusName}: {count}</span>
        ))}
      </div>
      {events.length > 0 && (
        <ol className="trace-list compact">
          {events.slice(-5).map((event) => (
            <li key={event.id || `${event.event_type}-${event.created_at}`}>
              <strong>{event.event_type}</strong>
              <span>{event.status}</span>
              <small>{event.message}</small>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function SourceList({ sources, selectedSources, onToggle, onSelectSource }) {
  const [query, setQuery] = useState('');
  if (!sources.length) {
    return <div className="empty-state">Upload files to create a reusable knowledge base.</div>;
  }
  const normalizedQuery = query.trim().toLowerCase();
  const filteredSources = normalizedQuery
    ? sources.filter((source) => (
      source.filename.toLowerCase().includes(normalizedQuery)
      || source.source_type.toLowerCase().includes(normalizedQuery)
      || source.status.toLowerCase().includes(normalizedQuery)
    ))
    : sources;
  const groups = groupSources(filteredSources);
  const orderedGroups = ['Documents', 'Images', 'Audio', 'Video', 'Data', 'Other'].filter((name) => groups[name]?.length);
  const selectedInView = filteredSources.filter((source) => selectedSources.includes(source.id)).length;
  function selectVisible() {
    filteredSources.forEach((source) => {
      if (!selectedSources.includes(source.id)) onToggle(source.id);
    });
  }
  function clearVisible() {
    filteredSources.forEach((source) => {
      if (selectedSources.includes(source.id)) onToggle(source.id);
    });
  }
  return (
    <div className="source-browser">
      <label className="search-field" htmlFor="source-search">
        <Search size={15} />
        <input
          id="source-search"
          type="search"
          placeholder="Search sources"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="source-toolbar">
        <span>{selectedInView}/{filteredSources.length} selected</span>
        <div>
          <button className="ghost compact-button" type="button" onClick={selectVisible} disabled={!filteredSources.length}>
            Select all
          </button>
          <button className="ghost compact-button" type="button" onClick={clearVisible} disabled={!selectedInView}>
            Clear
          </button>
        </div>
      </div>
      {!filteredSources.length && <div className="empty-state">No sources match the current search.</div>}
      {orderedGroups.map((groupName, groupIndex) => {
        const groupItems = groups[groupName];
        const processed = groupItems.filter((source) => source.status === 'processed').length;
        return (
          <details className="source-group" key={groupName} open={groupIndex === 0 || Boolean(normalizedQuery)}>
            <summary>
              <span className="source-group-title">
                <ChevronRight size={16} className="source-chevron" />
                <FolderOpen size={16} />
                <strong>{groupName}</strong>
              </span>
              <span>{processed}/{groupItems.length} processed</span>
            </summary>
            <div className="source-list">
              {groupItems.map((source) => (
                <div className="source-row" key={source.id}>
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedSources.includes(source.id)}
                      onChange={() => onToggle(source.id)}
                    />
                    <span>
                      <strong>{source.filename}</strong>
                      <small>{source.source_type} · {source.status} · {source.knowledge_base_version_id ? `v${source.knowledge_base_version_id.slice(0, 8)}` : 'pending'}</small>
                    </span>
                  </label>
                  <button className="ghost compact-button" type="button" onClick={() => onSelectSource(source)}>
                    View
                  </button>
                </div>
              ))}
            </div>
          </details>
        );
      })}
    </div>
  );
}

function SourceViewer({ source, sourceContent, citationResolution }) {
  if (citationResolution) {
    return (
      <div className="source-viewer" data-highlighted-citation={citationResolution.chunk_id || citationResolution.label}>
        <div><span>Citation</span><strong>{citationResolution.label}</strong></div>
        <div><span>Evidence</span><strong>{citationResolution.evidence_kind}</strong></div>
        <div><span>File</span><strong>{citationResolution.filename}</strong></div>
        {citationResolution.snapshot_restored && <div><span>Snapshot</span><strong>restored from saved run evidence</strong></div>}
        {citationResolution.page_number && <div><span>Page</span><strong>{citationResolution.page_number}</strong></div>}
        {citationResolution.timestamp_start !== null && citationResolution.timestamp_start !== undefined && <div><span>Time</span><strong>{citationResolution.timestamp_start}</strong></div>}
        {citationResolution.artifact_path && <div className="wide"><span>Artifact</span><strong>{citationResolution.artifact_path}</strong></div>}
        {citationResolution.graph_relationship && <div className="wide"><span>Relationship</span><strong>{citationResolution.graph_relationship.from} {citationResolution.graph_relationship.type} {citationResolution.graph_relationship.to}</strong></div>}
        <div className="wide"><span>Excerpt</span><pre className="answer">{citationResolution.text}</pre></div>
      </div>
    );
  }
  if (!source) {
    return <div className="empty-state">Select a source to inspect filename, status, type, and artifact reference.</div>;
  }
  return (
    <div className="source-viewer">
      <div className="detail-grid">
        <div><span>File</span><strong>{source.filename}</strong></div>
        <div><span>Type</span><strong>{source.source_type}</strong></div>
        <div><span>Status</span><strong>{source.status}</strong></div>
        <div><span>Artifact</span><strong>{source.local_path || 'not stored'}</strong></div>
        {source.error && <div className="wide"><span>Error</span><strong>{source.error}</strong></div>}
      </div>
      <h4>Extracted Evidence Blocks</h4>
      {sourceContent?.blocks?.length ? (
        <ol className="trace-list source-blocks">
          {sourceContent.blocks.map((block) => (
            <li key={block.id} data-block-id={block.id}>
              <strong>{block.block_type}</strong>
              <span>
                {block.page_number ? `page ${block.page_number}` : ''}
                {block.timestamp_start !== null && block.timestamp_start !== undefined ? ` time ${block.timestamp_start}` : ''}
                {block.frame_path ? ` frame ${block.frame_path}` : ''}
              </span>
              <pre className="answer">{block.text}</pre>
            </li>
          ))}
        </ol>
      ) : (
        <div className="empty-state">Extracted content is unavailable for this source.</div>
      )}
    </div>
  );
}

function ResultPanel({ result, recommended, runId, onResolveCitation }) {
  if (!result) {
    return <div className="empty-state">This result is not available for the selected run.</div>;
  }
  const grounding = result.metrics?.grounding_score ?? result.metrics?.retrieval_confidence ?? 0;
  return (
    <div className="result-panel">
      <div className="result-header">
        <div>
          <h3>{result.pipeline_type.replace('_', ' ')}</h3>
          <span>{result.techniques?.join(' · ') || 'No techniques recorded'}</span>
        </div>
        <div className="badges">
          {recommended && <span className="badge recommended">Recommended</span>}
          <span className="badge">{result.metrics?.latency_seconds ?? 0}s</span>
          <span className="badge">Grounding {grounding}</span>
        </div>
      </div>
      {result.warnings?.length > 0 && (
        <div className="inline-warning">
          <AlertTriangle size={16} />
          <span>{result.warnings.join(', ')}</span>
        </div>
      )}
      <section>
        <h4>Answer</h4>
        <pre className="answer">{result.answer}</pre>
      </section>
      <section>
        <h4>Citations</h4>
        <div className="citation-grid">
          {result.citations.length ? result.citations.map((citation, index) => (
            <button className="citation-card" type="button" key={`${citation.chunk_id || citation.citation || index}-${index}`} onClick={() => onResolveCitation(citation, runId)}>
              <strong>{citation.label || citation.citation || 'visual observation'}</strong>
              <span>score {citation.score ?? citation.rerank_score ?? 'n/a'}</span>
              {citation.metadata?.graph_relationship && <span>graph evidence</span>}
              {citation.metadata?.block_type && <span>{citation.metadata.block_type}</span>}
              {citation.metadata?.page_number && <span>page {citation.metadata.page_number}</span>}
              {citation.metadata?.timestamp_start !== undefined && <span>time {citation.metadata.timestamp_start}</span>}
              {citation.metadata?.frame_path && <span>frame {citation.metadata.frame_path}</span>}
            </button>
          )) : <div className="empty-state">No citations were produced.</div>}
        </div>
      </section>
      <section>
        <h4>Trace Viewer</h4>
        <ol className="trace-list">
          {result.trace.map((step, index) => (
            <li key={`${step.step}-${index}`}>
              <strong>{step.step}</strong>
              <span>{step.detail}</span>
              {step.tool && <small>tool: {step.tool}</small>}
            </li>
          ))}
        </ol>
      </section>
      <section>
        <h4>Metrics</h4>
        <div className="metric-grid">
          {Object.entries(result.metrics).map(([key, value]) => (
            <div key={key}><span>{key}</span><strong>{String(value)}</strong></div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ComparisonPanel({ run }) {
  if (!run?.comparison) {
    return <div className="empty-state">Run a benchmark to see normalized comparison metrics.</div>;
  }
  return (
    <div className="result-panel">
      <div className="result-header">
        <div>
          <h3>Recommendation</h3>
          <span>{run.recommendation.recommended_flow}</span>
        </div>
        <div className="badges">
          <span className="badge recommended">{run.comparison.best_pipeline}</span>
        </div>
      </div>
      <ul className="plain-list">
        {run.recommendation.reasons.map((reason) => <li key={reason}>{reason}</li>)}
      </ul>
      <h4>Quality Labels</h4>
      <h4>Side-by-side Answer Summaries</h4>
      <div className="comparison-grid">
        {run.results?.map((result) => (
          <div key={result.pipeline_type}>
            <strong>{result.pipeline_type.replace('_', ' ')}</strong>
            <span>{result.answer.slice(0, 180)}</span>
            <span>source coverage {result.metrics?.source_coverage_count ?? 0}</span>
          </div>
        ))}
      </div>
      <h4>Source Coverage</h4>
      <div className="comparison-grid">
        {Object.entries(run.comparison.pipelines).map(([name, data]) => (
          <div key={name}>
            <strong>{name.replace('_', ' ')}</strong>
            <span>{data.quality_label}</span>
            <span>evidence {data.evidence_score}</span>
            <span>{data.citations_count} citations</span>
            <span>{data.visual_evidence_used ? 'visual evidence used' : 'no visual evidence'}</span>
            <span>{data.graph_evidence_used ? 'graph evidence used' : 'no graph evidence'}</span>
          </div>
        ))}
      </div>
      <h4>Tradeoff Notes</h4>
      <ul className="plain-list">
        {run.comparison.tradeoff_notes.map((note) => <li key={note}>{note}</li>)}
      </ul>
      <div className="export-row">
        <a href={`${API_BASE}/api/runs/${run.id}/export.json`}>JSON export</a>
        <a href={`${API_BASE}/api/runs/${run.id}/export.md`}>Markdown export</a>
      </div>
    </div>
  );
}

function FeedbackPanel({ run, onSave }) {
  const [feedback, setFeedback] = useState(run?.user_feedback || '');
  useEffect(() => {
    setFeedback(run?.user_feedback || '');
  }, [run?.id, run?.user_feedback]);
  if (!run) {
    return <div className="empty-state">Feedback can be saved after a run is open.</div>;
  }
  return (
    <form onSubmit={(event) => { event.preventDefault(); onSave(run.id, feedback); }}>
      <label className="field-label" htmlFor="run-feedback">Run feedback</label>
      <textarea id="run-feedback" value={feedback} onChange={(event) => setFeedback(event.target.value)} placeholder="Record whether this run was useful." />
      <button type="submit">Save feedback</button>
    </form>
  );
}

function SettingsPanel({ settings, dependencies, adapters, reliability, providerHealth, resourceProfile, databaseDashboards, modelPullJob, modelPullEvents, onSaveSettings, onPullModels }) {
  const recommended = settings?.recommended_models || {};
  const overrides = settings?.overrides || {};
  const modelManagement = settings?.ollama_model_management || {};
  const missingModels = modelManagement.missing_models || [];
  const pullActive = ['queued', 'running'].includes(modelPullJob?.status);
  const [draft, setDraft] = useState({
    llm_model: settings?.overrides?.llm_model || recommended.llm_model || 'qwen3:8b',
    vlm_model: settings?.overrides?.vlm_model || recommended.vlm_model || 'qwen3-vl:4b',
    embedding_model: settings?.overrides?.embedding_model || recommended.embedding_model || 'bge-m3',
    transcription_provider: settings?.overrides?.transcription_provider || 'faster-whisper',
    transcription_model: settings?.overrides?.transcription_model || 'large-v3-turbo',
    vector_store: settings?.overrides?.vector_store || 'chroma',
    provider_mode: settings?.overrides?.provider_mode || 'real',
  });
  useEffect(() => {
    setDraft({
      llm_model: settings?.overrides?.llm_model || recommended.llm_model || 'qwen3:8b',
      vlm_model: settings?.overrides?.vlm_model || recommended.vlm_model || 'qwen3-vl:4b',
      embedding_model: settings?.overrides?.embedding_model || recommended.embedding_model || 'bge-m3',
      transcription_provider: settings?.overrides?.transcription_provider || 'faster-whisper',
      transcription_model: settings?.overrides?.transcription_model || 'large-v3-turbo',
      vector_store: settings?.overrides?.vector_store || 'chroma',
      provider_mode: settings?.overrides?.provider_mode || 'real',
    });
  }, [settings?.overrides, recommended.llm_model, recommended.vlm_model, recommended.embedding_model]);
  if (!settings && !dependencies && !adapters && !reliability && !providerHealth && !resourceProfile) {
    return <div className="empty-state">Settings are loading or the API is unavailable.</div>;
  }
  const merged = Object.fromEntries(
    Object.entries({ ...(settings || {}), ...(dependencies || {}) }).filter(([, value]) => isProviderRecord(value)),
  );
  return (
    <div>
      <div className="settings-grid">
        {Object.entries(merged).map(([name, value]) => (
          <div key={name}>
            <strong>{name}</strong>
            <span>{value.provider}</span>
            <span>{value.available ? 'available' : 'unavailable'}</span>
            {value.version && <span>{value.version}</span>}
            {value.free_bytes && <span>{value.free_bytes} bytes free</span>}
            {value.warning && <small>{value.warning}</small>}
          </div>
        ))}
      </div>
      <div className="settings-grid compact">
        <div>
          <strong>recommended models</strong>
          <span>profile: {recommended.profile || resourceProfile?.selected_profile || 'unknown'}</span>
          <span>llm: {recommended.llm_model || 'unavailable'}</span>
          <span>vlm: {recommended.vlm_model || 'unavailable'}</span>
          <span>embeddings: {recommended.embedding_model || 'unavailable'}</span>
          <small>{recommended.models_to_pull?.length ? `pull: ${recommended.models_to_pull.join(', ')}` : 'No model pull list reported.'}</small>
        </div>
        <div>
          <strong>overrides</strong>
          {Object.keys(overrides).length ? (
            Object.entries(overrides).map(([name, value]) => <span key={name}>{name}: {formatSettingValue(value)}</span>)
          ) : (
            <span>none</span>
          )}
        </div>
      </div>
      {reliability?.defaults && (
        <div className="settings-grid compact">
          <div>
            <strong>performance gates</strong>
            <span>{reliability.defaults.top_k_chunks} top-k chunks</span>
            <span>{reliability.defaults.reranked_chunks} reranked chunks</span>
            <small>{reliability.disk.warning ? 'disk warning active' : 'disk space ok'}</small>
          </div>
          <div>
            <strong>processing states</strong>
            <span>{reliability.processing_states.join(', ')}</span>
          </div>
        </div>
      )}
      {adapters && Object.keys(adapters).length > 0 && (
        <div className="settings-grid compact">
          {Object.entries(adapters).map(([group, choices]) => (
            <div key={group}>
              <strong>{group}</strong>
              <span>{Object.keys(choices).join(', ')}</span>
            </div>
          ))}
        </div>
      )}
      {databaseDashboards?.dashboards?.length > 0 && (
        <div className="settings-grid compact">
          <div>
            <strong>database dashboards</strong>
            <span>{databaseDashboards.warning}</span>
          </div>
          <div className="wide">
            <strong>open query tools</strong>
            <div className="export-row">
              {databaseDashboards.dashboards.map((dashboard) => (
                <a href={dashboard.url} key={dashboard.id} target="_blank" rel="noreferrer">
                  {dashboard.label}
                </a>
              ))}
            </div>
          </div>
        </div>
      )}
      {providerHealth && (
        <div className="settings-grid compact">
          <div>
            <strong>provider mode</strong>
            <span>{providerHealth.real_mode ? 'real mode' : 'deterministic fallback'}</span>
            <span>{providerHealth.ready ? 'ready' : 'missing provider'}</span>
            {providerHealth.required_missing?.length > 0 && <small>{providerHealth.required_missing.join(', ')}</small>}
          </div>
          <div>
            <strong>required models</strong>
            <span>{Object.entries(providerHealth.ollama?.models || {}).map(([model, info]) => `${model}:${info.installed ? 'installed' : 'missing'}`).join(', ')}</span>
          </div>
        </div>
      )}
      {resourceProfile && (
        <div className="settings-grid compact">
          <div>
            <strong>resource profile</strong>
            <span>{resourceProfile.selected_profile}</span>
            <span>{resourceProfile.safe_to_pull ? 'host Ollama pull ready' : 'model pulls blocked'}</span>
            <small>{resourceProfile.warnings?.join(', ') || 'Host Ollama manages model resources outside Docker.'}</small>
          </div>
          <div>
            <strong>app container resources</strong>
            <span>{resourceProfile.resources?.free_disk_gib} GiB disk free</span>
            <span>{resourceProfile.resources?.memory_gib ?? 'unknown'} GiB memory</span>
          </div>
          <div>
            <strong>models to pull</strong>
            <span>{resourceProfile.models_to_pull?.join(', ')}</span>
          </div>
        </div>
      )}
      <div className="settings-grid compact">
        <div>
          <strong>host Ollama models</strong>
          <span>{modelManagement.reachable ? 'reachable' : 'unreachable'}</span>
          <span>{missingModels.length ? `missing: ${missingModels.join(', ')}` : 'all configured models installed'}</span>
          {modelManagement.warning && <small>{modelManagement.warning}</small>}
        </div>
        <div>
          <strong>model pull</strong>
          <span>{pullActive ? `${modelPullJob.status}` : modelManagement.safe_to_pull ? 'ready' : 'blocked'}</span>
          <small>{modelManagement.pull_commands?.length ? modelManagement.pull_commands.join(', ') : 'No pull command needed.'}</small>
          <button type="button" disabled={!missingModels.length || !modelManagement.safe_to_pull || pullActive} onClick={() => onPullModels(missingModels)}>
            Pull missing models
          </button>
        </div>
      </div>
      {modelPullEvents?.length > 0 && (
        <ol className="trace-list compact">
          {modelPullEvents.map((event) => (
            <li key={event.id || `${event.event_type}-${event.created_at}`}>
              <strong>{event.event_type}</strong>
              <span>{event.status}</span>
              <small>{event.message}</small>
            </li>
          ))}
        </ol>
      )}
      <form className="settings-form" onSubmit={(event) => { event.preventDefault(); onSaveSettings(draft); }}>
        <label className="field-label" htmlFor="llm-model">LLM model</label>
        <input id="llm-model" value={draft.llm_model} onChange={(event) => setDraft({ ...draft, llm_model: event.target.value })} />
        <label className="field-label" htmlFor="vlm-model">VLM model</label>
        <input id="vlm-model" value={draft.vlm_model} onChange={(event) => setDraft({ ...draft, vlm_model: event.target.value })} />
        <label className="field-label" htmlFor="embedding-model">Embedding model</label>
        <input id="embedding-model" value={draft.embedding_model} onChange={(event) => setDraft({ ...draft, embedding_model: event.target.value })} />
        <label className="field-label" htmlFor="transcription-provider">Transcription provider</label>
        <select id="transcription-provider" value={draft.transcription_provider} onChange={(event) => setDraft({ ...draft, transcription_provider: event.target.value })}>
          <option value="faster-whisper">faster-whisper</option>
          <option value="whisper.cpp">whisper.cpp</option>
        </select>
        <label className="field-label" htmlFor="transcription-model">Transcription model</label>
        <input id="transcription-model" value={draft.transcription_model} onChange={(event) => setDraft({ ...draft, transcription_model: event.target.value })} />
        <label className="field-label" htmlFor="vector-store">Vector store</label>
        <select id="vector-store" value={draft.vector_store} onChange={(event) => setDraft({ ...draft, vector_store: event.target.value })}>
          <option value="chroma">Chroma</option>
          <option value="qdrant">Qdrant</option>
        </select>
        <label className="field-label" htmlFor="provider-mode">Provider mode</label>
        <select id="provider-mode" value={draft.provider_mode} onChange={(event) => setDraft({ ...draft, provider_mode: event.target.value })}>
          <option value="real">real</option>
          <option value="deterministic">deterministic</option>
        </select>
        <button type="submit"><Settings size={16} /> Save model settings</button>
      </form>
    </div>
  );
}

function RunHistory({ runs, activeRunId, onOpen }) {
  if (!runs.length) {
    return <div className="empty-state">No benchmark runs yet.</div>;
  }
  return (
    <div className="history-list">
      {runs.map((run) => (
        <button
          className={activeRunId === run.id ? 'history-row active' : 'history-row'}
          key={run.id}
          onClick={() => onOpen(run.id)}
          type="button"
        >
          <strong>{run.question}</strong>
          <span>{run.recommendation.recommended_flow} · {run.created_at ? new Date(run.created_at).toLocaleString() : 'saved run'}</span>
        </button>
      ))}
    </div>
  );
}

function JobProgress({ run, events, onCancel }) {
  if (!run?.job_status && !events.length) {
    return <div className="empty-state">Job progress appears after a run is queued or started.</div>;
  }
  const status = run?.job_status || events.at(-1)?.status || 'queued';
  const cancellable = ['queued', 'running'].includes(status);
  const completedSteps = new Set(events.map((event) => event.event_type));
  const currentIndex = Math.max(0, RAG_STEPS.findLastIndex((step) => completedSteps.has(step)) + 1);
  const resultCount = run?.results?.length || 0;
  return (
    <div className="job-progress" aria-live="polite">
      <div className="result-header compact">
        <div>
          <h3>RAG Flow Queue</h3>
          <span>{status}</span>
        </div>
        <strong>{resultCount}/3 RAG flows complete</strong>
        {cancellable && (
          <button className="ghost danger" type="button" onClick={() => onCancel(run.id)}>
            <XCircle size={16} /> Cancel
          </button>
        )}
      </div>
      <div className="progress-bar" aria-label={`RAG queue ${currentIndex} of ${RAG_STEPS.length} steps observed`}>
        <span style={{ width: `${Math.round((currentIndex / RAG_STEPS.length) * 100)}%` }} />
      </div>
      <div className="progress-counts">
        <span>traditional: {run?.traditional_result_id ? 'done' : 'waiting'}</span>
        <span>agentic: {run?.agentic_result_id ? 'done' : 'waiting'}</span>
        <span>hybrid graph: {run?.hybrid_result_id ? 'done' : 'waiting'}</span>
      </div>
      <ol className="trace-list compact">
        {events.map((event) => (
          <li key={event.id || `${event.event_type}-${event.created_at}`}>
            <strong>{event.event_type}</strong>
            <span>{event.status}</span>
            <small>{event.message}</small>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function App() {
  const [project, setProject] = useState(null);
  const [sources, setSources] = useState([]);
  const [runs, setRuns] = useState([]);
  const [settings, setSettings] = useState(null);
  const [dependencies, setDependencies] = useState(null);
  const [adapters, setAdapters] = useState(null);
  const [databaseDashboards, setDatabaseDashboards] = useState(null);
  const [reliability, setReliability] = useState(null);
  const [providerHealth, setProviderHealth] = useState(null);
  const [resourceProfile, setResourceProfile] = useState(null);
  const [activeRun, setActiveRun] = useState(null);
  const [jobEvents, setJobEvents] = useState([]);
  const [sourceJobId, setSourceJobId] = useState(null);
  const [sourceJobEvents, setSourceJobEvents] = useState([]);
  const [modelPullJob, setModelPullJob] = useState(null);
  const [modelPullEvents, setModelPullEvents] = useState([]);
  const [activeTab, setActiveTab] = useState('traditional');
  const [selectedSource, setSelectedSource] = useState(null);
  const [sourceContent, setSourceContent] = useState(null);
  const [citationResolution, setCitationResolution] = useState(null);
  const [selectedSources, setSelectedSources] = useState([]);
  const [question, setQuestion] = useState('');
  const [files, setFiles] = useState([]);
  const [uploadAction, setUploadAction] = useState('append');
  const [confirmClear, setConfirmClear] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [runMode, setRunMode] = useState('independent');
  const [parentRunId, setParentRunId] = useState('');
  const [status, setStatus] = useState('Starting');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function refreshProject(projectId = project?.id) {
    if (!projectId) return;
    logClientEvent('info', 'workspace_refresh_started', { project_id: projectId });
    const [summary, sourceList, runList, modelSettings, dependencySettings, adapterSettings, databaseDashboardSettings, reliabilitySettings, providerHealthSettings, resourceProfileSettings] = await Promise.all([
      api(`/api/projects/${projectId}`),
      api(`/api/projects/${projectId}/sources`),
      api(`/api/projects/${projectId}/runs`),
      api('/api/settings/models'),
      api('/api/settings/dependencies'),
      api('/api/settings/adapters'),
      api('/api/settings/database-dashboards'),
      api('/api/settings/reliability'),
      api('/api/settings/provider-health'),
      api('/api/settings/resource-profile'),
    ]);
    setProject(summary);
    setSources(sourceList);
    setRuns(runList);
    setSettings(modelSettings);
    setDependencies(dependencySettings);
    setAdapters(adapterSettings);
    setDatabaseDashboards(databaseDashboardSettings);
    setReliability(reliabilitySettings);
    setProviderHealth(providerHealthSettings);
    setResourceProfile(resourceProfileSettings);
    logClientEvent('info', 'workspace_refresh_completed', {
      project_id: projectId,
      source_count: summary.source_count,
      run_count: summary.run_count,
      provider_ready: providerHealthSettings.ready,
    });
  }

  async function bootstrap() {
    setError('');
    logClientEvent('info', 'app_bootstrap_started');
    try {
      await api('/health');
      const projects = await api('/api/projects');
      const currentProject = projects[0] || await api('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Local Benchmark Lab' }),
      });
      setProject(currentProject);
      await refreshProject(currentProject.id);
      setStatus('Ready');
      logClientEvent('info', 'app_bootstrap_completed', { project_id: currentProject.id });
    } catch (err) {
      setStatus('API unavailable');
      setError(err.message);
      logClientEvent('error', 'app_bootstrap_failed', { message: err.message });
    }
  }

  useEffect(() => {
    bootstrap();
  }, []);

  useEffect(() => {
    if (!activeRun?.id) {
      setJobEvents([]);
      return undefined;
    }
    let cancelled = false;
    async function refreshEvents() {
      try {
        const [events, run] = await Promise.all([
          api(`/api/runs/${activeRun.id}/events`),
          api(`/api/runs/${activeRun.id}`),
        ]);
        if (!cancelled) {
          setJobEvents(events);
          setActiveRun(run);
          if (!['queued', 'running'].includes(run.job_status) && run.project_id === project?.id) {
            await refreshProject(run.project_id);
          }
        }
      } catch {
        if (!cancelled) setJobEvents([]);
      }
    }
    refreshEvents();
    const timer = window.setInterval(refreshEvents, ['queued', 'running'].includes(activeRun.job_status) ? 1500 : 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeRun?.id, activeRun?.job_status, project?.id]);

  useEffect(() => {
    if (!project?.id) return undefined;
    const sourceSummary = summarizeSources(sources);
    const shouldPollSources = sourceSummary.active || Boolean(sourceJobId);
    if (!shouldPollSources) return undefined;
    let cancelled = false;
    async function refreshSourceProgress() {
      try {
        if (sourceJobId) {
          const events = await api(`/api/jobs/${sourceJobId}/events`);
          if (!cancelled) {
            setSourceJobEvents(events);
            const lastStatus = events.at(-1)?.status;
            if (TERMINAL_JOB_STATES.has(lastStatus)) setSourceJobId(null);
          }
        }
        if (!cancelled) await refreshProject(project.id);
      } catch (err) {
        logClientEvent('warn', 'source_progress_refresh_failed', { project_id: project.id, message: err.message });
      }
    }
    const timer = window.setInterval(refreshSourceProgress, 2500);
    refreshSourceProgress();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [project?.id, sourceJobId, sources]);

  useEffect(() => {
    if (!modelPullJob?.id || !['queued', 'running'].includes(modelPullJob.status)) return undefined;
    let cancelled = false;
    async function refreshModelPull() {
      try {
        const [job, events] = await Promise.all([
          api(`/api/jobs/${modelPullJob.id}`),
          api(`/api/jobs/${modelPullJob.id}/events`),
        ]);
        if (!cancelled) {
          setModelPullJob(job);
          setModelPullEvents(events);
          if (TERMINAL_JOB_STATES.has(job.status)) {
            await refreshProject(project?.id);
          }
        }
      } catch (err) {
        logClientEvent('warn', 'model_pull_refresh_failed', { job_id: modelPullJob.id, message: err.message });
      }
    }
    refreshModelPull();
    const timer = window.setInterval(refreshModelPull, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [modelPullJob?.id, modelPullJob?.status, project?.id]);

  const resultByType = useMemo(() => {
    const map = {};
    activeRun?.results?.forEach((result) => {
      map[result.pipeline_type] = result;
    });
    return map;
  }, [activeRun]);
  const sourceSummary = useMemo(() => summarizeSources(sources), [sources]);
  const sourceUploadActive = Boolean(sourceJobId)
    || sourceSummary.active
    || ['queued', 'processing', 'running'].includes(project?.processing_status);

  function toggleSelectedSource(sourceId) {
    setSelectedSources((current) => (
      current.includes(sourceId)
        ? current.filter((id) => id !== sourceId)
        : [...current, sourceId]
    ));
  }

  async function resolveCitation(citation, runId) {
    const params = new URLSearchParams();
    if (citation.chunk_id) params.set('chunk_id', citation.chunk_id);
    if (citation.label || citation.citation) params.set('label', citation.label || citation.citation);
    if (runId) params.set('run_id', runId);
    const resolved = await api(`/api/citations/resolve?${params.toString()}`);
    setCitationResolution(resolved);
    setSelectedSource(null);
    setSourceContent(null);
    logClientEvent('info', 'citation_resolved', { run_id: runId, chunk_id: citation.chunk_id, label: citation.label || citation.citation });
  }

  async function selectSource(source) {
    setSelectedSource(source);
    setCitationResolution(null);
    setSourceContent(null);
    try {
      const content = await api(`/api/sources/${source.id}/content`);
      setSourceContent(content);
      logClientEvent('info', 'source_selected', { source_id: source.id, filename: source.filename, block_count: content.blocks?.length || 0 });
    } catch {
      setSourceContent({ source, blocks: [] });
      logClientEvent('warn', 'source_content_unavailable', { source_id: source.id, filename: source.filename });
    }
  }

  async function uploadFiles(event) {
    event.preventDefault();
    if (!files.length || !project) return;
    setBusy(true);
    setError('');
    logClientEvent('info', 'upload_started', { project_id: project.id, file_count: files.length, upload_action: uploadAction });
    try {
      const form = new FormData();
      [...files].forEach((file) => form.append('files', file));
      form.append('upload_action', uploadAction);
      form.append('confirm_clear', String(confirmClear));
      if (newProjectName.trim()) form.append('new_project_name', newProjectName.trim());
      const response = await api(`/api/projects/${project.id}/sources`, { method: 'POST', body: form });
      setProject(response.project);
      setSourceJobId(response.job_id || null);
      setSourceJobEvents([]);
      setFiles([]);
      setConfirmClear(false);
      setNewProjectName('');
      setSelectedSources([]);
      await refreshProject(response.project.id);
      logClientEvent('info', 'upload_completed', { project_id: response.project.id, source_count: response.sources?.length || 0, job_id: response.job_id });
    } catch (err) {
      setError(err.message);
      logClientEvent('error', 'upload_failed', { project_id: project.id, message: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function runBenchmark(event) {
    event.preventDefault();
    if (!question.trim() || !project) return;
    setBusy(true);
    setError('');
    logClientEvent('info', 'benchmark_started', {
      project_id: project.id,
      selected_source_count: selectedSources.length,
      run_mode: runMode,
      question_length: question.trim().length,
    });
    try {
      const run = await api(`/api/projects/${project.id}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          selected_sources: selectedSources,
          run_mode: runMode === 'follow_up' ? 'follow_up' : 'independent',
          parent_run_id: runMode === 'follow_up' ? parentRunId || null : null,
        }),
      });
      setActiveRun(run);
      setActiveTab('comparison');
      setQuestion('');
      await refreshProject(project.id);
      logClientEvent('info', 'benchmark_completed', { project_id: project.id, run_id: run.id, recommended_flow: run.recommendation?.recommended_flow });
    } catch (err) {
      setError(err.message);
      logClientEvent('error', 'benchmark_failed', { project_id: project.id, message: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function openRun(runId) {
    setBusy(true);
    setError('');
    try {
      const run = await api(`/api/runs/${runId}`);
      setActiveRun(run);
      setActiveTab('comparison');
      setParentRunId(runId);
      logClientEvent('info', 'run_opened', { run_id: runId, recommended_flow: run.recommendation?.recommended_flow });
    } catch (err) {
      setError(err.message);
      logClientEvent('error', 'run_open_failed', { run_id: runId, message: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function cancelRun(runId) {
    setError('');
    try {
      await api(`/api/runs/${runId}/cancel`, { method: 'POST' });
      const run = await api(`/api/runs/${runId}`);
      setActiveRun(run);
      const events = await api(`/api/runs/${runId}/events`);
      setJobEvents(events);
      await refreshProject(project.id);
      logClientEvent('info', 'run_cancelled', { run_id: runId });
    } catch (err) {
      setError(err.message);
      logClientEvent('error', 'run_cancel_failed', { run_id: runId, message: err.message });
    }
  }

  async function saveModelSettings(payload) {
    setError('');
    try {
      const updated = await api('/api/settings/models', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setSettings(updated);
      setStatus('Model settings saved');
      logClientEvent('info', 'model_settings_saved', payload);
    } catch (err) {
      setError(err.message);
      logClientEvent('error', 'model_settings_save_failed', { message: err.message });
    }
  }

  async function pullMissingModels(models) {
    if (!models?.length || !project?.id) return;
    setError('');
    try {
      const job = await api('/api/settings/models/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ models, project_id: project.id }),
      });
      if (job.job_status === 'skipped') {
        setStatus('Configured models are already installed');
        await refreshProject(project.id);
        return;
      }
      const normalized = job.id ? job : { id: job.job_id, status: job.job_status || 'queued', ...job };
      setModelPullJob(normalized);
      setModelPullEvents([]);
      setStatus('Model pull queued');
      logClientEvent('info', 'model_pull_queued', { job_id: normalized.id, models });
    } catch (err) {
      setError(err.message);
      logClientEvent('error', 'model_pull_failed', { message: err.message, models });
    }
  }

  async function saveRunFeedback(runId, feedback) {
    setError('');
    try {
      const run = await api(`/api/runs/${runId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_feedback: feedback }),
      });
      setActiveRun(run);
      await refreshProject(project.id);
      logClientEvent('info', 'run_feedback_saved', { run_id: runId, feedback_length: feedback.length });
    } catch (err) {
      setError(err.message);
      logClientEvent('error', 'run_feedback_save_failed', { run_id: runId, message: err.message });
    }
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>RAGBench Studio</h1>
          <p>Local multimodal RAG benchmark workspace</p>
        </div>
        <button onClick={() => refreshProject()} disabled={busy || !project} title="Refresh workspace">
          <RefreshCw size={18} /> Refresh
        </button>
      </header>

      <StatusBanner error={error} busy={busy} status={status} />
      <SummaryStrip project={project} sources={sources} />

      <section className="workspace-layout">
        <div className="column">
          <section className="panel">
            <h2><Upload size={18} /> Upload And Processing</h2>
            <form onSubmit={uploadFiles}>
              <label className="field-label" htmlFor="source-files">Source files</label>
              <input id="source-files" type="file" multiple onChange={(event) => setFiles(event.target.files)} />
              <label className="field-label" htmlFor="upload-action">Upload behavior</label>
              <select id="upload-action" value={uploadAction} onChange={(event) => setUploadAction(event.target.value)}>
                <option value="append">Add to current knowledge base</option>
                <option value="clear_replace">Clear current data and start fresh</option>
                <option value="create_new">Create a new knowledge base</option>
              </select>
              {uploadAction === 'clear_replace' && (
                <label className="check-row">
                  <input type="checkbox" checked={confirmClear} onChange={(event) => setConfirmClear(event.target.checked)} />
                  Confirm clear and replace
                </label>
              )}
              {uploadAction === 'create_new' && (
                <>
                  <label className="field-label" htmlFor="new-project-name">New knowledge base name</label>
                  <input id="new-project-name" type="text" value={newProjectName} onChange={(event) => setNewProjectName(event.target.value)} />
                </>
              )}
              <button disabled={busy || sourceUploadActive || !files.length || (uploadAction === 'clear_replace' && !confirmClear)} type="submit">
                <Upload size={16} /> Upload
              </button>
            </form>
            <ExtractionProgress project={project} sources={sources} events={sourceJobEvents} />
          </section>

          <section className="panel">
            <h2><Database size={18} /> Project Sources</h2>
            <SourceList sources={sources} selectedSources={selectedSources} onToggle={toggleSelectedSource} onSelectSource={selectSource} />
          </section>

          <section className="panel">
            <h2><FileText size={18} /> Source Viewer</h2>
            <SourceViewer source={selectedSource} sourceContent={sourceContent} citationResolution={citationResolution} />
          </section>
        </div>

        <div className="column main-column">
          <section className="panel">
            <h2><Play size={18} /> Benchmark Run</h2>
            <form onSubmit={runBenchmark}>
              <label className="field-label" htmlFor="question">Question</label>
              <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a benchmark question about the uploaded sources." />
              <fieldset className="segmented">
                <legend>Run mode</legend>
                <label>
                  <input type="radio" name="run-mode" value="independent" checked={runMode === 'independent'} onChange={() => setRunMode('independent')} />
                  Independent
                </label>
                <label>
                  <input type="radio" name="run-mode" value="follow_up" checked={runMode === 'follow_up'} onChange={() => setRunMode('follow_up')} />
                  Follow-up
                </label>
              </fieldset>
              {runMode === 'follow_up' && (
                <>
                  <label className="field-label" htmlFor="parent-run">Parent run</label>
                  <select id="parent-run" value={parentRunId} onChange={(event) => setParentRunId(event.target.value)}>
                    <option value="">Use latest opened run</option>
                    {runs.map((run) => <option value={run.id} key={run.id}>{run.question}</option>)}
                  </select>
                </>
              )}
              <button disabled={busy || !question.trim() || !project} type="submit">
                <Play size={16} /> Run all RAG flows
              </button>
            </form>
          </section>

          <section className="panel results-panel">
            <JobProgress run={activeRun} events={jobEvents} onCancel={cancelRun} />
            <div className="tabs" role="tablist" aria-label="RAG result tabs">
              {PIPELINES.map((tab) => {
                const result = resultByType[tab.id];
                const label = activeRun?.comparison?.pipelines?.[tab.id]?.quality_label;
                const latency = result?.metrics?.latency_seconds;
                const isRecommended = activeRun?.recommendation?.recommended_flow === tab.id;
                return (
                  <button
                    aria-selected={activeTab === tab.id}
                    className={activeTab === tab.id ? 'active' : ''}
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    role="tab"
                    type="button"
                  >
                    <span>{tab.label}</span>
                    <small>{tab.id === 'comparison' ? activeRun?.comparison?.best_pipeline || 'waiting' : `${label || 'empty'} · ${latency ?? '-'}s`}</small>
                    {isRecommended && <em>recommended</em>}
                  </button>
                );
              })}
            </div>
            {!activeRun && <div className="empty-state">Run or reopen a benchmark to compare Traditional, Agentic, and Hybrid Graph results.</div>}
            {activeRun && activeTab !== 'comparison' && (
              <ResultPanel result={resultByType[activeTab]} recommended={activeRun.recommendation?.recommended_flow === activeTab} runId={activeRun.id} onResolveCitation={resolveCitation} />
            )}
            {activeRun && activeTab === 'comparison' && <ComparisonPanel run={activeRun} />}
          </section>
        </div>

        <div className="column">
          <section className="panel">
            <h2><History size={18} /> Run History</h2>
            <RunHistory runs={runs} activeRunId={activeRun?.id} onOpen={openRun} />
          </section>

          <section className="panel">
            <h2><Settings size={18} /> Settings</h2>
            <SettingsPanel
              settings={settings}
              dependencies={dependencies}
              adapters={adapters}
              databaseDashboards={databaseDashboards}
              reliability={reliability}
              providerHealth={providerHealth}
              resourceProfile={resourceProfile}
              modelPullJob={modelPullJob}
              modelPullEvents={modelPullEvents}
              onSaveSettings={saveModelSettings}
              onPullModels={pullMissingModels}
            />
          </section>

          <section className="panel">
            <h2><BarChart3 size={18} /> Grounding Notes</h2>
            <FeedbackPanel run={activeRun} onSave={saveRunFeedback} />
            {activeRun?.comparison ? (
              <ul className="plain-list">
                {activeRun.comparison.tradeoff_notes.map((note) => <li key={note}>{note}</li>)}
              </ul>
            ) : (
              <div className="empty-state">Grounding notes appear after a benchmark run.</div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}
