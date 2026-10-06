/**
 * app.js — Google Photos Memory Retrieval & Search Intelligence Engine
 * Client-side reactivity, interactive charts, real-time filtering, and grounded AI assistant.
 */

// Global State
let DATA = null;
let filteredItems = [];
let chartSourcesInstance = null;
let chartFunnelInstance = null;

// Channel color mapping matching Google branding
const SOURCE_COLORS = {
  play_store: '#4285F4', // Google Blue
  reddit: '#FF4500',     // Reddit Orange
  app_store: '#A142F4',  // Purple
  youtube: '#EA4335',    // YouTube Red
  help_forum: '#34A853', // Google Green
  web: '#24C1E0'         // Google Cyan
};

const SOURCE_LABELS = {
  play_store: 'Google Play',
  reddit: 'Reddit',
  app_store: 'App Store',
  youtube: 'YouTube',
  help_forum: 'Help Community',
  web: 'Tech Forums'
};

// Map failure point codes to readable labels
const STAGE_CODE_MAP = {
  'a': 'query_formulation',
  'b': 'ranking_precision',
  'c': 'index_parsing',
  'd': 'browse_fatigue',
  'query_formulation': 'query_formulation',
  'ranking_precision': 'ranking_precision',
  'index_parsing': 'index_parsing',
  'browse_fatigue': 'browse_fatigue'
};

const STAGE_LABELS = {
  query_formulation: 'Query Formulation',
  ranking_precision: 'Ranking / Precision',
  index_parsing: 'Index / Metadata',
  browse_fatigue: 'Browse / Fatigue'
};

// ── 1. Bootstrapping & Fetch ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupFilterHandlers();
  setupAssistant();
  setupReportActions();

  try {
    const res = await fetch('data/bundle.json');
    if (!res.ok) throw new Error('Failed to fetch bundle.json');
    DATA = await res.json();
    filteredItems = [...DATA.items];
    initializeDashboard();
  } catch (err) {
    console.warn('Direct fetch failed, checking fallback...', err);
    document.getElementById('report-content').innerHTML = `
      <div style="background:#fee2e2; border:1px solid #ef4444; border-radius:8px; padding:16px; color:#991b1b;">
        <strong>Error loading data bundle:</strong> Make sure the app is served via a web server (e.g. <code>python -m http.server 3000</code>) or deployed on Vercel.
      </div>`;
  }
});

// ── 2. Navigation Tabs ───────────────────────────────────────────────────────
function setupTabs() {
  const tabs = document.querySelectorAll('.nav-tab-item');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) targetPane.classList.add('active');

      // Refresh charts if dashboard became active
      if (tab.dataset.tab === 'tab-dashboard') {
        renderCharts();
      }
    });
  });
}

// ── 3. Initialize Dashboard & Views ─────────────────────────────────────────
function initializeDashboard() {
  updateSidebarCounts();
  updateKPICards();
  renderCharts();
  renderOpportunityCards();
  renderDataExplorer();
  renderOpportunityMatrix();
  renderSynthesisReport();
  updateAssistantEvidenceDrawer(DATA.items.slice(0, 5));
}

// ── 4. Sidebar & Filters ────────────────────────────────────────────────────
function setupFilterHandlers() {
  document.getElementById('btn-apply-filters').addEventListener('click', applyFilters);
  document.getElementById('btn-reset-filters').addEventListener('click', () => {
    document.getElementById('filter-source').value = 'ALL';
    document.getElementById('filter-stage').value = 'ALL';
    document.getElementById('filter-area').value = 'ALL';
    document.getElementById('filter-frustration').value = 'ALL';
    applyFilters();
  });

  document.getElementById('explorer-search').addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    const rows = document.querySelectorAll('#explorer-tbody tr');
    let visibleCount = 0;
    rows.forEach(row => {
      const match = !q || row.textContent.toLowerCase().includes(q);
      row.style.display = match ? '' : 'none';
      if (match) visibleCount++;
    });
    document.getElementById('explorer-counter').textContent = `Showing ${visibleCount} of ${filteredItems.length} items`;
  });
}

function applyFilters() {
  const sourceVal = document.getElementById('filter-source').value;
  const stageVal = document.getElementById('filter-stage').value;
  const areaVal = document.getElementById('filter-area').value;
  const frustVal = document.getElementById('filter-frustration').value;

  filteredItems = DATA.items.filter(item => {
    if (sourceVal !== 'ALL' && item.source !== sourceVal) return false;
    
    // Check failure points with code mapping
    if (stageVal !== 'ALL') {
      const itemStages = (item.failure_points || []).map(fp => STAGE_CODE_MAP[fp] || fp);
      if (!itemStages.includes(stageVal)) return false;
    }

    if (areaVal !== 'ALL' && !(item.assigned_areas || []).includes(areaVal)) return false;
    if (frustVal !== 'ALL' && item.frustration_level !== frustVal) return false;
    return true;
  });

  updateKPICards();
  renderCharts();
  renderDataExplorer();
}

function updateSidebarCounts() {
  const countEl = document.getElementById('sidebar-item-count');
  if (countEl && DATA) {
    countEl.textContent = DATA.items.length;
  }
}

// ── 5. KPI Cards ────────────────────────────────────────────────────────────
function updateKPICards() {
  const count = filteredItems.length;
  document.getElementById('kpi-evidence-count').textContent = count;

  // Compute failure stage breakdown
  const stageCounts = {
    'Ranking & Precision (b)': 0,
    'Query Formulation (a)': 0,
    'Browse Fatigue (d)': 0,
    'Metadata Parsing (c)': 0
  };

  filteredItems.forEach(item => {
    (item.failure_points || []).forEach(fp => {
      const canonical = STAGE_CODE_MAP[fp] || fp;
      if (canonical === 'ranking_precision' || fp === 'b') stageCounts['Ranking & Precision (b)']++;
      else if (canonical === 'query_formulation' || fp === 'a') stageCounts['Query Formulation (a)']++;
      else if (canonical === 'browse_fatigue' || fp === 'd') stageCounts['Browse Fatigue (d)']++;
      else if (canonical === 'index_parsing' || fp === 'c') stageCounts['Metadata Parsing (c)']++;
    });
  });

  const sortedStages = Object.entries(stageCounts).sort((a, b) => b[1] - a[1]);
  if (sortedStages.length > 0 && count > 0) {
    const topPct = ((sortedStages[0][1] / count) * 100).toFixed(1);
    document.getElementById('kpi-top-failure').textContent = `${topPct}%`;
  } else {
    document.getElementById('kpi-top-failure').textContent = 'N/A';
  }

  // Voiced vs Inferred (pattern_type === 'voiced' or 'voiced_need')
  const voiced = filteredItems.filter(i => i.pattern_type === 'voiced' || i.pattern_type === 'voiced_need').length;
  const voicedPct = count > 0 ? ((voiced / count) * 100).toFixed(1) : 0;
  document.getElementById('kpi-voiced-pct').textContent = `${voicedPct}%`;

  // Opportunity spaces active
  const activeSpaces = new Set();
  filteredItems.forEach(i => (i.assigned_areas || []).forEach(a => activeSpaces.add(a)));
  document.getElementById('kpi-opportunity-count').textContent = activeSpaces.size || (DATA.opportunity_areas ? DATA.opportunity_areas.length : 7);
}

// ── 6. Charts (Chart.js) ────────────────────────────────────────────────────
function renderCharts() {
  if (!window.Chart || !DATA) return;

  // Chart 1: Evidence Volume by Channel
  const sourceCounts = {};
  filteredItems.forEach(i => {
    sourceCounts[i.source] = (sourceCounts[i.source] || 0) + 1;
  });

  const sourceLabels = Object.keys(sourceCounts).map(k => SOURCE_LABELS[k] || k);
  const sourceValues = Object.values(sourceCounts);
  const sourceBackgrounds = Object.keys(sourceCounts).map(k => SOURCE_COLORS[k] || '#1a73e8');

  const ctxSources = document.getElementById('chart-sources');
  if (ctxSources) {
    if (chartSourcesInstance) chartSourcesInstance.destroy();
    chartSourcesInstance = new Chart(ctxSources, {
      type: 'bar',
      data: {
        labels: sourceLabels,
        datasets: [{
          data: sourceValues,
          backgroundColor: sourceBackgrounds,
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.raw} Customer Signals (${((ctx.raw / (filteredItems.length || 1)) * 100).toFixed(1)}%)`
            }
          }
        },
        scales: {
          x: { grid: { color: '#f1f5f9' }, ticks: { stepSize: 1, color: '#64748b' } },
          y: { grid: { display: false }, ticks: { font: { weight: 600 }, color: '#334155' } }
        }
      }
    });
  }

  // Chart 2: Failure Funnel (Doughnut)
  const fpCounts = {
    'Ranking & Precision': 0,
    'Query Formulation': 0,
    'Browse Fatigue': 0,
    'Index & Metadata': 0
  };

  filteredItems.forEach(item => {
    (item.failure_points || []).forEach(fp => {
      const canonical = STAGE_CODE_MAP[fp] || fp;
      if (canonical === 'ranking_precision' || fp === 'b') fpCounts['Ranking & Precision']++;
      else if (canonical === 'query_formulation' || fp === 'a') fpCounts['Query Formulation']++;
      else if (canonical === 'browse_fatigue' || fp === 'd') fpCounts['Browse Fatigue']++;
      else if (canonical === 'index_parsing' || fp === 'c') fpCounts['Index & Metadata']++;
    });
  });

  const ctxFunnel = document.getElementById('chart-funnel');
  if (ctxFunnel) {
    if (chartFunnelInstance) chartFunnelInstance.destroy();
    chartFunnelInstance = new Chart(ctxFunnel, {
      type: 'doughnut',
      data: {
        labels: Object.keys(fpCounts),
        datasets: [{
          data: Object.values(fpCounts),
          backgroundColor: ['#EA4335', '#4285F4', '#34A853', '#FBBC04'],
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '62%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 12, padding: 14, font: { size: 11, weight: 600 }, color: '#475569' }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.label}: ${ctx.raw} occurrences`
            }
          }
        }
      }
    });
  }
}

// ── 7. Opportunity Spaces 3-Column Card Grid ────────────────────────────────
function renderOpportunityCards() {
  const container = document.getElementById('opp-cards-container');
  if (!container || !DATA || !DATA.opportunity_areas) return;

  container.innerHTML = DATA.opportunity_areas.map((opp, idx) => {
    const name = opp["Opportunity Area"] || opp.name || `Opportunity Area ${idx + 1}`;
    const desc = opp["Description"] || opp.problem_hypothesis || opp.description || '';
    const evidenceCount = opp["Evidence Volume"] || opp.evidence_count || (opp.signals ? opp.signals.length : 0);
    const severity = opp["Severity Score"] || opp.avg_frustration || 2.5;

    const frustClass = severity >= 2.8 ? 'badge-high' : (severity >= 1.8 ? 'badge-med' : 'badge-low');
    const frustLabel = severity >= 2.8 ? 'High Frustration' : (severity >= 1.8 ? 'Medium' : 'Low');

    // Extract top quotes
    const quotesList = opp["Top Quotes"] || opp.representative_quotes || [];
    const quotesHtml = quotesList.slice(0, 2).map(q => {
      const qText = typeof q === 'string' ? q : (q.text || '');
      return `
        <div style="margin-top:10px; padding:10px 12px; background:#f8fafc; border-left:3px solid #1a73e8; border-radius:0 6px 6px 0; font-size:0.8rem; color:#475569; font-style:italic;">
          "${escapeHtml(qText)}"
        </div>
      `;
    }).join('');

    return `
      <div class="opp-card">
        <div>
          <div class="opp-header">
            <span class="rank-pill">#${opp.rank || (idx + 1)}</span>
            <span class="badge-pill ${frustClass}">${frustLabel} (${severity}/3)</span>
          </div>
          <h3 class="opp-title">${escapeHtml(name)}</h3>
          <p class="opp-desc">${escapeHtml(desc)}</p>
        </div>
        <div>
          ${quotesHtml}
          <div class="opp-meta">
            <span><strong>${evidenceCount}</strong> Customer Signals</span>
            <span>Blocker: <strong style="color:#0f172a;">${opp.primary_failure_point ? (STAGE_LABELS[STAGE_CODE_MAP[opp.primary_failure_point] || opp.primary_failure_point] || opp.primary_failure_point) : 'Cognitive Recall Gap'}</strong></span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// ── 8. Data Explorer Table & CSV Export ──────────────────────────────────────
function renderDataExplorer() {
  const tbody = document.getElementById('explorer-tbody');
  const counter = document.getElementById('explorer-counter');
  if (!tbody || !DATA) return;

  counter.textContent = `Showing ${filteredItems.length} of ${DATA.items.length} items`;

  tbody.innerHTML = filteredItems.map(item => {
    const badgeClass = `badge-${item.source}`;
    const sourceLabel = SOURCE_LABELS[item.source] || item.source;

    const fps = (item.failure_points || []).map(fp => {
      const canonical = STAGE_CODE_MAP[fp] || fp;
      return `
        <span class="badge-pill" style="background:#f1f5f9; color:#475569; font-size:0.72rem; margin-right:4px;">
          ${STAGE_LABELS[canonical] || canonical}
        </span>
      `;
    }).join('');

    const frustClass = item.frustration_level === 'high' ? 'badge-high' : (item.frustration_level === 'med' ? 'badge-med' : 'badge-low');

    const spaces = (item.assigned_areas || []).map(a => `
      <span style="display:inline-block; font-size:0.75rem; color:#1a73e8; font-weight:600; margin-bottom:2px;">
        • ${escapeHtml(a)}
      </span><br>
    `).join('');

    const link = item.url ? `<a href="${item.url}" target="_blank" rel="noopener noreferrer" style="color:#1a73e8; font-weight:600; text-decoration:none;">View ↗</a>` : '—';

    return `
      <tr>
        <td><span class="badge-pill ${badgeClass}">${sourceLabel}</span></td>
        <td style="line-height:1.45; font-size:0.86rem; color:#1e293b;">
          ${escapeHtml(item.text)}
        </td>
        <td>${fps || '—'}</td>
        <td><span class="badge-pill ${frustClass}">${(item.frustration_level || 'med').toUpperCase()}</span></td>
        <td>${spaces || 'Unassigned'}</td>
        <td>${link}</td>
      </tr>
    `;
  }).join('');
}

// CSV Export
document.getElementById('btn-export-csv').addEventListener('click', () => {
  if (!filteredItems.length) return;
  const headers = ['ID', 'Channel', 'Customer Feedback', 'Failure Points', 'Frustration', 'Assigned Spaces', 'URL'];
  const rows = filteredItems.map(i => [
    i.id,
    SOURCE_LABELS[i.source] || i.source,
    `"${(i.text || '').replace(/"/g, '""')}"`,
    `"${(i.failure_points || []).map(fp => STAGE_LABELS[STAGE_CODE_MAP[fp] || fp] || fp).join('; ')}"`,
    i.frustration_level || 'med',
    `"${(i.assigned_areas || []).join('; ')}"`,
    i.url || ''
  ]);

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `google_photos_discovery_signals_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// ── 9. Opportunity Matrix ───────────────────────────────────────────────────
function renderOpportunityMatrix() {
  const tbody = document.getElementById('matrix-tbody');
  const gridContainer = document.getElementById('breakdown-matrix-grid');
  if (!tbody || !DATA || !DATA.opportunity_areas) return;

  tbody.innerHTML = DATA.opportunity_areas.map((opp, idx) => {
    const name = opp["Opportunity Area"] || opp.name || `Opportunity Area ${idx + 1}`;
    const desc = opp["Description"] || opp.problem_hypothesis || opp.description || '';
    const evidenceCount = opp["Evidence Volume"] || opp.evidence_count || 0;
    const severity = opp["Severity Score"] || opp.avg_frustration || 2.5;
    const blocker = opp.primary_failure_point ? (STAGE_LABELS[STAGE_CODE_MAP[opp.primary_failure_point] || opp.primary_failure_point] || opp.primary_failure_point) : 'Episodic Recall Gap';

    const frustClass = severity >= 2.8 ? 'badge-high' : 'badge-med';
    return `
      <tr>
        <td><span class="rank-pill">#${opp.rank || (idx + 1)}</span></td>
        <td><strong style="color:#0f172a; font-family:'Google Sans',sans-serif;">${escapeHtml(name)}</strong></td>
        <td style="color:#475569; font-size:0.84rem; line-height:1.45;">${escapeHtml(desc)}</td>
        <td><strong>${evidenceCount}</strong> signals</td>
        <td><span class="badge-pill ${frustClass}">${severity}/3</span></td>
        <td><span style="color:#1a73e8; font-weight:600;">${blocker}</span></td>
      </tr>
    `;
  }).join('');

  // 4 Stage Breakdown Cards
  const stageStats = [
    { title: 'Ranking & Precision', count: '25 signals (80.6%)', desc: 'Over-broad search returns zero or irrelevant false positives without precision ranking', color: 'var(--gp-red)' },
    { title: 'Query Formulation', count: '13 signals (41.9%)', desc: 'Users lack vocabulary to articulate fuzzy emotional, temporal, or spatial cues into search queries', color: 'var(--gp-blue)' },
    { title: 'Browse Fatigue', count: '12 signals (38.7%)', desc: 'Infinite scroll abandonment after 500+ photos without visual chronological landmarks', color: 'var(--gp-green)' },
    { title: 'Metadata Parsing', count: '8 signals (25.8%)', desc: 'Stripped EXIF, lost WhatsApp timestamps, and misplaced cloud sync metadata', color: 'var(--gp-yellow)' }
  ];

  gridContainer.innerHTML = stageStats.map(s => `
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-top:4px solid ${s.color}; border-radius:8px; padding:16px;">
      <div style="font-weight:700; color:#0f172a; margin-bottom:4px; font-family:'Google Sans',sans-serif;">${s.title}</div>
      <div style="font-size:0.8rem; font-weight:700; color:${s.color}; margin-bottom:8px;">${s.count}</div>
      <p style="font-size:0.78rem; color:#64748b; line-height:1.4;">${s.desc}</p>
    </div>
  `).join('');
}

// ── 10. AI Discovery Assistant v2 — Live Groq Streaming ──────────────────────
let activeChatApi = '/api';
let conversationHistory = [];   // [{role, content}]
let isStreaming = false;

async function resolveChatApi() {
  // Test relative /api first (Vercel production / Vercel CLI dev)
  try {
    const r = await fetch('/api/health', { signal: AbortSignal.timeout(1800) });
    if (r.ok) {
      activeChatApi = '/api';
      return { ok: true, data: await r.json() };
    }
  } catch {}

  // Fallback for local python dual-server dev (port 8001)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    const localEndpoints = ['http://localhost:8001/api', 'http://localhost:8001'];
    for (const ep of localEndpoints) {
      try {
        const r = await fetch(`${ep}/health`, { signal: AbortSignal.timeout(1800) });
        if (r.ok) {
          activeChatApi = ep;
          return { ok: true, data: await r.json() };
        }
      } catch {}
    }
  }

  return { ok: false };
}

function setupAssistant() {
  // Inject welcome message
  renderWelcome();

  // Chip clicks
  document.querySelectorAll('.chip-btn').forEach(chip => {
    chip.addEventListener('click', () => {
      const p = chip.dataset.prompt;
      if (p && !isStreaming) sendMessage(p);
    });
  });

  // Form submit
  const form  = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');

  form.addEventListener('submit', e => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || isStreaming) return;
    input.value = '';
    resizeTextarea(input);
    sendMessage(q);
  });

  // Enter to send (shift+enter = newline)
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event('submit'));
    }
  });

  // Auto-resize textarea
  input.addEventListener('input', () => resizeTextarea(input));

  // Clear chat
  document.getElementById('btn-clear-chat').addEventListener('click', () => {
    conversationHistory = [];
    const thread = document.getElementById('chat-thread');
    thread.innerHTML = '';
    renderWelcome();
    updateEvidenceDrawer([]);
  });

  // FAB click → switch to assistant tab
  const fab = document.getElementById('chat-fab');
  if (fab) {
    fab.addEventListener('click', () => {
      document.querySelectorAll('.nav-tab-item').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      const tabBtn = document.getElementById('tabnav-assistant');
      const tabPane = document.getElementById('tab-assistant');
      if (tabBtn) tabBtn.classList.add('active');
      if (tabPane) tabPane.classList.add('active');
    });
  }

  // Check API health
  checkApiHealth();
}

function resizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

async function checkApiHealth() {
  const lbl = document.getElementById('api-status-label');
  const badge = document.getElementById('chat-mode-badge');
  const status = await resolveChatApi();
  if (status.ok) {
    if (lbl) { lbl.textContent = 'Connected ✓'; lbl.className = 'api-status ok'; }
    if (badge) { badge.textContent = '● Live AI'; badge.className = 'mode-badge mode-live'; }
  } else {
    if (lbl) { lbl.textContent = 'Offline — static mode'; lbl.className = 'api-status err'; }
    if (badge) { badge.textContent = '○ Static'; badge.className = 'mode-badge mode-offline'; }
  }
}

function renderWelcome() {
  const thread = document.getElementById('chat-thread');
  const count = DATA ? DATA.items.length : '—';
  thread.insertAdjacentHTML('beforeend', `
    <div class="welcome-wrap">
      <div class="welcome-icon">✦</div>
      <div class="welcome-title">AI Discovery Assistant</div>
      <div class="welcome-sub">
        I'm grounded on <strong>${count} verified customer signals</strong> from Google Play Store, Reddit, Apple App Store, YouTube, and Help Forums.<br><br>
        Ask me anything — about the data, user behavior, search pain points, or product strategy. Click a prompt chip to get started!
      </div>
    </div>`);
}

// ── Send a message ────────────────────────────────────────────────────────────
async function sendMessage(query) {
  if (isStreaming) return;
  isStreaming = true;

  const thread  = document.getElementById('chat-thread');
  const sendBtn = document.getElementById('btn-send');
  if (sendBtn) sendBtn.disabled = true;

  // Remove welcome wrap if present
  const welcome = thread.querySelector('.welcome-wrap');
  if (welcome) welcome.remove();

  // Append user bubble
  appendUserBubble(query);
  conversationHistory.push({ role: 'user', content: query });

  // Update evidence drawer optimistically
  const matched = findMatchingQuotes(query.split(/\s+/).filter(w => w.length > 3));
  updateEvidenceDrawer(matched);

  // Show typing indicator
  const typingEl = appendTypingIndicator();

  try {
    const apiOk = await pingApi();
    if (apiOk) {
      await streamFromApi(query, typingEl);
    } else {
      // Fallback to local static synthesis
      typingEl.remove();
      const result = PRESET_SYNTHESIS[query] || synthesizeCustomQuery(query);
      const botEl = appendBotBubble('', false);
      botEl.querySelector('.msg-bubble').innerHTML = window.marked
        ? marked.parse(result.response) : result.response;
      conversationHistory.push({ role: 'assistant', content: result.response });
    }
  } catch (err) {
    typingEl.remove();
    appendBotBubble(`⚠️ Error: ${err.message || 'Something went wrong.'}`, false, true);
  } finally {
    isStreaming = false;
    if (sendBtn) sendBtn.disabled = false;
    thread.scrollTop = thread.scrollHeight;
  }
}

async function pingApi() {
  const res = await resolveChatApi();
  return res.ok;
}

// ── Streaming from Groq API ───────────────────────────────────────────────────
async function streamFromApi(query, typingEl) {
  const thread = document.getElementById('chat-thread');
  const res = await fetch(`${activeChatApi}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: conversationHistory })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${res.status}`);
  }

  typingEl.remove();
  const botEl = appendBotBubble('', true);   // streaming=true → adds cursor
  const bubble = botEl.querySelector('.msg-bubble');
  let fullText = '';

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();   // keep incomplete last line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const payload = JSON.parse(line.slice(6));
        if (payload.error) throw new Error(payload.error);
        if (payload.done) {
          bubble.classList.remove('stream-cursor');
          bubble.innerHTML = window.marked ? marked.parse(fullText) : fullText;
          conversationHistory.push({ role: 'assistant', content: fullText });
          return;
        }
        if (payload.t) {
          fullText += payload.t;
          bubble.textContent = fullText;
          thread.scrollTop = thread.scrollHeight;
        }
      } catch (parseErr) {
        if (parseErr.message !== 'JSON') throw parseErr;
      }
    }
  }

  // EOF without done flag
  bubble.classList.remove('stream-cursor');
  bubble.innerHTML = window.marked ? marked.parse(fullText) : fullText;
  conversationHistory.push({ role: 'assistant', content: fullText });
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function appendUserBubble(text) {
  const thread = document.getElementById('chat-thread');
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML = `
    <div class="msg-bubble">${escapeHtml(text)}</div>
    <div class="msg-meta">${now}</div>`;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

function appendBotBubble(text, streaming = false, isError = false) {
  const thread = document.getElementById('chat-thread');
  const el = document.createElement('div');
  el.className = `msg msg-bot${isError ? ' msg-error' : ''}`;
  el.innerHTML = `
    <div class="msg-bubble${streaming ? ' stream-cursor' : ''}">${text}</div>
    <div class="msg-meta">AI Discovery Assistant</div>`;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

function appendTypingIndicator() {
  const thread = document.getElementById('chat-thread');
  const el = document.createElement('div');
  el.className = 'msg msg-bot';
  el.innerHTML = `<div class="typing-indicator">
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  </div>`;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

// ── Evidence Drawer ───────────────────────────────────────────────────────────
function findMatchingQuotes(keywords) {
  if (!DATA || !DATA.items) return [];
  const kws = keywords.map(k => k.toLowerCase()).filter(k => k.length > 2);
  const scored = DATA.items.map(item => {
    let score = 0;
    const txt = (item.text || '').toLowerCase();
    kws.forEach(kw => { if (txt.includes(kw)) score += 3; });
    (item.remembered_cues || []).forEach(c => { if (kws.includes(c)) score += 2; });
    return { item, score };
  });
  return scored.sort((a,b) => b.score - a.score).slice(0, 5).map(s => s.item);
}

function updateEvidenceDrawer(items) {
  const container = document.getElementById('evidence-quotes-container');
  const badge     = document.getElementById('evidence-count-badge');
  if (!container) return;

  if (badge) badge.textContent = `${items.length} matched`;

  if (!items.length) {
    container.innerHTML = '<div class="ev-placeholder">No matching signals for this query.</div>';
    return;
  }

  container.innerHTML = items.map(item => {
    const src  = SOURCE_LABELS[item.source] || item.source;
    const frust = (item.frustration_level || 'med').toUpperCase();
    const frustClass = item.frustration_level === 'high' ? 'badge-high' : 'badge-med';
    const link = item.url
      ? `<a href="${item.url}" target="_blank" rel="noopener noreferrer" style="color:#1a73e8;font-weight:600;text-decoration:none;">Source ↗</a>`
      : '';
    return `<div class="ev-card">
      <div class="ev-card-quote">"${escapeHtml(item.text)}"</div>
      <div class="ev-card-meta">
        <span class="badge-pill badge-${item.source}">${src}</span>
        <span class="badge-pill ${frustClass}">${frust}</span>
        ${link}
      </div>
    </div>`;
  }).join('');
}

// ── Static fallback knowledge base (used when API is offline) ─────────────────


// Grounded Knowledge Base Answers for core queries
const PRESET_SYNTHESIS = {
  "What kinds of old photos do users struggle to retrieve?": {
    response: `### 1. Categories of Old Photos Users Struggle to Retrieve

Based on our analysis of **31 grounded customer signals**, users face severe retrieval friction across 4 distinct photo categories:

| Category | Primary Friction Point | Evidence Frequency |
|---|---|---|
| **Screenshots & Receipts** | Utility documents polluting family memories; OCR misses handwritten text or blurry images. | 22.6% (7 signals) |
| **Specific Emotional Moments** | Weddings, vacations, deceased pets where exact month/year is forgotten. | 38.7% (12 signals) |
| **Object & Context Detail** | Looking for a specific car, dress, wallpaper, or document without a date anchor. | 29.0% (9 signals) |
| **WhatsApp & Cross-Device Media** | EXIF timestamps wiped on import, lumping thousands of photos onto the import date. | 16.1% (5 signals) |

> *"I was looking for a picture of my dog from 5 years ago wearing a red bandana. Search returned 400 dog photos but none with the bandana."* — Reddit User

> *"Every time I take a screenshot of a recipe or receipt, it clutters my main camera roll. Then when I actually need the receipt 6 months later, search finds nothing."* — Play Store Review`,
    keywords: ["dog", "screenshot", "receipt", "wedding", "pet", "bandana", "date", "trip"]
  },

  "What information do people actually remember about a photo?": {
    response: `### 2. Cognitive Memory: What People Actually Remember

When human memory retrieves a past moment, it relies on **episodic anchors** rather than metadata indexes:

* **People Present (51.6% of requests):** Who was in the frame (*"me and my brother"*, *"grandma at the hospital"*).
* **Life Stage & Coarse Time (48.4% of requests):** Relative time rather than calendar dates (*"when I lived in Boston"*, *"college freshman year"*, *"last summer"*).
* **Setting & Environment (41.9% of requests):** Visual environment (*"on the beach at sunset"*, *"snowy backyard"*, *"in front of the brick house"*).
* **Prominent Objects & Actions (38.7% of requests):** Notable focal items (*"holding a yellow balloon"*, *"cutting birthday cake"*, *"driving the old blue Civic"*).

#### Cognitive Recall Hierarchy:
1. **Who:** Social companions (Highest retention)
2. **Where:** Coarse geographic or atmospheric setting
3. **What:** Salient color, action, or object
4. **When:** Coarse season / life chapter (Exact calendar date is lost)`,
    keywords: ["people", "remember", "brother", "summer", "yellow", "balloon", "cake", "beach"]
  },

  "What information have they forgotten?": {
    response: `### 3. What Information Have Users Forgotten?

The fundamental breakdown between user mental models and Google Photos search is **the metadata gap**:

* **Exact Calendar Dates (80.6% forgotten):** Users never remember if a photo was taken on May 14th vs. October 22nd. Yet the chronological timeline forces date-based navigation.
* **Album Names & Filenames (32.3% forgotten):** Users do not remember IMG_4912.jpg, nor do they maintain manual curated albums.
* **Exact Location / Coordinates (12.9% forgotten):** They remember *"a hike with waterfalls"*, but not the GPS coordinate or county name.
* **Keywords Google's Model Recognizes (12.9% forgotten):** They struggle to guess the semantic label used by computer vision classifiers.

> *"I remember it was a rainy afternoon in my college apartment, but Google Photos only lets me scroll through years I don't know."* — Play Store User`,
    keywords: ["forgotten", "date", "album", "scroll", "timeline", "apartment", "college"]
  },

  "How do users formulate searches when their memory is incomplete?": {
    response: `### 4. Search Formulation Behavior Under Incomplete Memory

When users cannot formulate an exact query, they exhibit predictable behavioral patterns:

1. **Keyword Guess-and-Check:** Trying combinations like *"mom blue dress"*, then *"mom outdoor"*, then giving up.
2. **Coarse-to-Fine Degradation:** Starting with a keyword search, seeing 300 irrelevant results, then reverting to painful manual timeline scrolling.
3. **Over-Constraining / Under-Constraining:** Either adding too many words that yield zero results, or a single broad word (*"car"*) that returns thousands.
4. **Negative Filtering Need:** Users repeatedly express wanting to say *"pictures of Sarah WITHOUT John"*, or *"vacation NOT at home"*. Currently, Google Photos lacks negative operator support.

> *"Search is either all or nothing. If I type 'beach sunset mom', it shows every beach photo ever taken even if mom isn't in it."* — Help Forum Signal`,
    keywords: ["search", "formulation", "beach", "sunset", "mom", "negative", "scroll"]
  },

  "Why do screenshots and utility photos pollute the library?": {
    response: `### 5. The Utilitarian Photo Pollution Problem

Screenshots, receipts, parking spot snapshots, and serial numbers represent **high-urgency, low-sentiment utilitarian media**.

* **Root Problem:** Google Photos syncs the Android/iOS screenshot folder alongside irreplaceable family portraits.
* **Retrieval Breakdown:**
  * When browsing memories, utility photos ruin the nostalgic experience.
  * When searching for a document (*"insurance policy"* or *"parking ticket"*), search fails to isolate the relevant screenshot from 4,000 others.
* **User Wish:** Automatic isolation into a temporary utility vault with auto-expiration (e.g., 30-day auto-archive) and dedicated OCR-first retrieval.`,
    keywords: ["screenshot", "utility", "receipt", "document", "parking", "clutter", "camera roll"]
  },

  "What makes users abandon search and give up looking?": {
    response: `### 6. App Abandonment & Search Surrender

In **41.9% of analyzed user complaints**, retrieval failure resulted in complete session abandonment:

* **Infinite Scroll Fatigue:** After 3 minutes of aggressive swipe-scrolling through thousands of repetitive burst shots, users suffer cognitive exhaustion.
* **Zero-Result Wall:** When a query yields *"No results found"*, the engine offers no spelling suggestions, related tags, or semantic alternatives.
* **Platform Churn:** Frustrated users report switching to manual folder storage (Google Drive, Dropbox, local external hard drives) or Apple Photos.

> *"I spent 20 minutes scrolling trying to find my daughter's first haircut. I got so frustrated I just gave up and asked my wife to find it on her iPhone."* — Play Store Review`,
    keywords: ["abandon", "give up", "scrolling", "fatigue", "haircut", "daughter", "iphone"]
  }
};




// ── 11. Synthesis Report View & PDF Print ───────────────────────────────────
function renderSynthesisReport() {
  const reportContainer = document.getElementById('report-content');
  if (!reportContainer || !DATA || !DATA.synthesis_report) return;

  if (window.marked) {
    reportContainer.innerHTML = marked.parse(DATA.synthesis_report);
  } else {
    reportContainer.textContent = DATA.synthesis_report;
  }
}

function setupReportActions() {
  document.getElementById('btn-download-md').addEventListener('click', () => {
    if (!DATA || !DATA.synthesis_report) return;
    const blob = new Blob([DATA.synthesis_report], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `google_photos_synthesis_report_${new Date().toISOString().slice(0, 10)}.md`;
    link.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById('btn-print-report').addEventListener('click', () => {
    window.print();
  });
}

// ── Helper Utilities ────────────────────────────────────────────────────────
function escapeHtml(str) {
  if (!str) return '';
  return str.toString()
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
