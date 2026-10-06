// AutoOptML Frontend Client Logic with User Authentication & SQLite Database History

// Global state
const state = {
  sessionId: null,
  datasetName: '',
  profile: null,
  optimizationResult: null,
  benchmarkResults: null,
  currentStep: 1,
  currentUser: null,
  token: localStorage.getItem('autooptml_token') || null,
  charts: {}
};

// DOM ready
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  initAuthEventListeners();
  initAppEventListeners();
  checkAuthStatus();
  loadSampleDatasets();
});

// Helper for authenticated fetch
async function authFetch(url, options = {}) {
  const headers = options.headers || {};
  if (state.token) {
    headers['Authorization'] = `Bearer ${state.token}`;
  }
  options.headers = headers;
  return fetch(url, options);
}

// ----------------------------------------------------
// Authentication Handlers & UI Updates
// ----------------------------------------------------
function initAuthEventListeners() {
  const modalAuth = document.getElementById('modal-auth');
  const btnOpenLogin = document.getElementById('btn-open-login');
  const btnCloseAuth = document.getElementById('btn-close-auth');
  const btnQuickDemo = document.getElementById('btn-quick-demo');
  const btnModalDemo = document.getElementById('btn-modal-demo-login');
  const btnLogout = document.getElementById('btn-logout');

  btnOpenLogin.addEventListener('click', () => {
    switchAuthTab('login');
    modalAuth.classList.remove('hidden');
  });

  btnCloseAuth.addEventListener('click', () => modalAuth.classList.add('hidden'));

  // Quick Demo Logins
  btnQuickDemo.addEventListener('click', performDemoLogin);
  btnModalDemo.addEventListener('click', performDemoLogin);

  btnLogout.addEventListener('click', performLogout);

  // Tab switching inside auth modal
  const tabLogin = document.getElementById('tab-login');
  const tabRegister = document.getElementById('tab-register');
  tabLogin.addEventListener('click', () => switchAuthTab('login'));
  tabRegister.addEventListener('click', () => switchAuthTab('register'));

  // Form Submissions
  document.getElementById('form-login').addEventListener('submit', handleLoginForm);
  document.getElementById('form-register').addEventListener('submit', handleRegisterForm);

  // Experiments Drawer Handlers
  const modalExperiments = document.getElementById('modal-experiments');
  document.getElementById('btn-open-experiments').addEventListener('click', () => {
    loadUserExperiments();
    modalExperiments.classList.remove('hidden');
  });
  document.getElementById('btn-close-experiments').addEventListener('click', () => modalExperiments.classList.add('hidden'));
  document.getElementById('btn-experiments-done').addEventListener('click', () => modalExperiments.classList.add('hidden'));
}

function switchAuthTab(tab) {
  const tabLogin = document.getElementById('tab-login');
  const tabRegister = document.getElementById('tab-register');
  const formLogin = document.getElementById('form-login');
  const formRegister = document.getElementById('form-register');
  const modalTitle = document.getElementById('auth-modal-title');

  document.getElementById('login-error').classList.add('hidden');
  document.getElementById('reg-error').classList.add('hidden');

  if (tab === 'login') {
    tabLogin.classList.add('bg-white', 'text-black', 'shadow-sm');
    tabLogin.classList.remove('text-zinc-500');
    tabRegister.classList.remove('bg-white', 'text-black', 'shadow-sm');
    tabRegister.classList.add('text-zinc-500');
    formLogin.classList.remove('hidden');
    formRegister.classList.add('hidden');
    modalTitle.textContent = 'Sign In to AutoOptML';
  } else {
    tabRegister.classList.add('bg-white', 'text-black', 'shadow-sm');
    tabRegister.classList.remove('text-zinc-500');
    tabLogin.classList.remove('bg-white', 'text-black', 'shadow-sm');
    tabLogin.classList.add('text-zinc-500');
    formRegister.classList.remove('hidden');
    formLogin.classList.add('hidden');
    modalTitle.textContent = 'Create AutoOptML Account';
  }
}

async function checkAuthStatus() {
  if (!state.token) {
    updateAuthUI(null);
    return;
  }

  try {
    const res = await authFetch('/api/auth/me');
    if (res.ok) {
      const user = await res.json();
      state.currentUser = user;
      updateAuthUI(user);
      loadUserExperimentsCount();
    } else {
      performLogout();
    }
  } catch (err) {
    console.error('Failed to verify session', err);
    updateAuthUI(null);
  }
}

function updateAuthUI(user) {
  const loggedOutView = document.getElementById('auth-logged-out');
  const loggedInView = document.getElementById('auth-logged-in');
  const greetingBanner = document.getElementById('user-greeting-banner');

  if (user) {
    loggedOutView.classList.add('hidden');
    loggedInView.classList.remove('hidden');
    greetingBanner.classList.remove('hidden');

    document.getElementById('user-display-name').textContent = user.username;
    document.getElementById('user-avatar').textContent = user.username.charAt(0).toUpperCase();
    document.getElementById('greeting-username').textContent = user.username;
    document.getElementById('greeting-email').textContent = user.email;
  } else {
    loggedOutView.classList.remove('hidden');
    loggedInView.classList.add('hidden');
    greetingBanner.classList.add('hidden');
    document.getElementById('nav-experiments-badge').classList.add('hidden');
  }
  lucide.createIcons();
}

async function performDemoLogin() {
  try {
    const res = await fetch('/api/auth/demo-login', { method: 'POST' });
    if (!res.ok) throw new Error('Demo login failed');
    const data = await res.json();
    
    state.token = data.access_token;
    localStorage.setItem('autooptml_token', data.access_token);
    state.currentUser = data.user;

    document.getElementById('modal-auth').classList.add('hidden');
    updateAuthUI(data.user);
    loadUserExperimentsCount();
  } catch (err) {
    alert(`Demo Login Error: ${err.message}`);
  }
}

async function handleLoginForm(e) {
  e.preventDefault();
  const ident = document.getElementById('login-identifier').value.trim();
  const pass = document.getElementById('login-password').value;
  const errEl = document.getElementById('login-error');
  errEl.classList.add('hidden');

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ login_identifier: ident, password: pass })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Login failed');
    }

    const data = await res.json();
    state.token = data.access_token;
    localStorage.setItem('autooptml_token', data.access_token);
    state.currentUser = data.user;

    document.getElementById('modal-auth').classList.add('hidden');
    updateAuthUI(data.user);
    loadUserExperimentsCount();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove('hidden');
  }
}

async function handleRegisterForm(e) {
  e.preventDefault();
  const username = document.getElementById('reg-username').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const password = document.getElementById('reg-password').value;
  const errEl = document.getElementById('reg-error');
  errEl.classList.add('hidden');

  try {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Registration failed');
    }

    const data = await res.json();
    state.token = data.access_token;
    localStorage.setItem('autooptml_token', data.access_token);
    state.currentUser = data.user;

    document.getElementById('modal-auth').classList.add('hidden');
    updateAuthUI(data.user);
    loadUserExperimentsCount();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove('hidden');
  }
}

function performLogout() {
  state.token = null;
  state.currentUser = null;
  localStorage.removeItem('autooptml_token');
  updateAuthUI(null);
}

// ----------------------------------------------------
// Experiments Database (History & Restoring)
// ----------------------------------------------------
async function loadUserExperimentsCount() {
  if (!state.token) return;
  try {
    const res = await authFetch('/api/experiments');
    if (res.ok) {
      const data = await res.json();
      const count = data.experiments.length;
      const badge = document.getElementById('nav-experiments-badge');
      if (count > 0) {
        badge.textContent = count;
        badge.classList.remove('hidden');
      } else {
        badge.classList.add('hidden');
      }
    }
  } catch (err) {
    console.error('Failed to load count', err);
  }
}

async function loadUserExperiments() {
  const container = document.getElementById('experiments-list-container');
  if (!state.token) {
    container.innerHTML = `
      <div class="p-6 text-center text-xs text-zinc-500 bg-zinc-50 rounded-xl border border-zinc-200">
        Please <button onclick="document.getElementById('btn-open-login').click()" class="font-bold underline text-black">Sign In</button> to view your persistent experiment history.
      </div>`;
    return;
  }

  container.innerHTML = `<div class="p-6 text-center text-xs text-zinc-400">Loading experiments from SQLite...</div>`;

  try {
    const res = await authFetch('/api/experiments');
    if (!res.ok) throw new Error('Could not fetch experiments');
    const data = await res.json();
    const experiments = data.experiments;

    if (experiments.length === 0) {
      container.innerHTML = `<div class="p-6 text-center text-xs text-zinc-400 italic">No experiments saved yet. Run an AutoML benchmark to see it recorded here.</div>`;
      return;
    }

    container.innerHTML = '';
    experiments.forEach(exp => {
      const card = document.createElement('div');
      card.className = 'p-4 rounded-xl border border-zinc-200 bg-zinc-50 hover:bg-white hover:border-black transition flex flex-col justify-between space-y-3';
      card.innerHTML = `
        <div class="flex items-start justify-between gap-2">
          <div>
            <h4 class="font-bold text-xs text-black">${exp.title}</h4>
            <p class="text-[11px] text-zinc-500 font-mono mt-0.5">${exp.created_at} • ${exp.n_samples} samples • ${exp.n_features} features</p>
          </div>
          <span class="text-[10px] px-2 py-0.5 rounded-full bg-zinc-200 text-zinc-800 font-semibold uppercase">
            ${exp.task_type.replace('_', ' ')}
          </span>
        </div>

        <div class="p-2.5 rounded-lg bg-white border border-zinc-200 flex items-center justify-between text-xs">
          <div>
            <span class="text-[10px] text-zinc-400 uppercase tracking-wider block font-medium">Champion Model</span>
            <span class="font-bold text-black">${exp.best_model_name}</span>
          </div>
          <div class="text-right">
            <span class="text-[10px] text-zinc-400 uppercase tracking-wider block font-medium">${exp.primary_metric.toUpperCase()}</span>
            <span class="font-black text-black font-mono">${Number(exp.best_score).toFixed(4)}</span>
          </div>
        </div>

        <div class="flex items-center justify-between pt-1">
          <button onclick="restoreExperiment('${exp.id}')" class="px-3 py-1.5 rounded-lg bg-black hover:bg-zinc-800 text-white text-xs font-bold flex items-center space-x-1.5 shadow-sm transition">
            <i data-lucide="eye" class="w-3.5 h-3.5"></i>
            <span>Restore Run</span>
          </button>
          <button onclick="deleteExperiment('${exp.id}')" class="text-xs text-zinc-400 hover:text-red-600 p-1 transition" title="Delete experiment">
            <i data-lucide="trash-2" class="w-4 h-4"></i>
          </button>
        </div>
      `;
      container.appendChild(card);
    });
    lucide.createIcons();
  } catch (err) {
    container.innerHTML = `<div class="p-4 text-xs text-red-600">Error: ${err.message}</div>`;
  }
}

async function restoreExperiment(experimentId) {
  try {
    const res = await authFetch(`/api/experiments/${experimentId}`);
    if (!res.ok) throw new Error('Failed to retrieve experiment');
    const exp = await res.json();

    state.benchmarkResults = {
      leaderboard: exp.leaderboard,
      best_model: {
        model_id: exp.id,
        algorithm_name: exp.best_model_name,
        config_name: exp.best_config_name,
        primary_metric: exp.primary_metric,
        score: exp.best_score
      },
      diagnostics: exp.diagnostics,
      evaluation_summary: {
        models_tested: exp.leaderboard.length,
        primary_metric: exp.primary_metric
      }
    };
    state.datasetName = exp.dataset_name;
    state.profile = {
      meta_features: exp.meta_features,
      summary: {
        target_column: exp.target_column,
        inferred_task: exp.task_type
      }
    };

    document.getElementById('modal-experiments').classList.add('hidden');
    renderStep4Results();
    goToStep(4);
  } catch (err) {
    alert(`Could not restore experiment: ${err.message}`);
  }
}

async function deleteExperiment(experimentId) {
  if (!confirm('Are you sure you want to delete this saved experiment?')) return;
  try {
    const res = await authFetch(`/api/experiments/${experimentId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Deletion failed');
    loadUserExperiments();
    loadUserExperimentsCount();
  } catch (err) {
    alert(`Deletion error: ${err.message}`);
  }
}

// ----------------------------------------------------
// App Navigation & Interactive Stepper
// ----------------------------------------------------
function initAppEventListeners() {
  document.getElementById('step-nav-1').addEventListener('click', () => goToStep(1));
  document.getElementById('step-nav-2').addEventListener('click', () => { if (state.profile) goToStep(2); });
  document.getElementById('step-nav-3').addEventListener('click', () => { if (state.optimizationResult) goToStep(3); });
  document.getElementById('step-nav-4').addEventListener('click', () => { if (state.benchmarkResults) goToStep(4); });

  // Abstract modal
  const modalAbs = document.getElementById('modal-abstract');
  document.getElementById('btn-abstract').addEventListener('click', () => modalAbs.classList.remove('hidden'));
  document.getElementById('btn-close-abstract').addEventListener('click', () => modalAbs.classList.add('hidden'));
  document.getElementById('btn-modal-close-action').addEventListener('click', () => modalAbs.classList.add('hidden'));

  // File drag & drop
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('border-black', 'bg-zinc-100');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('border-black', 'bg-zinc-100');
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('border-black', 'bg-zinc-100');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Step 1 to 2 transition
  document.getElementById('btn-proceed-step-2').addEventListener('click', runAnalysisAndOptimization);
  document.getElementById('btn-to-step-3').addEventListener('click', () => goToStep(3));

  // Target col change event
  document.getElementById('select-target-col').addEventListener('change', onTargetColChange);
  document.getElementById('select-task-type').addEventListener('change', onTaskTypeChange);

  // Step 3: Start training
  document.getElementById('btn-start-training').addEventListener('click', executeTrainingBenchmark);

  // Deployment code copy / download
  document.getElementById('btn-copy-code').addEventListener('click', copyCodeToClipboard);
  document.getElementById('btn-download-code').addEventListener('click', downloadPythonCode);
  document.getElementById('btn-jump-code').addEventListener('click', () => {
    document.getElementById('code-export-section').scrollIntoView({ behavior: 'smooth' });
  });
}

function goToStep(stepNumber) {
  state.currentStep = stepNumber;

  document.getElementById('section-step-1').classList.toggle('hidden', stepNumber !== 1);
  document.getElementById('section-step-2').classList.toggle('hidden', stepNumber !== 2);
  document.getElementById('section-step-3').classList.toggle('hidden', stepNumber !== 3);
  document.getElementById('section-step-4').classList.toggle('hidden', stepNumber !== 4);

  for (let i = 1; i <= 4; i++) {
    const navItem = document.getElementById(`step-nav-${i}`);
    navItem.classList.remove('active');
    if (i < stepNumber) {
      navItem.classList.add('completed');
      navItem.classList.remove('opacity-60');
    } else if (i === stepNumber) {
      navItem.classList.add('active');
      navItem.classList.remove('opacity-60');
    } else {
      navItem.classList.remove('completed');
      if (
        (i === 2 && !state.profile) ||
        (i === 3 && !state.optimizationResult) ||
        (i === 4 && !state.benchmarkResults)
      ) {
        navItem.classList.add('opacity-60');
      } else {
        navItem.classList.remove('opacity-60');
      }
    }
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
  lucide.createIcons();
}

// ----------------------------------------------------
// Sample Datasets Loader
// ----------------------------------------------------
async function loadSampleDatasets() {
  try {
    const res = await fetch('/api/sample-datasets');
    const data = await res.json();
    renderSampleDatasets(data.datasets);
  } catch (err) {
    console.error('Failed to load sample datasets', err);
  }
}

function renderSampleDatasets(datasets) {
  const container = document.getElementById('sample-datasets-container');
  container.innerHTML = '';

  datasets.forEach(ds => {
    const card = document.createElement('div');
    card.className = 'p-4 rounded-xl border border-zinc-200 bg-white hover:border-black hover:shadow-sm transition cursor-pointer flex flex-col justify-between group';
    
    let taskBadge = '';
    if (ds.task === 'binary_classification') {
      taskBadge = '<span class="text-[10px] px-2 py-0.5 rounded-full bg-zinc-100 text-zinc-900 border border-zinc-300 font-semibold">Binary Classification</span>';
    } else if (ds.task === 'multiclass_classification') {
      taskBadge = '<span class="text-[10px] px-2 py-0.5 rounded-full bg-zinc-100 text-zinc-900 border border-zinc-300 font-semibold">Multiclass</span>';
    } else {
      taskBadge = '<span class="text-[10px] px-2 py-0.5 rounded-full bg-zinc-100 text-zinc-900 border border-zinc-300 font-semibold">Regression</span>';
    }

    card.innerHTML = `
      <div>
        <div class="flex items-center justify-between gap-2 mb-2">
          <h4 class="font-extrabold text-sm text-black group-hover:text-black">${ds.name}</h4>
          ${taskBadge}
        </div>
        <p class="text-xs text-zinc-500 line-clamp-2">${ds.description}</p>
      </div>
      <div class="flex items-center justify-between mt-3 pt-2.5 border-t border-zinc-100 text-[11px] text-zinc-500">
        <span>${ds.samples} samples • ${ds.features} features</span>
        <span class="text-black flex items-center gap-1 font-bold group-hover:translate-x-0.5 transition">
          Load <i data-lucide="chevron-right" class="w-3 h-3"></i>
        </span>
      </div>
    `;

    card.addEventListener('click', () => loadSample(ds.id, ds.name));
    container.appendChild(card);
  });
  lucide.createIcons();
}

async function loadSample(datasetId, datasetName) {
  try {
    const res = await fetch('/api/load-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: datasetId })
    });
    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();
    state.sessionId = data.session_id;
    state.datasetName = datasetName;
    state.profile = data.profile;

    renderDatasetPreview();
  } catch (err) {
    alert(`Error loading sample dataset: ${err.message}`);
  }
}

// ----------------------------------------------------
// Custom File Upload
// ----------------------------------------------------
async function handleFileUpload(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload-dataset', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    state.sessionId = data.session_id;
    state.datasetName = data.dataset_name;
    state.profile = data.profile;

    renderDatasetPreview();
  } catch (err) {
    alert(`File upload error: ${err.message}`);
  }
}

// ----------------------------------------------------
// Render Preview Panel
// ----------------------------------------------------
function renderDatasetPreview() {
  const previewPanel = document.getElementById('dataset-preview-panel');
  previewPanel.classList.remove('hidden');

  document.getElementById('loaded-dataset-name').textContent = state.datasetName;
  const summary = state.profile.summary;
  document.getElementById('loaded-dataset-stats').textContent = 
    `${summary.total_rows.toLocaleString()} rows • ${summary.total_columns} columns (${summary.missing_percentage}% missing)`;

  // Populate Target Column selector
  const targetSelect = document.getElementById('select-target-col');
  targetSelect.innerHTML = '';
  state.profile.column_profiles.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col.name;
    opt.textContent = `${col.name} (${col.inferred_type})`;
    if (col.name === summary.target_column) {
      opt.selected = true;
    }
    targetSelect.appendChild(opt);
  });

  // Populate Task Type
  const taskSelect = document.getElementById('select-task-type');
  taskSelect.value = summary.inferred_task;

  // Render first 10 rows
  const table = document.getElementById('preview-table');
  const cols = state.profile.preview_data.columns;
  const rows = state.profile.preview_data.rows;

  let tableHtml = `<thead class="bg-zinc-100 text-zinc-600 border-b border-zinc-200 font-semibold"><tr>`;
  cols.forEach(col => {
    const isTarget = (col === summary.target_column);
    tableHtml += `<th class="py-2.5 px-3 ${isTarget ? 'text-black bg-zinc-200 font-bold' : ''}">${col} ${isTarget ? '★' : ''}</th>`;
  });
  tableHtml += `</tr></thead><tbody class="divide-y divide-zinc-200">`;

  rows.forEach(r => {
    tableHtml += `<tr class="hover:bg-zinc-50">`;
    cols.forEach(col => {
      const isTarget = (col === summary.target_column);
      const val = r[col] !== null ? r[col] : '<span class="text-zinc-400 italic">null</span>';
      tableHtml += `<td class="py-2 px-3 ${isTarget ? 'font-bold text-black bg-zinc-100' : ''}">${val}</td>`;
    });
    tableHtml += `</tr>`;
  });
  tableHtml += `</tbody>`;
  table.innerHTML = tableHtml;

  previewPanel.scrollIntoView({ behavior: 'smooth' });
  lucide.createIcons();
}

async function onTargetColChange() {
  const newTarget = document.getElementById('select-target-col').value;
  const currentTask = document.getElementById('select-task-type').value;

  try {
    const res = await fetch('/api/reprofile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        target_col: newTarget,
        override_task: currentTask
      })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.profile = data.profile;
    document.getElementById('select-task-type').value = state.profile.summary.inferred_task;
  } catch (err) {
    console.error('Reprofiling failed', err);
  }
}

async function onTaskTypeChange() {
  const targetCol = document.getElementById('select-target-col').value;
  const overrideTask = document.getElementById('select-task-type').value;

  try {
    const res = await fetch('/api/reprofile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        target_col: targetCol,
        override_task: overrideTask
      })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.profile = data.profile;
  } catch (err) {
    console.error('Task override failed', err);
  }
}

// ----------------------------------------------------
// Run Meta-Feature Profiling & Conditional Optimization
// ----------------------------------------------------
async function runAnalysisAndOptimization() {
  const targetCol = document.getElementById('select-target-col').value;
  const taskType = document.getElementById('select-task-type').value;
  
  if (state.profile.summary.target_column !== targetCol || state.profile.summary.inferred_task !== taskType) {
    await onTaskTypeChange();
  }

  renderStep2Dashboard();

  try {
    const res = await fetch('/api/optimize-search-space', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        meta_features: state.profile.meta_features
      })
    });
    if (!res.ok) throw new Error(await res.text());

    state.optimizationResult = await res.json();
    renderStep3SearchSpace();

    goToStep(2);
  } catch (err) {
    alert(`Optimization error: ${err.message}`);
  }
}

// ----------------------------------------------------
// Render Step 2: Meta-Features
// ----------------------------------------------------
function renderStep2Dashboard() {
  const meta = state.profile.meta_features;
  const summary = state.profile.summary;
  const targetAnalysis = state.profile.target_analysis;

  document.getElementById('kpi-shape').textContent = `${summary.total_rows.toLocaleString()} × ${summary.total_columns}`;
  document.getElementById('kpi-ratio').textContent = `Ratio N/P: ${summary.ratio_samples_to_features}`;

  document.getElementById('kpi-features').textContent = `${meta.n_numeric} Num • ${meta.n_categorical} Cat`;
  document.getElementById('kpi-types-detail').textContent = `${meta.n_features} effective modeling features`;

  document.getElementById('kpi-missing').textContent = `${summary.missing_percentage}%`;
  document.getElementById('kpi-missing-cells').textContent = `${summary.total_missing_cells} cells need imputation`;

  const formattedTask = summary.inferred_task.replace('_', ' ').toUpperCase();
  document.getElementById('kpi-target-task').textContent = formattedTask;
  if (targetAnalysis.imbalance_ratio) {
    document.getElementById('kpi-target-balance').textContent = `Imbalance: ${targetAnalysis.imbalance_ratio}:1 (${targetAnalysis.is_imbalanced ? 'Imbalanced' : 'Balanced'})`;
  } else if (targetAnalysis.skewness !== undefined) {
    document.getElementById('kpi-target-balance').textContent = `Target Skewness: ${targetAnalysis.skewness}`;
  }

  // Warnings
  const warningsContainer = document.getElementById('diagnostic-warnings-container');
  warningsContainer.innerHTML = '';
  state.profile.warnings.forEach(w => {
    const banner = document.createElement('div');
    banner.className = `p-3.5 rounded-xl border text-xs flex items-start space-x-2.5 bg-zinc-50 border-zinc-300 text-zinc-900`;
    banner.innerHTML = `
      <i data-lucide="alert-circle" class="w-4 h-4 shrink-0 mt-0.5 text-black"></i>
      <div>
        <strong class="font-bold text-black block">${w.title}</strong>
        <p class="text-[11px] text-zinc-600 mt-0.5">${w.message}</p>
      </div>
    `;
    warningsContainer.appendChild(banner);
  });

  // Column Profiles Table
  document.getElementById('columns-count-label').textContent = `${state.profile.column_profiles.length} attributes profiled`;
  const tbody = document.getElementById('column-profiles-tbody');
  tbody.innerHTML = '';

  state.profile.column_profiles.forEach(col => {
    const tr = document.createElement('tr');
    tr.className = col.is_target ? 'bg-zinc-100 font-semibold' : 'hover:bg-zinc-50';
    
    tr.innerHTML = `
      <td class="py-2 px-3 font-semibold text-black">
        ${col.name} ${col.is_target ? '<span class="text-xs text-zinc-900 font-bold ml-1">(Target)</span>' : ''}
      </td>
      <td class="py-2 px-3">
        <span class="text-[10px] px-2 py-0.5 rounded-md border border-zinc-300 bg-zinc-100 text-zinc-900 font-bold">${col.inferred_type}</span>
      </td>
      <td class="py-2 px-3 ${col.missing_count > 0 ? 'text-black font-bold' : 'text-zinc-500'}">
        ${col.missing_percentage}% (${col.missing_count})
      </td>
      <td class="py-2 px-3 text-zinc-600">${col.unique_count}</td>
      <td class="py-2 px-3 text-zinc-600 text-[11px] truncate max-w-xs">${col.sample_values.join(', ')}</td>
    `;
    tbody.appendChild(tr);
  });

  lucide.createIcons();
}

// ----------------------------------------------------
// Render Step 3: Conditional Search-Space Inspector
// ----------------------------------------------------
function renderStep3SearchSpace() {
  const opt = state.optimizationResult;
  const summary = opt.optimization_summary;

  document.getElementById('stat-prune-rate').textContent = `${summary.pruning_efficiency_percent}% Pruned`;
  document.getElementById('retained-count').textContent = summary.retained_count;
  document.getElementById('pruned-count').textContent = summary.pruned_count;

  // Triggered rules
  const rulesContainer = document.getElementById('triggered-rules-container');
  rulesContainer.innerHTML = '';
  opt.triggered_rules.forEach(rule => {
    const card = document.createElement('div');
    card.className = 'p-3.5 rounded-xl bg-white border border-zinc-300 shadow-sm flex items-start space-x-2.5';
    card.innerHTML = `
      <div class="h-6 w-6 rounded-lg bg-zinc-100 text-black flex items-center justify-center shrink-0 mt-0.5 border border-zinc-200">
        <i data-lucide="check" class="w-3.5 h-3.5"></i>
      </div>
      <div>
        <div class="flex items-center space-x-2">
          <span class="text-xs font-bold text-black">${rule.title}</span>
          <span class="text-[9px] font-mono px-1.5 py-0.2 rounded bg-zinc-100 text-zinc-800 border border-zinc-200 font-semibold">${rule.code}</span>
        </div>
        <p class="text-[11px] text-zinc-600 mt-0.5 leading-relaxed">${rule.description}</p>
      </div>
    `;
    rulesContainer.appendChild(card);
  });

  // Retained algorithms
  const selectedContainer = document.getElementById('selected-algorithms-container');
  selectedContainer.innerHTML = '';
  opt.selected_algorithms.forEach(alg => {
    const card = document.createElement('div');
    card.className = 'p-4 rounded-xl bg-white border border-zinc-300 hover:border-black shadow-sm transition space-y-2';

    let configBadges = alg.configurations.map(c => 
      `<span class="text-[11px] px-2 py-0.5 rounded-md bg-zinc-100 text-zinc-900 border border-zinc-200 font-mono font-medium">${c.name}</span>`
    ).join(' ');

    card.innerHTML = `
      <div class="flex items-start justify-between">
        <div class="flex items-center space-x-2.5">
          <input type="checkbox" checked value="${alg.id}" class="alg-checkbox w-4 h-4 rounded text-black border-zinc-300 focus:ring-0">
          <div>
            <h4 class="font-bold text-sm text-black">${alg.name}</h4>
            <span class="text-[11px] text-zinc-500">${alg.family}</span>
          </div>
        </div>
        <span class="text-[10px] px-2 py-0.5 rounded-full bg-zinc-100 text-zinc-900 border border-zinc-300 font-bold">
          ${alg.config_count} Architectures
        </span>
      </div>
      <div class="pt-2 border-t border-zinc-100">
        <span class="text-[10px] uppercase font-bold text-zinc-500 tracking-wider block mb-1">Conditioned Hyperparameter Candidates:</span>
        <div class="flex flex-wrap gap-1.5">
          ${configBadges}
        </div>
      </div>
    `;
    selectedContainer.appendChild(card);
  });

  // Pruned algorithms
  const prunedContainer = document.getElementById('pruned-algorithms-container');
  prunedContainer.innerHTML = '';
  if (opt.pruned_algorithms.length === 0) {
    prunedContainer.innerHTML = `<div class="p-4 rounded-xl bg-zinc-50 border border-zinc-200 text-xs text-zinc-500 italic">No candidate algorithms required pruning. All models compatible.</div>`;
  } else {
    opt.pruned_algorithms.forEach(pruned => {
      const card = document.createElement('div');
      card.className = 'p-4 rounded-xl bg-zinc-50 border border-zinc-300 space-y-2';
      card.innerHTML = `
        <div class="flex items-center justify-between">
          <h4 class="font-bold text-xs text-zinc-900 line-through">${pruned.name}</h4>
          <span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-zinc-200 text-zinc-800 font-bold">${pruned.rule}</span>
        </div>
        <p class="text-[11px] text-zinc-600 leading-relaxed">${pruned.reason}</p>
      `;
      prunedContainer.appendChild(card);
    });
  }

  // Populate Metric Dropdown
  const metricSelect = document.getElementById('select-primary-metric');
  metricSelect.innerHTML = '';
  const primaryOpt = document.createElement('option');
  primaryOpt.value = summary.recommended_primary_metric;
  primaryOpt.textContent = `★ ${summary.metric_label} (Recommended)`;
  primaryOpt.selected = true;
  metricSelect.appendChild(primaryOpt);

  summary.secondary_metrics.forEach(m => {
    const optEl = document.createElement('option');
    optEl.value = m;
    optEl.textContent = m.toUpperCase().replace('_', ' ');
    metricSelect.appendChild(optEl);
  });

  document.getElementById('select-cv-folds').value = summary.cv_folds.toString();

  lucide.createIcons();
}

// ----------------------------------------------------
// Execute Training & Benchmarking
// ----------------------------------------------------
async function executeTrainingBenchmark() {
  const checkedBoxes = Array.from(document.querySelectorAll('.alg-checkbox:checked'));
  if (checkedBoxes.length === 0) {
    alert('Please select at least one candidate algorithm to train.');
    return;
  }

  const selectedIds = new Set(checkedBoxes.map(cb => cb.value));
  const activeAlgorithms = state.optimizationResult.selected_algorithms.filter(alg => selectedIds.has(alg.id));

  const primaryMetric = document.getElementById('select-primary-metric').value;
  const cvFolds = parseInt(document.getElementById('select-cv-folds').value, 10);
  const meta = state.profile.meta_features;

  goToStep(4);
  document.getElementById('training-loader').classList.remove('hidden');
  document.getElementById('results-container').classList.add('hidden');

  try {
    const res = await authFetch('/api/train-and-evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        dataset_name: state.datasetName,
        target_col: meta.target_col,
        task_type: meta.task_type,
        selected_algorithms: activeAlgorithms,
        primary_metric: primaryMetric,
        cv_folds: cvFolds,
        effective_features: meta.effective_features,
        numeric_cols: meta.numeric_cols,
        categorical_cols: meta.categorical_cols
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Training failed');
    }

    state.benchmarkResults = await res.json();
    renderStep4Results();
    loadUserExperimentsCount();
  } catch (err) {
    alert(`Training execution error: ${err.message}`);
    goToStep(3);
  } finally {
    document.getElementById('training-loader').classList.add('hidden');
    document.getElementById('results-container').classList.remove('hidden');
  }
}

// ----------------------------------------------------
// Render Step 4: Leaderboard, Winner & Diagnostics
// ----------------------------------------------------
function renderStep4Results() {
  const results = state.benchmarkResults;
  const best = results.best_model;
  const isRegression = (state.profile.meta_features.task_type === 'regression');

  document.getElementById('winner-model-name').textContent = best.algorithm_name;
  document.getElementById('winner-config-name').textContent = `Configuration: ${best.config_name || 'Standard'}`;
  document.getElementById('winner-metric-label').textContent = `${results.evaluation_summary.primary_metric.toUpperCase()} Score`;
  document.getElementById('winner-metric-score').textContent = Number(best.score).toFixed(4);

  const btnDownload = document.getElementById('btn-download-best-model');
  btnDownload.onclick = () => {
    window.location.href = `/api/download-model/${best.model_id}`;
  };

  document.getElementById('leaderboard-count-label').textContent = `${results.leaderboard.length} Architectures Benchmarked`;
  document.getElementById('th-primary-metric').textContent = `${results.evaluation_summary.primary_metric.toUpperCase()} (Primary)`;

  const tbody = document.getElementById('leaderboard-tbody');
  tbody.innerHTML = '';

  results.leaderboard.forEach(entry => {
    const tr = document.createElement('tr');
    tr.className = (entry.rank === 1) ? 'bg-zinc-100 font-semibold' : 'hover:bg-zinc-50';

    let badgesHtml = (entry.badges || []).map(b => 
      `<span class="text-[10px] px-1.5 py-0.5 rounded border border-zinc-300 bg-white text-zinc-900 font-bold">${b.label}</span>`
    ).join(' ');

    const colAcc = !isRegression ? entry.metrics.accuracy : entry.metrics.rmse;
    const colF1 = !isRegression ? entry.metrics.f1_macro : entry.metrics.r2;

    tr.innerHTML = `
      <td class="py-3 px-4 font-black ${entry.rank === 1 ? 'text-black' : 'text-zinc-500'}">
        #${entry.rank}
      </td>
      <td class="py-3 px-4 font-bold text-black">
        <div>${entry.algorithm_name}</div>
        <div class="text-[10px] text-zinc-500 font-normal font-sans">${entry.config_name}</div>
      </td>
      <td class="py-3 px-4 text-zinc-600 font-sans text-xs">${entry.family || ''}</td>
      <td class="py-3 px-4 font-black text-black">${Number(entry.primary_score).toFixed(4)}</td>
      <td class="py-3 px-4 text-zinc-700">${Number(colAcc || 0).toFixed(4)}</td>
      <td class="py-3 px-4 text-zinc-700">${Number(colF1 || 0).toFixed(4)}</td>
      <td class="py-3 px-4 text-zinc-500">${entry.metrics.train_time || 0}s</td>
      <td class="py-3 px-4">${badgesHtml}</td>
    `;
    tbody.appendChild(tr);
  });

  // Render Charts in High-Contrast Light Theme
  renderTradeoffChart(results.leaderboard);
  renderDiagnosticsChart(results.diagnostics, isRegression);
  renderFeatureImportanceChart(results.diagnostics.feature_importances);

  fetchInferenceCode(best.model_id);
  lucide.createIcons();
}

// ----------------------------------------------------
// Chart 1: Trade-off Scatter/Bar (Light Theme)
// ----------------------------------------------------
function renderTradeoffChart(leaderboard) {
  const ctx = document.getElementById('chart-tradeoff').getContext('2d');
  if (state.charts.tradeoff) state.charts.tradeoff.destroy();

  const labels = leaderboard.map(l => `${l.algorithm_name.split(' ')[0]} (#${l.rank})`);
  const scores = leaderboard.map(l => l.primary_score);
  const times = leaderboard.map(l => l.metrics.train_time);

  state.charts.tradeoff = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Primary Score',
          data: scores,
          backgroundColor: '#09090b',
          borderColor: '#000000',
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: 'y'
        },
        {
          label: 'Train Time (sec)',
          data: times,
          backgroundColor: '#a1a1aa',
          borderColor: '#71717a',
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#09090b', font: { size: 10, weight: 'bold' } }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#09090b', font: { weight: 'bold' } }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#71717a' }
        }
      },
      plugins: {
        legend: { labels: { color: '#09090b', font: { size: 11, weight: 'bold' } } }
      }
    }
  });
}

// ----------------------------------------------------
// Chart 2: Diagnostics (Light Theme)
// ----------------------------------------------------
function renderDiagnosticsChart(diagnostics, isRegression) {
  const ctx = document.getElementById('chart-diagnostics').getContext('2d');
  if (state.charts.diagnostics) state.charts.diagnostics.destroy();

  const titleEl = document.getElementById('chart-diagnostics-title');
  const subEl = document.getElementById('chart-diagnostics-subtitle');

  if (!isRegression && diagnostics.confusion_matrix) {
    titleEl.innerHTML = `<i data-lucide="grid" class="w-4 h-4 text-black"></i> Confusion Matrix (Holdout Set)`;
    subEl.textContent = 'Predicted vs True class distribution counts.';

    const cm = diagnostics.confusion_matrix;
    const classes = cm.classes;
    const labels = [];
    const counts = [];
    const colors = [];

    cm.matrix.forEach((row, rIdx) => {
      row.forEach((val, cIdx) => {
        labels.push(`True: ${classes[rIdx]} → Pred: ${classes[cIdx]}`);
        counts.push(val);
        colors.push(rIdx === cIdx ? '#18181b' : '#d4d4d8');
      });
    });

    state.charts.diagnostics = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: colors,
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#09090b', font: { size: 10 } } }
        }
      }
    });

  } else if (isRegression && diagnostics.residuals) {
    titleEl.innerHTML = `<i data-lucide="scatter-chart" class="w-4 h-4 text-black"></i> Actual vs. Predicted (Holdout Set)`;
    subEl.textContent = `Scatter alignment (Mean Residual: ${diagnostics.residuals.mean_residual})`;

    const points = diagnostics.residuals.sample_points.map(p => ({
      x: p.actual,
      y: p.predicted
    }));

    state.charts.diagnostics = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Prediction Samples',
          data: points,
          backgroundColor: '#09090b',
          borderColor: '#09090b',
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Actual Value', color: '#09090b' },
            grid: { color: 'rgba(0, 0, 0, 0.05)' },
            ticks: { color: '#09090b' }
          },
          y: {
            title: { display: true, text: 'Predicted Value', color: '#09090b' },
            grid: { color: 'rgba(0, 0, 0, 0.05)' },
            ticks: { color: '#09090b' }
          }
        },
        plugins: {
          legend: { labels: { color: '#09090b' } }
        }
      }
    });
  }
}

// ----------------------------------------------------
// Chart 3: Feature Importances (Light Theme)
// ----------------------------------------------------
function renderFeatureImportanceChart(importances) {
  const section = document.getElementById('feature-importance-section');
  if (!importances || importances.length === 0) {
    section.classList.add('hidden');
    return;
  }
  section.classList.remove('hidden');

  const ctx = document.getElementById('chart-importance').getContext('2d');
  if (state.charts.importance) state.charts.importance.destroy();

  const labels = importances.map(i => i.feature);
  const values = importances.map(i => i.importance);

  state.charts.importance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Relative Importance Weight',
        data: values,
        backgroundColor: '#18181b',
        borderColor: '#09090b',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#09090b' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#09090b', font: { size: 11, weight: 'bold' } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// ----------------------------------------------------
// Code Generation & Deployment
// ----------------------------------------------------
async function fetchInferenceCode(modelId) {
  try {
    const res = await fetch(`/api/export-code/${modelId}`);
    const code = await res.text();
    document.getElementById('code-snippet-pre').textContent = code;
  } catch (err) {
    document.getElementById('code-snippet-pre').textContent = `# Error loading code: ${err.message}`;
  }
}

function copyCodeToClipboard() {
  const code = document.getElementById('code-snippet-pre').textContent;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById('btn-copy-code');
    const orig = btn.innerHTML;
    btn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-black"></i><span>Copied!</span>`;
    lucide.createIcons();
    setTimeout(() => {
      btn.innerHTML = orig;
      lucide.createIcons();
    }, 2000);
  });
}

function downloadPythonCode() {
  const code = document.getElementById('code-snippet-pre').textContent;
  const blob = new Blob([code], { type: 'text/x-python' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `inference_pipeline.py`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
