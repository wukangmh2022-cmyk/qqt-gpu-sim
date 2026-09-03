// monitor/web/app.js - 前端仪表盘业务逻辑与交互驱动

let charts = {};
let historyData = [];

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  fetchStatus();
  fetchHistory();
  fetchConfigs();

  document.getElementById('btnRefresh').addEventListener('click', () => {
    fetchStatus();
    fetchHistory();
  });

  document.getElementById('btnTriggerEval').addEventListener('click', triggerEval);
  document.getElementById('btnOpenConfigs').addEventListener('click', () => openModal('modalConfigs'));
  document.getElementById('btnOpenLaunch').addEventListener('click', () => openModal('modalLaunch'));

  // 轮询 (每 6 秒自动刷新)
  setInterval(() => {
    fetchStatus();
    fetchHistory();
  }, 6000);
});

// --------------------------- Chart.js 初始化 ---------------------------
function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    color: '#94a3b8',
    plugins: {
      legend: { labels: { color: '#94a3b8', boxWidth: 12 } }
    },
    scales: {
      x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
      y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
    }
  };

  // 1. 胜率趋势
  charts.winRates = new Chart(document.getElementById('chartWinRates'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: '空场景 vs Hunter', borderColor: '#38bdf8', backgroundColor: '#38bdf8', data: [], tension: 0.2 },
        { label: '全池图 vs Hunter', borderColor: '#c084fc', backgroundColor: '#c084fc', data: [], tension: 0.2 },
        { label: '空场景 vs 木桩', borderColor: '#4ade80', backgroundColor: '#4ade80', data: [], tension: 0.2 },
        { label: '全池图 vs 木桩', borderColor: '#fb923c', backgroundColor: '#fb923c', data: [], tension: 0.2 },
      ]
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, min: 0, max: 100, ticks: { callback: v => v + '%' } }
      }
    }
  });

  // 2. 局长趋势
  charts.ticks = new Chart(document.getElementById('chartTicks'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: '空场景 vs Hunter (ticks)', borderColor: '#38bdf8', data: [], tension: 0.2 },
        { label: '全池图 vs Hunter (ticks)', borderColor: '#c084fc', data: [], tension: 0.2 },
        { label: '空场景 vs 木桩 (ticks)', borderColor: '#4ade80', data: [], tension: 0.2 },
        { label: '全池图 vs 木桩 (ticks)', borderColor: '#fb923c', data: [], tension: 0.2 },
      ]
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, min: 0, max: 1800 }
      }
    }
  });

  // 3. 放炮与命中
  charts.aggression = new Chart(document.getElementById('chartAggression'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        { label: '空场景放炮/局', backgroundColor: 'rgba(56, 189, 248, 0.7)', data: [] },
        { label: '全池图放炮/局', backgroundColor: 'rgba(192, 132, 252, 0.7)', data: [] },
        { label: '全池图命中/局', type: 'line', borderColor: '#fde047', data: [] }
      ]
    },
    options: commonOptions
  });

  // 4. 自杀率走势
  charts.safety = new Chart(document.getElementById('chartSafety'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: '空场景自杀数', borderColor: '#f87171', backgroundColor: '#f87171', data: [], tension: 0.2 },
        { label: '全池图自杀数', borderColor: '#fb7185', backgroundColor: '#fb7185', data: [], tension: 0.2 },
      ]
    },
    options: commonOptions
  });
}

// --------------------------- 数据拉取与更新 ---------------------------
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    // 1. 状态徽标与评估进度
    const badge = document.getElementById('healthBadge');
    if (data.evalInProgress) {
      badge.className = 'badge badge-warning';
      badge.textContent = `⏳ 正在评测中: ${data.currentEvaluating}...`;
    } else {
      const status = data.health ? data.health.status : 'HEALTHY';
      if (status === 'HEALTHY') {
        badge.className = 'badge badge-success';
        badge.textContent = '🟢 策略迭代健康';
      } else if (status === 'WARNING') {
        badge.className = 'badge badge-warning';
        badge.textContent = '🟡 策略警告 (需留意)';
      } else {
        badge.className = 'badge badge-danger';
        badge.textContent = '🔴 策略死锁 / 严重退化！';
      }
    }

    // 2. 告警浮层
    const alertSec = document.getElementById('alertSection');
    if (data.activeAlert) {
      alertSec.classList.remove('hidden');
      document.getElementById('alertTitle').textContent = `🚨 [${data.activeAlert.severity.toUpperCase()}] ${data.activeAlert.status}: ${data.activeAlert.model}`;
      const issuesList = document.getElementById('alertIssues');
      issuesList.innerHTML = data.activeAlert.issues.map(i => `<li>${i}</li>`).join('');
      const suggBox = document.getElementById('alertSuggestions');
      suggBox.innerHTML = `<strong>建议介入措施：</strong><br>` + data.activeAlert.suggestedActions.join('<br>');
    } else {
      alertSec.classList.add('hidden');
    }

    // 3. 更新快照选择列表
    const select = document.getElementById('selectModelToEval');
    const resumeSelect = document.getElementById('launchResumeSelect');
    const currVal = select.value;
    const currResume = resumeSelect.value;

    select.innerHTML = '<option value="">-- 选择快照执行即时评测 --</option>';
    resumeSelect.innerHTML = '';
    (data.availableCheckpoints || []).forEach(name => {
      const opt1 = document.createElement('option');
      opt1.value = name; opt1.textContent = name;
      select.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = name; opt2.textContent = name;
      resumeSelect.appendChild(opt2);
    });

    if (currVal) select.value = currVal;
    if (currResume) resumeSelect.value = currResume;
    updateLaunchPreview();

  } catch (err) {
    console.error('fetchStatus 失败', err);
  }
}

async function fetchHistory() {
  try {
    const res = await fetch('/api/history');
    historyData = await res.json();
    renderKPIs();
    renderCharts();
    renderTable();
  } catch (err) {
    console.error('fetchHistory 失败', err);
  }
}

function renderKPIs() {
  if (!historyData || !historyData.length) return;
  const latest = historyData[historyData.length - 1];
  const doms = latest.domains || {};

  document.getElementById('kpiStatus').textContent = latest.health ? latest.health.status : 'HEALTHY';
  document.getElementById('kpiLatestCkpt').textContent = latest.modelName;

  if (doms.open_hunter) {
    document.getElementById('kpiOpenHunterWin').textContent = `${doms.open_hunter.winRate}%`;
    document.getElementById('kpiOpenHunterTicks').textContent = `局均 ${doms.open_hunter.avgTicks} ticks | 炮 ${doms.open_hunter.avgBombs}`;
  }
  if (doms.full_hunter) {
    document.getElementById('kpiFullHunterWin').textContent = `${doms.full_hunter.winRate}%`;
    document.getElementById('kpiFullHunterTicks').textContent = `局均 ${doms.full_hunter.avgTicks} ticks | 命中 ${doms.full_hunter.avgHits}`;
  }
  if (doms.open_idle) {
    document.getElementById('kpiOpenIdleWin').textContent = `${doms.open_idle.winRate}%`;
    document.getElementById('kpiOpenIdleTicks').textContent = `速杀局均 ${doms.open_idle.avgTicks} ticks`;
  }
  if (doms.full_idle) {
    document.getElementById('kpiFullIdleWin').textContent = `${doms.full_idle.winRate}%`;
    document.getElementById('kpiFullIdleTimeout').textContent = `超时: ${doms.full_idle.timeouts} 局 | 负: ${doms.full_idle.losses}`;
  }
}

function renderCharts() {
  if (!historyData || !historyData.length) return;

  const labels = historyData.map(h => {
    const name = h.modelName.replace('params_it', 'it').replace('_patch3_k32', '').replace('_hlgauss_top25foractor', '');
    return name.length > 18 ? name.slice(0, 18) + '…' : name;
  });

  // 1. 胜率
  charts.winRates.data.labels = labels;
  charts.winRates.data.datasets[0].data = historyData.map(h => (h.domains.open_hunter ? h.domains.open_hunter.winRate : null));
  charts.winRates.data.datasets[1].data = historyData.map(h => (h.domains.full_hunter ? h.domains.full_hunter.winRate : null));
  charts.winRates.data.datasets[2].data = historyData.map(h => (h.domains.open_idle ? h.domains.open_idle.winRate : null));
  charts.winRates.data.datasets[3].data = historyData.map(h => (h.domains.full_idle ? h.domains.full_idle.winRate : null));
  charts.winRates.update();

  // 2. 局长
  charts.ticks.data.labels = labels;
  charts.ticks.data.datasets[0].data = historyData.map(h => (h.domains.open_hunter ? h.domains.open_hunter.avgTicks : null));
  charts.ticks.data.datasets[1].data = historyData.map(h => (h.domains.full_hunter ? h.domains.full_hunter.avgTicks : null));
  charts.ticks.data.datasets[2].data = historyData.map(h => (h.domains.open_idle ? h.domains.open_idle.avgTicks : null));
  charts.ticks.data.datasets[3].data = historyData.map(h => (h.domains.full_idle ? h.domains.full_idle.avgTicks : null));
  charts.ticks.update();

  // 3. 进攻性
  charts.aggression.data.labels = labels;
  charts.aggression.data.datasets[0].data = historyData.map(h => (h.domains.open_hunter ? h.domains.open_hunter.avgBombs : null));
  charts.aggression.data.datasets[1].data = historyData.map(h => (h.domains.full_hunter ? h.domains.full_hunter.avgBombs : null));
  charts.aggression.data.datasets[2].data = historyData.map(h => (h.domains.full_hunter ? h.domains.full_hunter.avgHits : null));
  charts.aggression.update();

  // 4. 自杀率
  charts.safety.data.labels = labels;
  charts.safety.data.datasets[0].data = historyData.map(h => (h.domains.open_hunter ? h.domains.open_hunter.suicides : null));
  charts.safety.data.datasets[1].data = historyData.map(h => (h.domains.full_hunter ? h.domains.full_hunter.suicides : null));
  charts.safety.update();
}

function renderTable() {
  const tbody = document.getElementById('historyTableBody');
  if (!historyData || !historyData.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center">暂无评测记录</td></tr>';
    return;
  }

  tbody.innerHTML = '';
  // 倒序展示最新
  [...historyData].reverse().forEach(h => {
    const d = h.domains || {};
    const tr = document.createElement('tr');

    const statusBadge = h.health && h.health.status === 'HEALTHY'
      ? '<span class="badge badge-success">正常</span>'
      : (h.health && h.health.status === 'WARNING' ? '<span class="badge badge-warning">警告</span>' : '<span class="badge badge-danger">退化</span>');

    const ohStr = d.open_hunter ? `${d.open_hunter.winRate}% (局长${d.open_hunter.avgTicks})` : '--';
    const fhStr = d.full_hunter ? `${d.full_hunter.winRate}% (局长${d.full_hunter.avgTicks})` : '--';
    const fiStr = d.full_idle ? `${d.full_idle.winRate}% (超时${d.full_idle.timeouts})` : '--';
    const expStr = d.full_idle?.avgExplored ? `${d.full_idle.avgExplored}%` : (d.open_hunter?.avgExplored ? `${d.open_hunter.avgExplored}%` : '--');
    const idleStr = d.open_hunter?.avgIdle ? `${d.open_hunter.avgIdle}%` : '--';
    const entStr = d.open_hunter?.avgEntropy ? `${d.open_hunter.avgEntropy}` : '--';

    tr.innerHTML = `
      <td><strong>${h.modelName}</strong></td>
      <td>${h.timestamp}</td>
      <td>${h.evalDurationSeconds}s</td>
      <td>${ohStr}</td>
      <td>${fhStr}</td>
      <td>${fiStr}</td>
      <td><span class="highlight-blue">${expStr}</span></td>
      <td><span class="${parseFloat(idleStr) > 40 ? 'highlight-orange' : ''}">${idleStr}</span></td>
      <td><span class="highlight-green">${entStr}</span></td>
      <td>${statusBadge}</td>
      <td>
        <a href="http://localhost:8080/?model=${h.modelName}" target="_blank" class="btn btn-sm btn-primary">🎮 观战</a>
        <button class="btn btn-sm btn-secondary" onclick="reEvalModel('${h.modelName}')">⚡ 复测</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// --------------------------- 即时评估与介入 ---------------------------
async function triggerEval() {
  const model = document.getElementById('selectModelToEval').value;
  if (!model) return alert('请先选择要评估的模型快照');
  reEvalModel(model);
}

async function reEvalModel(modelName) {
  try {
    const res = await fetch('/api/eval_now', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName })
    });
    const data = await res.json();
    if (data.error) {
      alert(`启动失败: ${data.error}`);
    } else {
      alert(`已成功在后台拉起对 ${modelName} 的 128 局极速评测！预计 3.1 分钟产出结果。`);
      fetchStatus();
    }
  } catch (e) {
    alert(`请求异常: ${e.message}`);
  }
}

// --------------------------- TOML 配置管理 ---------------------------
async function fetchConfigs() {
  try {
    const res = await fetch('/api/configs');
    const configs = await res.json();
    const sel1 = document.getElementById('configSelect');
    const sel2 = document.getElementById('launchConfigSelect');

    sel1.innerHTML = ''; sel2.innerHTML = '';
    configs.forEach(c => {
      const opt1 = document.createElement('option');
      opt1.value = c; opt1.textContent = c;
      sel1.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = c; opt2.textContent = c;
      sel2.appendChild(opt2);
    });

    // 默认选中
    if (configs.includes('repro_it68_scheme1_actor_top25_critic_all_patch3_k32.toml')) {
      sel1.value = 'repro_it68_scheme1_actor_top25_critic_all_patch3_k32.toml';
      sel2.value = 'repro_it68_scheme1_actor_top25_critic_all_patch3_k32.toml';
    }
    loadConfigFile();
    updateLaunchPreview();

    sel2.addEventListener('change', updateLaunchPreview);
    document.getElementById('launchResumeSelect').addEventListener('change', updateLaunchPreview);
    document.getElementById('launchScriptInput').addEventListener('input', updateLaunchPreview);
  } catch (e) {
    console.error('fetchConfigs 失败', e);
  }
}

async function loadConfigFile() {
  const name = document.getElementById('configSelect').value;
  if (!name) return;
  try {
    const res = await fetch(`/api/config?name=${encodeURIComponent(name)}`);
    const data = await res.json();
    document.getElementById('configContent').value = data.content || '';
    document.getElementById('configSaveTip').textContent = '';
  } catch (e) {
    console.error('loadConfigFile 失败', e);
  }
}

async function saveConfigFile() {
  const name = document.getElementById('configSelect').value;
  const content = document.getElementById('configContent').value;
  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, content })
    });
    const data = await res.json();
    if (data.success) {
      document.getElementById('configSaveTip').textContent = `✅ ${name} 已保存成功！`;
      setTimeout(() => { document.getElementById('configSaveTip').textContent = ''; }, 3000);
    } else {
      alert(`保存失败: ${data.error}`);
    }
  } catch (e) {
    alert(`保存异常: ${e.message}`);
  }
}

// --------------------------- 回滚与重启命令预览 ---------------------------
function updateLaunchPreview() {
  const resume = document.getElementById('launchResumeSelect').value || 'params_it00000068_scheme1_actor_top25_critic_all_patch3_k32';
  const cfg = document.getElementById('launchConfigSelect').value || 'repro_it68_scheme1_actor_top25_critic_all_patch3_k32.toml';
  const script = document.getElementById('launchScriptInput').value || 'scripts/launch_8node_it68_hlgauss_top25_patch3.sh';

  const preview = `CONFIG=configs/${cfg} RESUME_CKPT=ckpt/${resume}.pkl bash ${script}`;
  document.getElementById('launchCommandPreview').textContent = preview;
}

function executeRestart() {
  const cmd = document.getElementById('launchCommandPreview').textContent;
  navigator.clipboard.writeText(cmd).then(() => {
    alert(`启动指令已复制到剪贴板！\n\n${cmd}\n\n可在训练终端直接粘贴执行，或由 Agent 调度器接管。`);
    closeModal('modalLaunch');
  }).catch(() => {
    alert(`请手动复制指令并在终端执行：\n\n${cmd}`);
  });
}

// --------------------------- Modal 通用控制 ---------------------------
function openModal(id) {
  document.getElementById(id).classList.remove('hidden');
}

function closeModal(id) {
  document.getElementById(id).classList.add('hidden');
}
