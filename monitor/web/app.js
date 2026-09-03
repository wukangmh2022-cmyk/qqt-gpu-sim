// monitor/web/app.js - 前端仪表盘业务逻辑与交互驱动

let charts = {};
let historyData = [];
let telemetryData = [];

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  fetchStatus();
  fetchTrainTelemetry();
  fetchHistory();
  fetchLeagueMatrix();
  fetchConfigs();

  document.getElementById('btnRefresh').addEventListener('click', () => {
    fetchStatus();
    fetchTrainTelemetry();
    fetchHistory();
    fetchLeagueMatrix();
  });

  document.getElementById('btnTriggerEval').addEventListener('click', triggerEval);
  document.getElementById('btnOpenConfigs').addEventListener('click', () => openModal('modalConfigs'));
  document.getElementById('btnOpenLaunch').addEventListener('click', () => openModal('modalLaunch'));

  // 轮询 (每 6 秒自动刷新)
  setInterval(() => {
    fetchStatus();
    fetchTrainTelemetry();
    fetchHistory();
    fetchLeagueMatrix();
  }, 6000);
});

// --------------------------- Tab 切换逻辑 ---------------------------
function switchViewTab(tab) {
  const btnTrain = document.getElementById('tabBtnTrain');
  const btnEval = document.getElementById('tabBtnEval');
  const secTrain = document.getElementById('viewTrainSection');
  const secEval = document.getElementById('viewEvalSection');

  if (tab === 'train') {
    btnTrain.classList.add('active');
    btnEval.classList.remove('active');
    secTrain.classList.remove('hidden');
    secEval.classList.add('hidden');
  } else {
    btnEval.classList.add('active');
    btnTrain.classList.remove('active');
    secEval.classList.remove('hidden');
    secTrain.classList.add('hidden');
  }
}

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

  // ---------------- 主训练流指标 (4 图) ----------------
  // T1. Loss
  charts.trainLoss = new Chart(document.getElementById('chartTrainLoss'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'PPO Total Loss', borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.1)', data: [], tension: 0.15, fill: true }
      ]
    },
    options: commonOptions
  });

  // T2. Reward & Kill
  charts.trainRewardKill = new Chart(document.getElementById('chartTrainRewardKill'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Mean Step Reward', borderColor: '#10b981', yAxisID: 'y', data: [], tension: 0.15 },
        { label: 'Rollout Kill Rate', borderColor: '#8b5cf6', yAxisID: 'y1', data: [], tension: 0.15 }
      ]
    },
    options: {
      ...commonOptions,
      scales: {
        x: commonOptions.scales.x,
        y: { ...commonOptions.scales.y, position: 'left', title: { display: true, text: 'Reward', color: '#10b981' } },
        y1: { ...commonOptions.scales.y, position: 'right', min: 0, max: 1.0, title: { display: true, text: 'Kill Rate', color: '#8b5cf6' }, grid: { drawOnChartArea: false } }
      }
    }
  });

  // T3. EpLen & Alpha
  charts.trainEpLenAlpha = new Chart(document.getElementById('chartTrainEpLenAlpha'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: '采样平均局长 (ticks)', borderColor: '#38bdf8', yAxisID: 'y', data: [], tension: 0.15 },
        { label: '奖励退火系数 (α)', borderColor: '#eab308', yAxisID: 'y1', data: [], borderDash: [5, 5] }
      ]
    },
    options: {
      ...commonOptions,
      scales: {
        x: commonOptions.scales.x,
        y: { ...commonOptions.scales.y, position: 'left', title: { display: true, text: 'Ticks', color: '#38bdf8' } },
        y1: { ...commonOptions.scales.y, position: 'right', min: 0, max: 1.0, title: { display: true, text: 'Alpha', color: '#eab308' }, grid: { drawOnChartArea: false } }
      }
    }
  });

  // T4. SPS
  charts.trainSps = new Chart(document.getElementById('chartTrainSps'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        { label: '吞吐速度 (SPS)', backgroundColor: 'rgba(56, 189, 248, 0.7)', data: [] }
      ]
    },
    options: commonOptions
  });

  // ---------------- 外部对战验收指标 (4 图) ----------------

  // 1. 胜率趋势 (Winrate vs. Iteration)
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

  // 1.5 相对 Elo 战力分曲线 (Anchor Elo vs. Iteration)
  charts.elo = new Chart(document.getElementById('chartElo'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: '综合 Elo 战力', borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.1)', data: [], tension: 0.2, fill: true },
        { label: '空场景 Elo', borderColor: '#a78bfa', data: [], borderDash: [4, 4] },
        { label: '复杂图 Elo', borderColor: '#f472b6', data: [], borderDash: [4, 4] },
      ]
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, min: 1000, max: 2000 }
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

    // 0. 集群与训练心跳状态
    const clBadge = document.getElementById('clusterBadge');
    if (data.heartbeat) {
      const hb = data.heartbeat;
      if (hb.status === 'RUNNING') {
        clBadge.className = 'badge badge-success';
        clBadge.textContent = `🟢 训练心跳正常 (${hb.idleSeconds}s前)`;
      } else if (hb.status === 'HANG_SUSPECTED') {
        clBadge.className = 'badge badge-warning';
        clBadge.textContent = `🟡 疑似假死卡顿 (停滞 ${hb.idleSeconds}s)`;
      } else if (hb.status === 'DISCONNECTED') {
        clBadge.className = 'badge badge-danger';
        clBadge.textContent = `🔴 集群掉线/断联`;
      } else {
        clBadge.className = 'badge badge-secondary';
        clBadge.textContent = `⚪ 集群空闲 (${hb.mode === 'remote' ? '待命' : '本地未运行'})`;
      }
    }

    // 0.5 训练总轮次推进度 (now it / num it)
    if (data.progress) {
      const p = data.progress;
      const navText = document.getElementById('navProgressText');
      const navBar = document.getElementById('navProgressBar');
      if (navText) {
        navText.textContent = `${p.currentIter.toLocaleString()} / ${p.totalIters.toLocaleString()} (${p.percentage.toFixed(1)}%)`;
      }
      if (navBar) {
        navBar.style.width = `${Math.min(100, Math.max(0, p.percentage))}%`;
      }
    }

    // 1. 策略状态徽标与评估进度
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

    // 2. 告警浮层 (支持策略退化告警 与 集群掉线/卡死告警)
    const alertSec = document.getElementById('alertSection');
    if (data.activeAlert) {
      alertSec.classList.remove('hidden');
      document.getElementById('alertTitle').textContent = `🚨 [${data.activeAlert.severity.toUpperCase()}] ${data.activeAlert.status}: ${data.activeAlert.model}`;
      const issuesList = document.getElementById('alertIssues');
      issuesList.innerHTML = data.activeAlert.issues.map(i => `<li>${i}</li>`).join('');
      const suggBox = document.getElementById('alertSuggestions');
      suggBox.innerHTML = `<strong>建议介入措施：</strong><br>` + data.activeAlert.suggestedActions.join('<br>');
    } else if (data.heartbeat && (data.heartbeat.status === 'HANG_SUSPECTED' || data.heartbeat.status === 'DISCONNECTED')) {
      alertSec.classList.remove('hidden');
      document.getElementById('alertTitle').textContent = `🚨 [集群状态异常] ${data.heartbeat.message}`;
      const issuesList = document.getElementById('alertIssues');
      issuesList.innerHTML = `<li>最新日志输出: ${data.heartbeat.lastLine || '无'}</li><li>最后心跳更新于: ${data.heartbeat.lastHeartbeat} (${data.heartbeat.idleSeconds} 秒前)</li><li>SSH 通信状态: ${data.heartbeat.sshDetail}</li>`;
      const suggBox = document.getElementById('alertSuggestions');
      suggBox.innerHTML = `<strong>排查建议：</strong> 检查 GPU 显存 OOM、NCCL 多机通信死锁或 Slurm 节点运行状态；若已确认掉线请点击右上角 [🚀 回滚/重启干预] 重启训练。`;
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

async function fetchTrainTelemetry() {
  try {
    const res = await fetch('/api/train_telemetry');
    telemetryData = await res.json();
    renderTrainKPIs();
    renderTrainCharts();
  } catch (err) {
    console.error('fetchTrainTelemetry 失败', err);
  }
}

function renderTrainKPIs() {
  if (!telemetryData || !telemetryData.length) return;
  const totalIters = latest.total_iters || 24000;
  const pct = ((latest.iter / totalIters) * 100).toFixed(1);
  document.getElementById('kpiTrainIter').textContent = `${latest.iter.toLocaleString()} / ${totalIters.toLocaleString()}`;
  document.getElementById('kpiTrainSteps').textContent = `全局步数: ${latest.gs ? latest.gs.toLocaleString() : '--'} (已完成 ${pct}%)`;

  document.getElementById('kpiTrainLoss').textContent = latest.loss.toFixed(4);
  document.getElementById('kpiTrainLossSub').textContent = `α=${latest.alpha.toFixed(2)} (密集退火)`;

  document.getElementById('kpiTrainRew').textContent = latest.rew.toFixed(3);
  document.getElementById('kpiTrainRewSub').textContent = `Step 累计回报`;

  document.getElementById('kpiTrainKill').textContent = `${(latest.kill * 100).toFixed(1)}%`;
  document.getElementById('kpiTrainKillSub').textContent = `斩杀率: ${latest.kill.toFixed(3)}`;

  document.getElementById('kpiTrainSps').textContent = `${Math.round(latest.sps).toLocaleString()}`;
  document.getElementById('kpiTrainEpLen').textContent = `采样平均局长: ${latest.ep_len ? latest.ep_len.toFixed(1) : '--'}t`;
}

function renderTrainCharts() {
  if (!telemetryData || !telemetryData.length) return;

  const labels = telemetryData.map(d => `it${d.iter}`);

  // T1. Loss
  charts.trainLoss.data.labels = labels;
  charts.trainLoss.data.datasets[0].data = telemetryData.map(d => d.loss);
  charts.trainLoss.update();

  // T2. Reward & Kill
  charts.trainRewardKill.data.labels = labels;
  charts.trainRewardKill.data.datasets[0].data = telemetryData.map(d => d.rew);
  charts.trainRewardKill.data.datasets[1].data = telemetryData.map(d => d.kill);
  charts.trainRewardKill.update();

  // T3. EpLen & Alpha
  charts.trainEpLenAlpha.data.labels = labels;
  charts.trainEpLenAlpha.data.datasets[0].data = telemetryData.map(d => d.ep_len);
  charts.trainEpLenAlpha.data.datasets[1].data = telemetryData.map(d => d.alpha);
  charts.trainEpLenAlpha.update();

  // T4. SPS
  charts.trainSps.data.labels = labels;
  charts.trainSps.data.datasets[0].data = telemetryData.map(d => d.sps);
  charts.trainSps.update();
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

  if (latest.elo) {
    document.getElementById('kpiElo').textContent = `${latest.elo.composite}`;
    document.getElementById('kpiEloSub').textContent = `道场: ${latest.elo.vsOpenHunter} | 复杂: ${latest.elo.vsFullHunter}`;
  } else {
    document.getElementById('kpiElo').textContent = `--`;
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

  // 按 iteration 数值严格排序对齐横轴
  const sortedHistory = [...historyData].sort((a, b) => {
    const itA = a.iteration !== undefined ? a.iteration : (parseInt((a.modelName.match(/it(\d+)/) || [])[1]) || 0);
    const itB = b.iteration !== undefined ? b.iteration : (parseInt((b.modelName.match(/it(\d+)/) || [])[1]) || 0);
    return itA - itB;
  });

  const labels = sortedHistory.map(h => {
    const it = h.iteration !== undefined ? h.iteration : (parseInt((h.modelName.match(/it(\d+)/) || [])[1]) || 0);
    return `Iter ${it}`;
  });

  // 1. 胜率趋势 (Winrate vs. Iteration)
  charts.winRates.data.labels = labels;
  charts.winRates.data.datasets[0].data = sortedHistory.map(h => (h.domains.open_hunter ? h.domains.open_hunter.winRate : null));
  charts.winRates.data.datasets[1].data = sortedHistory.map(h => (h.domains.full_hunter ? h.domains.full_hunter.winRate : null));
  charts.winRates.data.datasets[2].data = sortedHistory.map(h => (h.domains.open_idle ? h.domains.open_idle.winRate : null));
  charts.winRates.data.datasets[3].data = sortedHistory.map(h => (h.domains.full_idle ? h.domains.full_idle.winRate : null));
  charts.winRates.update();

  // 1.5 相对 Elo 天梯战力曲线 (Anchor Elo vs. Iteration)
  charts.elo.data.labels = labels;
  charts.elo.data.datasets[0].data = sortedHistory.map(h => (h.elo ? h.elo.composite : null));
  charts.elo.data.datasets[1].data = sortedHistory.map(h => (h.elo ? h.elo.vsOpenHunter : null));
  charts.elo.data.datasets[2].data = sortedHistory.map(h => (h.elo ? h.elo.vsFullHunter : null));
  charts.elo.update();

  // 2. 局长
  charts.ticks.data.labels = labels;
  charts.ticks.data.datasets[0].data = sortedHistory.map(h => (h.domains.open_hunter ? h.domains.open_hunter.avgTicks : null));
  charts.ticks.data.datasets[1].data = sortedHistory.map(h => (h.domains.full_hunter ? h.domains.full_hunter.avgTicks : null));
  charts.ticks.data.datasets[2].data = sortedHistory.map(h => (h.domains.open_idle ? h.domains.open_idle.avgTicks : null));
  charts.ticks.data.datasets[3].data = sortedHistory.map(h => (h.domains.full_idle ? h.domains.full_idle.avgTicks : null));
  charts.ticks.update();

  // 3. 进攻性
  charts.aggression.data.labels = labels;
  charts.aggression.data.datasets[0].data = sortedHistory.map(h => (h.domains.open_hunter ? h.domains.open_hunter.avgBombs : null));
  charts.aggression.data.datasets[1].data = sortedHistory.map(h => (h.domains.full_hunter ? h.domains.full_hunter.avgBombs : null));
  charts.aggression.data.datasets[2].data = sortedHistory.map(h => (h.domains.full_hunter ? h.domains.full_hunter.avgHits : null));
  charts.aggression.update();

  // 4. 自杀率
  charts.safety.data.labels = labels;
  charts.safety.data.datasets[0].data = sortedHistory.map(h => (h.domains.open_hunter ? h.domains.open_hunter.suicides : null));
  charts.safety.data.datasets[1].data = sortedHistory.map(h => (h.domains.full_hunter ? h.domains.full_hunter.suicides : null));
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

    const milestoneBadge = h.isMilestone
      ? ` <span class="badge" style="background: #f59e0b; color: #fff; font-size: 0.72rem;" title="${h.milestoneReason || '里程碑'}">👑 里程碑</span>`
      : '';

    const alphaBadge = h.meta?.annealed_hyperparams?.alpha !== undefined
      ? `<br><small style="color: #38bdf8;">α=${h.meta.annealed_hyperparams.alpha.toFixed(2)}</small>`
      : '';

    tr.innerHTML = `
      <td><strong>${h.modelName}</strong>${milestoneBadge}${alphaBadge}</td>
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

// --------------------------- 跨代对抗联赛矩阵与闭环检测 ---------------------------
async function fetchLeagueMatrix() {
  try {
    const res = await fetch('/api/league');
    const data = await res.json();
    renderLeagueMatrix(data);
  } catch (err) {
    console.error('fetchLeagueMatrix 失败', err);
  }
}

function renderLeagueMatrix(data) {
  if (!data || !data.matrix || !data.matrix.length) return;

  // 1. 状态徽标与涡流比
  const badgeStatus = document.getElementById('badgeCycleStatus');
  const badgeRatio = document.getElementById('badgeCyclicRatio');

  if (data.hodge) {
    const ratio = data.hodge.cyclic_ratio || 0;
    badgeRatio.textContent = `涡流比: ${ratio}%`;
    if (data.has_cycles || ratio > 25.0) {
      badgeStatus.className = 'badge badge-danger';
      badgeStatus.textContent = '🔴 闭环克制 (石头剪刀布)';
    } else if (ratio > 10.0) {
      badgeStatus.className = 'badge badge-warning';
      badgeStatus.textContent = '🟡 轻微克制回旋';
    } else {
      badgeStatus.className = 'badge badge-success';
      badgeStatus.textContent = '🟢 传递性良好 (绝对晋级)';
    }
  }

  // 2. 闭环警告浮层
  const banner = document.getElementById('cycleWarningBanner');
  if (data.has_cycles && data.cycles && data.cycles.length > 0) {
    banner.classList.remove('hidden');
    document.getElementById('cycleWarningTitle').textContent = `🚨 警告：检测到 ${data.cycles.length} 个跨代对决闭环回路 (石头剪刀布)！`;
    const list = document.getElementById('cycleWarningList');
    list.innerHTML = data.cycles.map(c => `<li><strong>${c.summary}</strong> (对局相互克制，绝对胜率失效)</li>`).join('');
  } else {
    banner.classList.add('hidden');
  }

  // 3. 渲染矩阵表格
  const thead = document.getElementById('leagueMatrixHeader');
  const tbody = document.getElementById('leagueMatrixBody');

  const models = data.matrix.map(m => m.model);
  const shorten = (name) => {
    const m = name.match(/it(\d+)/);
    return m ? `Iter ${parseInt(m[1])}` : (name.length > 12 ? name.slice(0, 10) + '..' : name);
  };

  thead.innerHTML = `<tr><th>对手 (P0 \\ P1)</th>` + models.map(m => `<th>${shorten(m)}</th>`).join('') + `</tr>`;

  tbody.innerHTML = '';
  data.matrix.forEach(row => {
    const tr = document.createElement('tr');
    let html = `<td><strong>${shorten(row.model)}</strong></td>`;
    models.forEach(target => {
      const score = row.scores[target];
      if (row.model === target) {
        html += `<td style="color: #64748b; background: rgba(100, 116, 139, 0.1);">--</td>`;
      } else if (score >= 60.0) {
        html += `<td style="color: #4ade80; font-weight: bold; background: rgba(74, 222, 128, 0.1);">${score}%</td>`;
      } else if (score <= 40.0) {
        html += `<td style="color: #f87171; font-weight: bold; background: rgba(248, 113, 113, 0.1);">${score}%</td>`;
      } else {
        html += `<td style="color: #94a3b8;">${score}%</td>`;
      }
    });
    tr.innerHTML = html;
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
