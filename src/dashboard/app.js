const fields = [...document.querySelectorAll('#settings input')];
const algorithmSelect = document.querySelector('#algorithm');
const settingsStatus = document.querySelector('#settings-status');
let initialized = false;
const zh = {
  approach: '接近', descend: '下降', close: '夾爪閉合', lift: '抬升', to_bin: '移往托盤', lower: '放下', open: '夾爪張開', settle: '放置穩定', retreat: '撤離', home: '回原位',
  infeed: '投料', assign: '派工', reserve_wait: '安全等待', grasp: '抓取', release: '鬆開', place: '放置', missed: '漏件', safety_recover: '安全回復', safety_stop: '安全停止',
  left_bin: '左側托盤', right_bin: '右側托盤', shared_middle: '共享中間區', exclusive_left: '左側專屬區', exclusive_right: '右側專屬區',
  middle: '中間件', left: '左側件', right: '右側件', tail_exit: '尾端離開'
};
const label = value => zh[value] || value;

async function control(action, values) {
  const response = await fetch('/api/control', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action, values})});
  if (!response.ok) throw new Error((await response.json()).error || 'request failed');
  return response.json();
}
function setRows(element, rows, empty) {
  element.replaceChildren();
  if (!rows.length) { element.textContent = empty; return; }
  rows.forEach(row => { const item = document.createElement('div'); item.className = 'row'; item.innerHTML = row; element.append(item); });
}
function render(state) {
  document.querySelector('#connection').textContent = state.paused ? '已暫停' : '運行中';
  document.querySelector('#sim-time').textContent = `${state.time_s.toFixed(3)} s`;
  const viewer = state.viewer || {};
  document.querySelector('#viewer-status').textContent = viewer.active ? 'MuJoCo 視窗已開啟；按鈕可將它帶到前景。' : viewer.launching || viewer.requested ? '正在開啟 MuJoCo 視窗…' : 'MuJoCo 視窗已關閉，可隨時重新開啟。';
  ['spawned', 'placed', 'missed'].forEach(key => document.querySelector(`#${key}`).textContent = state.counts[key]);
  const feedback = state.feedback || {};
  const armRows = Object.entries(feedback.arms || {}).map(([arm, value]) => `<strong>手臂 ${arm}</strong><span>週期 ${value.cycle_s.toFixed(2)} s</span><span>抓取 ${value.grasped}/${value.attempts} · 放置 ${value.placed}</span>`);
  armRows.unshift(`<strong>產線負載</strong><span>線上 ${feedback.active_parts ?? 0} 件</span><span>中央週期估計 ${(feedback.cycle_estimate_s ?? 0).toFixed(2)} s</span>`);
  setRows(document.querySelector('#feedback'), armRows, '尚無執行回饋。');
  if (!initialized) { fields.forEach(field => field.value = field.name === 'seed' ? state.seed : state.parameters[field.name]); algorithmSelect.value = state.algorithm.id; initialized = true; }
  const missions = Object.entries(state.missions).map(([arm, task]) => `<strong>手臂 ${arm}</strong><span>${task.object_id} · ${label(task.stage)}</span><span>${label(task.placement_zone)} · ${task.route || 'direct'}</span>`);
  setRows(document.querySelector('#missions'), missions, '兩台手臂皆可接收任務。');
  document.querySelector('#deferred').textContent = state.deferred.length ? `安全等待：${state.deferred.join('、')} 正等待中央走廊淨空。` : '';
  const check = state.preflight || {};
  document.querySelector('#preflight').textContent = check.status === 'clear' ? `路徑碰撞預檢：通過（${check.object_id} / 手臂 ${check.arm}）` : check.status === 'deferred' ? `路徑碰撞預檢：暫緩，${check.reason}` : '路徑碰撞預檢：等待任務';
  document.querySelector('#algorithm-name').textContent = state.algorithm.name;
  const joint = state.joint_plan || {};
  const jointRows = joint.status === 'selected' ? [
    `<strong>聯合策略</strong><span>A=${joint.route_a} · B=${joint.route_b}</span><span>評估 ${joint.evaluated} 組</span>`,
    `<strong>貝式期望</strong><span>完工機率 ${(100 * joint.completion_probability).toFixed(1)}%</span><span>效用 ${joint.expected_utility}</span>`,
    `<strong>全局成本</strong><span>工期 ${joint.makespan_s.toFixed(2)} s</span><span>同步率 ${(100 * joint.simultaneous_ratio).toFixed(1)}%</span>`
  ] : joint.status === 'no_safe_joint_strategy' ? [
    `<strong>無安全聯合解</strong><span>9 組已全數檢查</span><span>${joint.reason}</span>`
  ] : [];
  setRows(document.querySelector('#joint-plan'), jointRows, '等待兩件物體進入聯合決策。');
  const assignments = state.decision.assignments.map(task => `<strong>${task.object_id}</strong><span>手臂 ${task.arm} · ${label(task.zone)}</span><span>${label(task.placement)}</span>`);
  setRows(document.querySelector('#assignments'), assignments, '最新一輪沒有可行派工。');
  document.querySelector('#rejected').textContent = JSON.stringify(state.decision.rejected, null, 2);
  const eventKeys = {object_id: '物件', object_class: '類別', arm: '手臂', placement: '放置區', reason: '原因'};
  const events = document.querySelector('#events'); events.replaceChildren(); state.events.slice().reverse().forEach(event => { const item = document.createElement('li'); item.textContent = `${event.time_s.toFixed(3)}s  ${label(event.event)}  ${Object.entries(event).filter(([key]) => !['time_s','event'].includes(key)).map(([key, value]) => `${eventKeys[key] || key}=${label(value)}`).join(' ')}`; events.append(item); });
}
async function refresh() { try { render(await (await fetch('/api/state')).json()); } catch (_) { document.querySelector('#connection').textContent = '未連線'; } }
document.querySelector('#restart').onclick = () => control('restart').then(render);
document.querySelector('#pause').onclick = () => control('pause').then(render);
document.querySelector('#resume').onclick = () => control('start').then(render).catch(error => window.alert(error.message));
document.querySelector('#open-mujoco').onclick = () => control('open_mujoco').then(render).catch(error => window.alert(error.message));
fields.forEach(field => field.addEventListener('input', () => { settingsStatus.textContent = '有尚未套用的變更'; }));
algorithmSelect.addEventListener('change', () => { settingsStatus.textContent = '有尚未套用的變更'; });
document.querySelector('#settings').onsubmit = event => {
  event.preventDefault();
  const values = Object.fromEntries(fields.map(field => [field.name, field.name === 'seed' ? Number.parseInt(field.value, 10) : Number.parseFloat(field.value)]));
  values.algorithm = algorithmSelect.value;
  settingsStatus.textContent = '正在套用…';
  control('settings', values).then(state => { settingsStatus.textContent = '已套用並重新播放'; render(state); }).catch(error => { settingsStatus.textContent = '套用失敗'; window.alert(error.message); });
};
refresh(); setInterval(refresh, 350);
