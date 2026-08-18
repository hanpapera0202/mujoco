const blocks = [
  { id: 'infeed', kind: 'input', title: '投料設定', detail: 'Seed、批量、X/Y 範圍與皮帶速度' },
  { id: 'screen', kind: 'decision', title: 'CSPR 快篩', detail: '可達性、期限、固定障礙與關節限位' },
  { id: 'joint', kind: 'decision', title: 'BC-GP-JSP 中央聯合策略', detail: 'GA 分配、PSO 權重與貝式效用' },
  { id: 'a', kind: 'action', title: 'Nova5 A 任務', detail: '高位準備、跟帶、下探、雙指抓取' },
  { id: 'b', kind: 'action', title: 'Nova5 B 任務', detail: '高位準備、保留走廊、雙指抓取' },
  { id: 'verify', kind: 'decision', title: '物理驗證', detail: '碰撞盒、抓取接觸、托盤落點、事件紀錄' },
];
const palette = document.querySelector('#palette');
const flow = document.querySelector('#flow');
const output = document.querySelector('#json');
let sequence = blocks.map(block => block.id);
function renderBlock(block, source) {
  const element = document.createElement('article');
  element.className = 'block'; element.draggable = true; element.dataset.id = block.id; element.dataset.kind = block.kind;
  element.innerHTML = `<strong>${block.title}</strong><small>${block.detail}</small>`;
  element.addEventListener('dragstart', event => event.dataTransfer.setData('text/plain', block.id));
  if (source === 'palette') element.addEventListener('dblclick', () => { if (!sequence.includes(block.id)) { sequence.push(block.id); refresh(); } });
  return element;
}
function refresh() {
  palette.replaceChildren(...blocks.map(block => renderBlock(block, 'palette')));
  flow.replaceChildren(...sequence.map(id => renderBlock(blocks.find(block => block.id === id), 'flow')));
  output.value = JSON.stringify({ version: 1, algorithm: 'bc_jsp', pipeline: sequence, note: 'Exported by Nova5 visual blocks. Copy into VS Code as a research configuration.' }, null, 2);
}
flow.addEventListener('dragover', event => event.preventDefault());
flow.addEventListener('drop', event => { event.preventDefault(); const id = event.dataTransfer.getData('text/plain'); const target = event.target.closest('.block'); sequence = sequence.filter(item => item !== id); const index = target ? sequence.indexOf(target.dataset.id) : sequence.length; sequence.splice(Math.max(0, index), 0, id); refresh(); });
document.querySelector('#reset').onclick = () => { sequence = blocks.map(block => block.id); refresh(); };
document.querySelector('#copy').onclick = async () => { await navigator.clipboard.writeText(output.value); document.querySelector('#status').textContent = 'JSON 已複製，可在 VS Code 建立 research_pipeline.json 後貼上。'; };
refresh();
