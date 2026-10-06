// Live job page: poll the job, move the stepper, fill the log and the gates, ask the open question.
(() => {
  const root = document.getElementById('job');
  const api = root.dataset.api, answerUrl = root.dataset.answer, cancelUrl = root.dataset.cancel;
  const logEl = document.getElementById('log'), slot = document.getElementById('question-slot');
  const statusEl = document.getElementById('status'), statusText = document.getElementById('status-text');
  const versionsEl = document.getElementById('versions'), repairsCard = document.getElementById('repairs-card');
  const repairsEl = document.getElementById('repairs'), elapsedEl = document.getElementById('elapsed');
  const LABELS = { running: 'Running', waiting: 'Waiting for you', done: 'Finished', failed: 'Failed' };
  let cursor = 0, shownQuestion = null, finished = false;

  const h = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;   // textContent only: log lines are never parsed as HTML
    return e;
  };

  function appendLog(lines) {
    const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 30;
    for (const line of lines) {
      const cls = line.startsWith('==') ? 'h' : /: FAIL\s*$/.test(line) ? 'bad' : line.includes('applied:') ? 'act' : '';
      logEl.appendChild(h('span', cls, line + '\n'));
    }
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;
  }

  function renderStepper(s) {
    document.querySelectorAll('.stp').forEach((el) => {
      const n = Number(el.dataset.n);
      el.classList.remove('done', 'active', 'waiting');
      if (s.status === 'done' || n < s.step) el.classList.add('done');
      else if (n === s.step) el.classList.add('active', ...(s.status === 'waiting' ? ['waiting'] : []));
    });
  }

  function renderVersions(s) {
    if (!s.versions.length) return;
    versionsEl.replaceChildren();
    for (const v of s.versions) {
      const box = h('div', 'version');
      box.appendChild(h('h4', '', `Design version ${v.n}`));
      const chips = h('div', 'chips');
      for (const g of v.gates) chips.appendChild(h('span', `chip ${g.status}`, `${g.id} ${g.name}`));
      box.appendChild(chips);
      versionsEl.appendChild(box);
    }
    repairsCard.hidden = !s.repairs.length;
    repairsEl.replaceChildren(...s.repairs.map((r) => h('li', '', r)));
  }

  function send(url, body) {
    return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  }

  function renderQuestion(q) {
    if (!q) { slot.replaceChildren(); shownQuestion = null; return; }
    if (shownQuestion === q.id) return;
    shownQuestion = q.id;
    const box = h('div', 'question');
    box.appendChild(h('h3', '', q.title));
    box.appendChild(h('p', 'hint', q.hint));
    box.appendChild(h('pre', '', q.body));
    const actions = h('div', 'actions');
    const input = h('input'); input.type = 'text'; input.placeholder = 'or type a change, then press Send';
    const mk = (label, cls, fn) => { const b = h('button', `btn ${cls}`, label); b.type = 'button'; b.addEventListener('click', fn); return b; };
    const reply = async (answer) => {
      actions.querySelectorAll('button').forEach((b) => (b.disabled = true));
      const r = await send(answerUrl, { id: q.id, answer });
      if (!r.ok) shownQuestion = null;   // stale question: let the next poll redraw it
      poll();
    };
    actions.append(
      mk('Continue', 'btn-green', () => reply('y')),
      mk('Stop the run', 'btn-ghost', () => reply('n')),
      input,
      mk('Send', 'btn-black', () => input.value.trim() && reply(input.value.trim())),
    );
    box.appendChild(actions);
    slot.replaceChildren(box);
    box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function poll() {
    if (finished) return;
    let s;
    try { s = await (await fetch(`${api}?since=${cursor}`)).json(); } catch (e) { return; }
    cursor = s.cursor;
    appendLog(s.log);
    renderStepper(s);
    renderVersions(s);
    renderQuestion(s.pending);
    statusEl.className = `status ${s.status}`;
    statusText.textContent = LABELS[s.status];
    elapsedEl.textContent = `${Math.floor(s.elapsed / 60)}m ${s.elapsed % 60}s since start`;
    if (s.status === 'done' || s.status === 'failed') { finished = true; location.reload(); }
  }

  const cancel = document.getElementById('cancel');
  if (cancel) cancel.addEventListener('click', async () => { await send(cancelUrl); cancel.disabled = true; cancel.textContent = 'Stopping…'; });
  poll();
  setInterval(poll, 1200);
})();
