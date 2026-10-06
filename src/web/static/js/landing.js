// Pipeline explorer (auto-plays, click to jump) and the results carousel.
(() => {
  const data = JSON.parse(document.getElementById('ex-data').textContent);
  const buttons = [...document.querySelectorAll('.explorer-list button')];
  const el = (id) => document.getElementById(id);
  const bar = el('ex-bar'), play = el('ex-play');
  const STEP_MS = 5000;
  let current = 0, started = performance.now(), playing = true;

  function show(i) {
    current = i; started = performance.now();
    const [, name, text, , llm, code, example] = data[i];
    el('ex-title').textContent = `${i + 1}. ${name}`;
    el('ex-text').textContent = text;
    el('ex-llm').textContent = llm;
    el('ex-code').textContent = code;
    el('ex-example').textContent = example;
    buttons.forEach((b, j) => b.classList.toggle('on', j === i));
  }
  buttons.forEach((b, i) => b.addEventListener('click', () => show(i)));
  play.addEventListener('click', () => {
    playing = !playing;
    play.textContent = playing ? '❚❚' : '▶';
    play.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    started = performance.now();
  });
  function tick(now) {
    if (playing) {
      const f = (now - started) / STEP_MS;
      bar.style.width = `${Math.min(f, 1) * 100}%`;
      if (f >= 1) show((current + 1) % data.length);
    }
    requestAnimationFrame(tick);
  }
  show(0);
  requestAnimationFrame(tick);

  const track = el('track');
  const by = (dir) => track.scrollBy({ left: dir * (track.firstElementChild.offsetWidth + 22), behavior: 'smooth' });
  el('prev').addEventListener('click', () => by(-1));
  el('next').addEventListener('click', () => by(1));
})();
