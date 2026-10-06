// Show the upload fields only when "Upload your own" is chosen.
(() => {
  const sync = () => {
    const chosen = document.querySelector('input[name="data"]:checked');
    document.body.dataset.data = chosen ? chosen.value : '';
  };
  const nct = document.getElementById('nct');
  document.querySelectorAll('input[name="data"]').forEach((r) =>
    r.addEventListener('change', () => { sync(); nct.value = r.dataset.nct || ''; }));
  sync();
})();
