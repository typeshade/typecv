const button = document.querySelector('#theme');
let preference;
try {
  preference = localStorage.getItem('typecv-theme');
} catch {
  /* Optional storage. */
}
if (preference === 'dark' || preference === 'light')
  document.documentElement.dataset.theme = preference;
button?.addEventListener('click', () => {
  const dark =
    document.documentElement.dataset.theme === 'dark' ||
    (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem('typecv-theme', next);
  } catch {
    /* Optional storage. */
  }
});
