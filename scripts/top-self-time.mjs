// Lists the functions with the most self time in a CPU profile (`.cpuprofile`, as CDP writes it).
//
//   node scripts/top-self-time.mjs profile.cpuprofile [rows]
import { readFileSync } from 'node:fs';

const [file, rows = '25'] = process.argv.slice(2);
const profile = JSON.parse(readFileSync(file, 'utf8'));
const self = new Map(profile.nodes.map((n) => [n.id, 0]));
const dt = profile.timeDeltas;
profile.samples.forEach((id, i) => self.set(id, self.get(id) + (dt[i] ?? 0)));
const byName = new Map();
for (const n of profile.nodes) {
  const f = n.callFrame;
  const key = `${f.functionName || '(anonymous)'} ${f.url.split('/').slice(-2).join('/')}:${f.lineNumber + 1}`;
  byName.set(key, (byName.get(key) ?? 0) + self.get(n.id));
}
const total = [...byName.values()].reduce((a, b) => a + b, 0);
for (const [key, us] of [...byName].sort((a, b) => b[1] - a[1]).slice(0, Number(rows)))
  console.log(
    `${((100 * us) / total).toFixed(1).padStart(5)} %  ${(us / 1000).toFixed(0).padStart(7)} ms  ${key}`,
  );
