import { readdir } from 'node:fs/promises';
import path from 'node:path';

export async function markdownFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  return (
    await Promise.all(
      entries.map((entry) =>
        entry.isDirectory()
          ? markdownFiles(path.join(dir, entry.name))
          : entry.name.endsWith('.md')
            ? [path.join(dir, entry.name)]
            : [],
      ),
    )
  ).flat();
}

export async function filesWithExtension(dir, extension) {
  const entries = await readdir(dir, { withFileTypes: true });
  return (
    await Promise.all(
      entries.map((entry) =>
        entry.isDirectory()
          ? filesWithExtension(path.join(dir, entry.name), extension)
          : entry.name.endsWith(extension)
            ? [path.join(dir, entry.name)]
            : [],
      ),
    )
  ).flat();
}
