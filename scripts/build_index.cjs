#!/usr/bin/env node
// build_index.cjs — crée ingested/index.json avec embeddings MiniLM (local, pas d'OpenAI)

const fs = require('fs');
const path = require('path');

// mini parse d’arguments
const args = process.argv.slice(2);
function getArg(name, def) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
}

const INGESTED_DIR = path.resolve(getArg('ingested', './ingested'));
const OUT_FILE = path.resolve(getArg('out', './ingested/index.json'));
const CHUNK_SIZE = parseInt(getArg('chunk', '900'), 10);

function readAllMarkdown(dir) {
  if (!fs.existsSync(dir)) return [];
  const files = fs.readdirSync(dir);
  const md = [];
  for (const f of files) {
    const p = path.join(dir, f);
    const stat = fs.statSync(p);
    if (stat.isDirectory()) {
      md.push(...readAllMarkdown(p));
    } else if (f.endsWith('.md') || f.endsWith('.txt')) {
      md.push(p);
    }
  }
  return md;
}

function chunkText(txt, size) {
  const chunks = [];
  let i = 0;
  while (i < txt.length) {
    const end = Math.min(i + size, txt.length);
    let cut = end;
    // essaie de couper sur une fin de phrase si possible
    const slice = txt.slice(i, end);
    const dot = slice.lastIndexOf('. ');
    if (dot > size * 0.6) cut = i + dot + 1;
    chunks.push(txt.slice(i, cut).trim());
    i = cut;
  }
  return chunks.filter(Boolean);
}

function cosine(a, b) {
  let s = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    s += x * y; na += x * x; nb += y * y;
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d ? s / d : 0;
}

(async () => {
  console.log(`[build_index] Lecture de ${INGESTED_DIR}`);
  const files = readAllMarkdown(INGESTED_DIR);
  if (!files.length) {
    console.error(`Aucun fichier trouvé dans ${INGESTED_DIR}`);
    process.exit(1);
  }
  console.log(`[build_index] ${files.length} fichier(s) .md`);

  // import ESM dynamiquement depuis CJS
  const { pipeline } = await import('@xenova/transformers');
  console.log(`[build_index] Chargement du modèle MiniLM…`);
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  console.log(`[build_index] Modèle prêt.`);

  const entries = [];
  let id = 0;

  for (const file of files) {
    const raw = fs.readFileSync(file, 'utf8');
    const base = path.basename(file).replace(/\.(md|txt)$/i, '');
    const source = base; // tu peux affiner l’étiquette source ici

    const chunks = chunkText(raw, CHUNK_SIZE);
    console.log(` - ${base}: ${chunks.length} chunk(s)`);

    for (const text of chunks) {
      const out = await extractor(text, { pooling: 'mean', normalize: true });
      const embedding = Array.from(out.data); // Float32Array -> Array
      entries.push({ id: id++, source, text, embedding });
    }
  }

  const index = {
    model: 'Xenova/all-MiniLM-L6-v2',
    dims: entries[0]?.embedding?.length || 384,
    count: entries.length,
    built_at: new Date().toISOString(),
    entries
  };

  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify(index));
  console.log(`[build_index] Écrit: ${OUT_FILE} (${entries.length} vecteurs)`);

  // petit test local: score du 1er vs 2e (optionnel)
  if (entries.length > 1) {
    const s = cosine(entries[0].embedding, entries[1].embedding).toFixed(3);
    console.log(`[build_index] Cosine(entries[0], entries[1]) = ${s}`);
  }
})().catch(err => {
  console.error(err);
  process.exit(1);
});
