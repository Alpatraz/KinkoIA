// scripts/crawl_and_ingest_sources.js
// CommonJS – Node 18/20+
//
// Usage exemples :
//   node scripts/crawl_and_ingest_sources.js --only fitofan.com --firecrawl-only
//   node scripts/crawl_and_ingest_sources.js --only uve.zendesk.com --html-cap 120000
//   node scripts/crawl_and_ingest_sources.js --only wkccanada.com --timeout 15000
//
// ENV attendus (via .env.local ou l'environnement):
//   FIRECRAWL_API_KEY=fc-xxxxxxxxxxxxxxxxx
//   LLAMA_CLOUD_API_KEY=llx-xxxxxxxxxxxxxxx
//
// Conseils RAM : export NODE_OPTIONS="--max-old-space-size=6144 --expose-gc"

require('fs');
const fs = require('fs');
const path = require('path');

// Charge .env.local si présent (sans casser dotenvx si tu l'utilises déjà)
try {
  const dotenvPath = process.env.DOTENV_PATH || path.resolve(process.cwd(), '.env.local');
  if (fs.existsSync(dotenvPath)) {
    require('dotenv').config({ path: dotenvPath });
  }
} catch { /* noop */ }

// --- CLI args (sans dépendance) ------------------------------------------------
const argv = (() => {
  const out = {};
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = args[i + 1];
      if (!next || next.startsWith('--')) {
        out[key] = true;
      } else {
        out[key] = next;
        i++;
      }
    } else {
      (out._ || (out._ = [])).push(a);
    }
  }
  return out;
})();

const ONLY = argv.only || null;                  // ex: fitofan.com
const FORCE_FIRECRAWL = !!argv['firecrawl-only'];
const HTML_CAP = Number(argv['html-cap'] || 120000);
const TIMEOUT_MS = Number(argv['timeout'] || 20000);
const CONCURRENCY = Math.max(1, Number(argv['concurrency'] || 1));
const CSV_PATH = argv.csv || path.resolve(process.cwd(), 'sources.csv');
const OUT_DIR = path.resolve(process.cwd(), 'ingested');

const FIRECRAWL_KEY = process.env.FIRECRAWL_API_KEY || '';
const LLAMA_KEY = process.env.LLAMA_CLOUD_API_KEY || '';

const hasFirecrawl = !!FIRECRAWL_KEY;
const hasLlama = !!LLAMA_KEY;

// -----------------------------------------------------------------------------

function log(...a) { console.log(...a); }
function warn(...a) { console.warn(...a); }
function err(...a) { console.error(...a); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function sanitizeFilename(s) {
  return s
    .replace(/^https?:\/\//, '')
    .replace(/[^\w.-]+/g, '_')
    .slice(0, 180);
}

function isPdfUrl(u) {
  return /\.pdf($|\?)/i.test(u);
}

function getDomain(u) {
  try { return new URL(u).hostname; } catch { return ''; }
}

function cap(text, max = HTML_CAP) {
  if (!text) return '';
  const s = String(text);
  if (s.length <= max) return s;
  return s.slice(0, max);
}

function abortSignal(timeoutMs) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  return { signal: ctrl.signal, done: () => clearTimeout(t) };
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function writeOut(url, text, meta = {}) {
  ensureDir(OUT_DIR);
  const file = path.join(OUT_DIR, sanitizeFilename(url) + '.md');
  const head = [
    `---`,
    `source_url: ${url}`,
    `fetched_at: ${new Date().toISOString()}`,
    `chars: ${text.length}`,
    ...Object.entries(meta).map(([k, v]) => `${k}: ${v}`),
    `---`,
    ``,
  ].join('\n');
  fs.writeFileSync(file, head + text, 'utf8');
  log(`   ✔️  Saved: ${path.relative(process.cwd(), file)} (${text.length} chars)`);
}

// --- FETCHERS -----------------------------------------------------------------

// 1) Firecrawl (markdown minimal)
async function fetchWithFirecrawl(url) {
  if (!hasFirecrawl) throw new Error('FIRECRAWL_API_KEY manquant');

  const { signal, done } = abortSignal(TIMEOUT_MS);
  try {
    const res = await fetch('https://api.firecrawl.dev/v1/scrape', {
      method: 'POST',
      signal,
      headers: {
        'Authorization': `Bearer ${FIRECRAWL_KEY}`,
        'Content-Type': 'application/json',
        'Accept-Encoding': 'identity' // évite brotli → mémoire
      },
      body: JSON.stringify({
        url,
        formats: ['markdown'],
        onlyMainContent: true,
        screenshot: false,
        removeBase64: true,
      }),
    });
    if (!res.ok) throw new Error(`Firecrawl HTTP ${res.status}`);
    const j = await res.json().catch(() => ({}));
    const md = (j?.data?.markdown ?? j?.markdown ?? '');
    return cap(md);
  } finally {
    done();
  }
}

// 2) LlamaParse (PDF)
async function parsePdfWithLlama(url) {
  if (!hasLlama) throw new Error('LLAMA_CLOUD_API_KEY manquant');

  const { signal, done } = abortSignal(Math.max(TIMEOUT_MS, 60000)); // PDF = plus long
  try {
    // Endpoint "parse" par URL (simple et robuste)
    const res = await fetch('https://api.cloud.llamaindex.ai/api/v1/parse', {
      method: 'POST',
      signal,
      headers: {
        'Authorization': `Bearer ${LLAMA_KEY}`,
        'Content-Type': 'application/json',
        'Accept-Encoding': 'identity'
      },
      body: JSON.stringify({ urls: [url] })
    });
    if (!res.ok) throw new Error(`LlamaParse HTTP ${res.status}`);
    const j = await res.json().catch(() => ({}));
    const txt = (j?.data?.[0]?.text || j?.data?.[0]?.markdown || '');
    if (!txt) throw new Error('LlamaParse: texte vide');
    return cap(txt);
  } finally {
    done();
  }
}

// 3) Zendesk Help Center API (léger)
function tryBuildZendeskApiUrl(hcUrl) {
  // Ex: https://uve.zendesk.com/hc/en-us/sections/39296282019603-FAQ
  try {
    const u = new URL(hcUrl);
    if (!/zendesk\.com$/i.test(u.hostname)) return null;
    const parts = u.pathname.split('/').filter(Boolean); // ['hc','en-us','sections','392...-FAQ']
    const hcIdx = parts.indexOf('hc');
    if (hcIdx === -1 || parts.length < hcIdx + 4) return null;
    const locale = parts[hcIdx + 1] || 'en-us';
    const type = parts[hcIdx + 2]; // sections|categories
    const idSlug = parts[hcIdx + 3] || '';
    const id = idSlug.split('-')[0]; // 39296282019603
    if (!/^\d+$/.test(id)) return null;

    if (type === 'sections') {
      return `${u.origin}/api/v2/help_center/${locale}/sections/${id}/articles.json?per_page=100&page=1`;
    }
    if (type === 'categories') {
      return `${u.origin}/api/v2/help_center/${locale}/categories/${id}/articles.json?per_page=100&page=1`;
    }
    return null;
  } catch { return null; }
}

async function fetchZendeskArticlesList(apiUrl) {
  const { signal, done } = abortSignal(TIMEOUT_MS);
  try {
    const res = await fetch(apiUrl, {
      method: 'GET',
      signal,
      headers: { 'Accept': 'application/json', 'Accept-Encoding': 'identity' }
    });
    if (!res.ok) throw new Error(`Zendesk API HTTP ${res.status}`);
    const j = await res.json().catch(() => ({}));
    const articles = j?.articles || j?.data || [];
    const chunks = [];
    for (const a of articles) {
      const title = a.title || a.name || '';
      const body = a.body || a.content || '';
      const url = a.html_url || a.url || '';
      const t = `# ${title}\n\n${body}\n\n[Source](${url})\n`;
      chunks.push(cap(stripHtml(t)));
    }
    return chunks.join('\n\n---\n\n');
  } finally {
    done();
  }
}

// 4) Fallback HTML fetch + extraction simple
function stripHtml(html) {
  if (!html) return '';
  // enlève scripts/styles et réduit l’espace
  return String(html)
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

async function fetchHtmlAsText(url) {
  const { signal, done } = abortSignal(TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      method: 'GET',
      signal,
      headers: { 'Accept': 'text/html,*/*', 'Accept-Encoding': 'identity' }
    });
    if (!res.ok) throw new Error(`Fetch HTTP ${res.status}`);
    const html = await res.text();
    return cap(stripHtml(html));
  } finally {
    done();
  }
}

// -----------------------------------------------------------------------------

function loadCsv(urlOnly = false) {
  const p = CSV_PATH;
  if (!fs.existsSync(p)) {
    err(`❌ CSV introuvable: ${p}`);
    process.exit(1);
  }
  const raw = fs.readFileSync(p, 'utf8');
  // Auto-détection délimiteur ; ou ,
  const delimiter = raw.includes(';') ? ';' : ',';
  const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const rows = [];
  for (const l of lines) {
    const parts = l.split(delimiter).map(s => s.trim());
    // on prend la 1re cellule comme URL par convention
    if (parts[0] && /^https?:\/\//i.test(parts[0])) rows.push({ url: parts[0], cells: parts });
  }
  log(`📄 CSV: ${rows.length} lignes chargées (delimiter="${delimiter}")`);
  return rows;
}

async function processUrl(u) {
  const domain = getDomain(u);

  log(`\n⏳ Fetch: ${u}`);

  // PDFs → LlamaParse si possible
  if (isPdfUrl(u)) {
    log(`   → PDF détecté`);
    if (!hasLlama) {
      warn(`   ⚠️ Pas de LLAMA_CLOUD_API_KEY – PDF sauté pour: ${u}`);
      return;
    }
    const txt = await parsePdfWithLlama(u);
    const meta = { via: 'llamaparse', type: 'pdf' };
    writeOut(u, txt, meta);
    return;
  }

  // Zendesk API (léger) si possible
  const zendeskApi = tryBuildZendeskApiUrl(u);
  if (zendeskApi && !FORCE_FIRECRAWL) {
    try {
      log(`   → Zendesk API: ${zendeskApi}`);
      const txt = await fetchZendeskArticlesList(zendeskApi);
      if (txt && txt.length) {
        writeOut(u, txt, { via: 'zendesk_api' });
        return;
      }
    } catch (e) {
      warn(`   ⚠️ Zendesk API a échoué (${e.message}). On tente autre chose…`);
    }
  }

  // Firecrawl (par défaut, ou si --firecrawl-only)
  if (hasFirecrawl) {
    try {
      log(`   → Firecrawl pour ${domain}`);
      const md = await fetchWithFirecrawl(u);
      if (md && md.length) {
        writeOut(u, md, { via: 'firecrawl', format: 'markdown' });
        return;
      }
    } catch (e) {
      warn(`   ⚠️ Firecrawl a échoué (${e.message}). Fallback…`);
    }
  } else if (FORCE_FIRECRAWL) {
    warn(`   ⚠️ --firecrawl-only requis mais FIRECRAWL_API_KEY absent. URL sautée.`);
    return;
  }

  // Fallback: HTML → texte simple
  try {
    const txt = await fetchHtmlAsText(u);
    writeOut(u, txt, { via: 'html_fallback' });
  } catch (e) {
    warn(`   ⚠️ Fallback HTML a échoué: ${e.message}`);
  }
  finally {
    if (global.gc) { try { global.gc(); } catch {} }
  }
}

// Concurrence basique
async function runQueue(urls) {
  let i = 0;
  let active = 0;
  let done = 0;

  return new Promise((resolve) => {
    const next = () => {
      if (done === urls.length) return resolve();

      while (active < CONCURRENCY && i < urls.length) {
        const u = urls[i++];
        active++;
        processUrl(u)
          .catch(e => warn(`   ⚠️ Erreur: ${e.message}`))
          .finally(async () => {
            done++;
            active--;
            if (global.gc) { try { global.gc(); } catch {} }
            next();
          });
      }
    };
    next();
  });
}

// -----------------------------------------------------------------------------

(async function main() {
  try {
    log(`[dotenv] ENV chargées. 🔑 Firecrawl: ${hasFirecrawl ? FIRECRAWL_KEY.slice(0, 7) + '…' : '—'}${hasLlama ? ` | LlamaParse: ${LLAMA_KEY.slice(0, 7)}…` : ''}`);

    const rows = loadCsv();
    let urls = rows.map(r => r.url);

    if (ONLY) {
      urls = urls.filter(u => getDomain(u).includes(ONLY));
    }

    if (urls.length === 0) {
      log('Rien à traiter (filtre trop strict ?)'); 
      process.exit(0);
    }

    log(`\n▶️  Démarrage – ${urls.length} URL${urls.length > 1 ? 's' : ''}`);
    log(`   Options: cap=${HTML_CAP} | timeout=${TIMEOUT_MS}ms | concurrency=${CONCURRENCY} | firecrawl-only=${FORCE_FIRECRAWL}`);

    ensureDir(OUT_DIR);
    await runQueue(urls);

    log('\n✅ Terminé.');
  } catch (e) {
    err(`\n❌ Fatal: ${e.stack || e.message}`);
    process.exit(1);
  }
})();
