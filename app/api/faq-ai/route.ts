// app/api/faq-ai/route.ts
import { NextRequest, NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

/* =========================
   Types
========================= */

type FaqAiRequest = { q: string; lang?: string };

type IndexChunk = {
  id: string;
  url: string;
  title?: string;
  text: string;
  tokens?: number;
};

type IndexFile = {
  chunks: IndexChunk[];
};

type Retrieved = {
  chunk: IndexChunk;
  score: number;
};

/* =========================
   CORS utils
========================= */

function corsHeaders(req: NextRequest): Headers {
  const origin = req.headers.get("origin") ?? "*";
  const h = new Headers();
  h.set("Access-Control-Allow-Origin", origin);
  h.set("Vary", "Origin");
  h.set("Access-Control-Allow-Methods", "OPTIONS, POST");
  h.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
  h.set("Access-Control-Max-Age", "86400");
  h.set("Content-Type", "application/json; charset=utf-8");
  return h;
}

export async function OPTIONS(req: NextRequest) {
  return new NextResponse(null, { status: 204, headers: corsHeaders(req) });
}

/* =========================
   Lecture de l'index / fallback
========================= */

let cachedIndex: IndexFile | null = null;

async function loadIndex(): Promise<IndexFile> {
  if (cachedIndex) return cachedIndex;

  const root = process.cwd();
  const indexPath = path.join(root, "ingested", "index.json");
  const ingestedDir = path.join(root, "ingested");

  // 1) Essayer index.json
  try {
    const buf = await fs.readFile(indexPath, "utf8");
    const parsed = JSON.parse(buf) as unknown;

    if (
      parsed &&
      typeof parsed === "object" &&
      "chunks" in (parsed as Record<string, unknown>) &&
      Array.isArray((parsed as Record<string, unknown>).chunks)
    ) {
      const chunks = (parsed as { chunks: unknown[] }).chunks.map((c, i) => {
        const obj = c as Record<string, unknown>;
        return {
          id: String(obj.id ?? i.toString()),
          url: String(obj.url ?? ""),
          title: obj.title ? String(obj.title) : undefined,
          text: String(obj.text ?? (obj as Record<string, unknown>).content ?? ""),
          tokens: typeof obj.tokens === "number" ? obj.tokens : undefined,
        } satisfies IndexChunk;
      });
      cachedIndex = { chunks };
      return cachedIndex;
    }
  } catch {
    // ignore ; tentative fallback dessous
  }

  // 2) Fallback: lire tous les .md de /ingested et créer 1 chunk par fichier
  const chunks: IndexChunk[] = [];
  try {
    const files = await fs.readdir(ingestedDir);
    for (const file of files) {
      if (!file.toLowerCase().endsWith(".md")) continue;
      const full = path.join(ingestedDir, file);
      const content = await fs.readFile(full, "utf8");
      const guessedUrl = file
        .replaceAll("_", "/")
        .replace(/\.md$/i, "")
        .replace(/^https?:\/\//i, "");
      chunks.push({
        id: file,
        url: "https://" + guessedUrl,
        title: file,
        text: content,
      });
    }
  } catch {
    // pas de .md non plus → index vide
  }

  cachedIndex = { chunks };
  return cachedIndex;
}

/* =========================
   Récupération : scoring simple TF-IDF-ish
========================= */

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function buildIdfMap(chunks: IndexChunk[]): Map<string, number> {
  const df = new Map<string, number>();
  const N = Math.max(1, chunks.length);
  for (const ch of chunks) {
    const seen = new Set<string>();
    for (const t of new Set(tokenize(ch.text))) {
      if (seen.has(t)) continue;
      seen.add(t);
      df.set(t, (df.get(t) ?? 0) + 1);
    }
  }
  const idf = new Map<string, number>();
  df.forEach((v, k) => {
    idf.set(k, Math.log((1 + N) / (1 + v)) + 1);
  });
  return idf;
}

function scoreChunk(queryTokens: string[], ch: IndexChunk, idf: Map<string, number>): number {
  if (!ch.text) return 0;
  const tokens = tokenize(ch.text);
  if (tokens.length === 0) return 0;

  // TF
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);

  // Score = somme TF * IDF pour les tokens de la requête
  let score = 0;
  for (const qt of queryTokens) {
    const f = tf.get(qt) ?? 0;
    const w = idf.get(qt) ?? 0;
    score += f * w;
  }

  // Bonus si le titre matche
  if (ch.title) {
    const titleTokens = new Set(tokenize(ch.title));
    let hits = 0;
    for (const qt of queryTokens) if (titleTokens.has(qt)) hits++;
    score *= 1 + Math.min(0.3, hits * 0.05);
  }

  return score;
}

function topK(context: IndexFile, q: string, k = 6): Retrieved[] {
  const chunks = context.chunks ?? [];
  if (chunks.length === 0) return [];
  const qTokens = tokenize(q).slice(0, 24);
  const idf = buildIdfMap(chunks);

  const scored: Retrieved[] = chunks
    .map((ch) => ({ chunk: ch, score: scoreChunk(qTokens, ch, idf) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);

  return scored;
}

/* =========================
   Modèles OpenRouter (fallback multi-modèles)
========================= */

const DEFAULT_MODEL_LIST =
  "qwen/qwen-2.5-72b-instruct:free,google/gemma-2-9b-it:free,mistralai/mistral-nemo:free";

function parseModelList(s: string): string[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

const MODEL_LIST: string[] = process.env.RAG_MODEL
  ? parseModelList(process.env.RAG_MODEL)
  : parseModelList(DEFAULT_MODEL_LIST);

const DEFAULT_MODEL: string = MODEL_LIST[0] ?? "google/gemma-2-9b-it:free";

/* =========================
   Prompt & appel OpenRouter
========================= */

function systemPrompt(siteName: string | undefined): string {
  const tag = siteName ? ` pour ${siteName}` : "";
  return [
    `Tu es “Sempaï Kinko”, un assistant d’aide et de vente${tag}.`,
    `Objectif : répondre clairement, brièvement, et utilement.`,
    `Règles :`,
    `- Langue : réponds dans la langue demandée (fr par défaut si non précisé).`,
    `- Si la réponse n'est pas dans le contexte, dis-le simplement et propose d'aider à la trouver (pas d'invention).`,
    `- Quand c’est pertinent, oriente vers l’achat/inscription/contact.`,
    `- Ajoute une courte section "Sources" avec 1–3 liens pertinents tirés du contexte (pas d’autres liens).`,
  ].join("\n");
}

function buildUserPrompt(q: string, lang: string | undefined, retrieved: Retrieved[]): string {
  const ctx = retrieved
    .map((r, i) => {
      const head = r.chunk.title ? `${r.chunk.title} — ${r.chunk.url}` : r.chunk.url;
      const body = r.chunk.text.slice(0, 4000);
      return `[#${i + 1}] ${head}\n${body}`;
    })
    .join("\n\n---\n\n");

  const sources = Array.from(new Set(retrieved.map((r) => r.chunk.url))).slice(0, 3);

  return [
    `Question: ${q}`,
    lang ? `Langue attendue: ${lang}` : `Langue attendue: fr`,
    ``,
    `Contexte (extraits provenant du site) :`,
    `${ctx || "(aucun extrait pertinent trouvé)"}`,
    ``,
    `Consignes de réponse :`,
    `- Réponds directement à la question, sans meta-commentaires.`,
    `- Si l'info n'est pas dans le contexte, dis-le et propose une alternative concrète (page à visiter, contact, etc.).`,
    `- Termine par une section "Sources" avec ces liens uniquement (si disponibles) :`,
    sources.length ? sources.map((u) => `- ${u}`).join("\n") : `- (Aucune source disponible)`,
  ].join("\n");
}

async function callOpenRouterChat(
  apiKey: string,
  model: string,
  system: string,
  user: string
): Promise<string> {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      // Recommandés par OpenRouter (analytics/quotas)
      "HTTP-Referer": process.env.OPENROUTER_SITE_URL ?? "https://example.com",
      "X-Title": process.env.OPENROUTER_SITE_NAME ?? "Kinko FAQ AI",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature: 0.3,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`OpenRouter error ${res.status}: ${text}`);
  }

  type Choice = { message: { role: "assistant"; content: string } };
  type ORResponse = { choices: Choice[] };

  const json = (await res.json()) as ORResponse;
  const content = json.choices?.[0]?.message?.content?.trim();
  if (!content) throw new Error("Réponse vide du modèle.");
  return content;
}

async function answerWithFallback(
  apiKey: string,
  system: string,
  user: string
): Promise<{ answer: string; modelUsed: string }> {
  const tried: string[] = [];
  for (const m of MODEL_LIST) {
    try {
      const ans = await callOpenRouterChat(apiKey, m, system, user);
      return { answer: ans, modelUsed: m };
    } catch {
      tried.push(m);
    }
  }
  const lastModel = DEFAULT_MODEL;
  if (!tried.includes(lastModel)) {
    const ans = await callOpenRouterChat(apiKey, lastModel, system, user);
    return { answer: ans, modelUsed: lastModel };
  }
  throw new Error(`Tous les modèles ont échoué: ${tried.join(", ")}`);
}

/* ========= Temps réel Shopify : prochaine compétition ========= */

type ShopifyField = { key: string; value: string | null };
type ShopifyMetaNode = {
  id: string;
  handle: string;
  type: string;
  updatedAt: string;
  fields: ShopifyField[];
};

type NextEvent = {
  id: string;
  title: string;
  dateISO: string;
  location?: string;
  link?: string;
  source?: string;
};

const SHOPIFY_SHOP = process.env.SHOPIFY_SHOP ?? "";
const SHOPIFY_ADMIN_TOKEN = process.env.SHOPIFY_ADMIN_TOKEN ?? "";
const SHOPIFY_API_VERSION = process.env.SHOPIFY_API_VERSION ?? "2024-07";
const SHOPIFY_PUBLIC_BASE = (process.env.SHOPIFY_PUBLIC_BASE ?? "").replace(/\/+$/, "");
const SHOPIFY_META_TYPES = (process.env.SHOPIFY_METAOBJECT_TYPES ?? "event,competition")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
const SHOPIFY_CALENDAR_URL = process.env.SHOPIFY_CALENDAR_URL ?? "";
const SHOPIFY_GQL_URL = SHOPIFY_SHOP
  ? `https://${SHOPIFY_SHOP}/admin/api/${SHOPIFY_API_VERSION}/graphql.json`
  : "";

async function shopifyAdminGQL<T extends Record<string, unknown>>(
  query: string,
  variables?: Record<string, unknown>
): Promise<T> {
  if (!SHOPIFY_SHOP || !SHOPIFY_ADMIN_TOKEN) {
    throw new Error("SHOPIFY_SHOP ou SHOPIFY_ADMIN_TOKEN manquant.");
  }
  const res = await fetch(SHOPIFY_GQL_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN,
    },
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error Next runtime accepte 'no-store' côté serveur
    cache: "no-store",
    body: JSON.stringify({ query, variables }),
  });
  const json = (await res.json()) as { data?: T; errors?: unknown };
  if (!res.ok || json.errors) {
    throw new Error(`Shopify GraphQL error ${res.status}: ${JSON.stringify(json.errors ?? json)}`);
  }
  if (!json.data) throw new Error("Réponse Shopify vide.");
  return json.data;
}

function extractEventLike(fields: ShopifyField[]) {
  const byKey = new Map<string, string>();
  for (const f of fields) if (f.value) byKey.set(f.key.toLowerCase(), f.value);

  const pick = (candidates: string[]): string | undefined => {
    for (const k of candidates) {
      const v = byKey.get(k);
      if (v && String(v).trim()) return String(v);
    }
    const rx = new RegExp(candidates.join("|"), "i");
    for (const [k, v] of byKey) if (rx.test(k) && v.trim()) return v.trim();
    return undefined;
  };

  const title = pick(["title", "name", "nom", "titre"]) ?? "Compétition";
  const dateRaw = pick(["date", "start_date", "start", "when", "date_time", "datetime"]);
  const location = pick(["location", "lieu", "city", "place", "ville"]);
  const link = pick(["url", "link", "registration", "inscription"]);

  return { title, dateRaw, location, link };
}

function toISO(s?: string): string | null {
  if (!s) return null;
  const d1 = new Date(s);
  if (!Number.isNaN(d1.getTime())) return d1.toISOString();
  const m = s.match(/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/);
  if (m) {
    return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])).toISOString();
  }
  return null;
}

function formatFR(iso: string): string {
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
      timeZone: "UTC",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

async function fetchNextEvent(): Promise<NextEvent | null> {
  if (!SHOPIFY_META_TYPES.length) return null;

  const q = `
    query Meta($type: String!, $cursor: String) {
      metaobjects(type: $type, first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id handle type updatedAt
          fields { key value }
        }
      }
    }
  `;

  const now = Date.now();
  let best:
    | { iso: string; node: ShopifyMetaNode; meta: ReturnType<typeof extractEventLike> }
    | null = null;

  for (const type of SHOPIFY_META_TYPES) {
    let cursor: string | null = null;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const data = await shopifyAdminGQL<{
        metaobjects: {
          pageInfo: { hasNextPage: boolean; endCursor: string | null };
          nodes: ShopifyMetaNode[];
        };
      }>(q, { type, cursor });

      const { nodes, pageInfo } = data.metaobjects;

      for (const n of nodes) {
        const meta = extractEventLike(n.fields);
        const iso = toISO(meta.dateRaw);
        if (!iso) continue;
        const ts = new Date(iso).getTime();
        if (Number.isNaN(ts) || ts < now) continue;
        if (!best || ts < new Date(best.iso).getTime()) {
          best = { iso, node: n, meta };
        }
      }

      if (!pageInfo.hasNextPage) break;
      cursor = pageInfo.endCursor;
      await new Promise((r) => setTimeout(r, 120));
    }
  }

  if (!best) return null;

  const publicSource =
    best.meta.link ||
    SHOPIFY_CALENDAR_URL ||
    (SHOPIFY_PUBLIC_BASE ? `${SHOPIFY_PUBLIC_BASE}/pages/calendrier` : "");

  return {
    id: best.node.id,
    title: best.meta.title,
    dateISO: best.iso,
    location: best.meta.location,
    link: best.meta.link || undefined,
    source: publicSource || undefined,
  };
}

/* =========================
   Handler POST
========================= */

export async function POST(req: NextRequest) {
  const headers = corsHeaders(req);

  try {
    const { q, lang } = (await req.json()) as FaqAiRequest;
    if (!q || typeof q !== "string") {
      return new NextResponse(JSON.stringify({ error: "Paramètre 'q' manquant." }), {
        status: 400,
        headers,
      });
    }

    const index = await loadIndex();

    // Détection d’intention “prochaine compétition / date tournoi…”
    const wantsNext =
      /prochain|prochaine|next|bient[oô]t|date|quand/i.test(q) &&
      /(comp[ée]tition|tournoi|event|év[ée]nement)/i.test(q);

    let retrieved = topK(index, q, 6);

    // Chunk “temps réel” Shopify si pertinent
    if (wantsNext) {
      try {
        const ev = await fetchNextEvent();
        if (ev) {
          const prettyDate = formatFR(ev.dateISO);
          const lines = [
            `Prochaine compétition (temps réel Shopify)`,
            `Titre : ${ev.title}`,
            `Date : ${prettyDate}`,
            ev.location ? `Lieu : ${ev.location}` : ``,
            ev.link ? `Inscription : ${ev.link}` : ``,
            ev.source ? `Source : ${ev.source}` : ``,
          ]
            .filter(Boolean)
            .join("\n");

          const realtimeChunk: IndexChunk = {
            id: `realtime_event_${ev.id}`,
            url: ev.link || ev.source || SHOPIFY_PUBLIC_BASE || "",
            title: `Prochaine compétition : ${ev.title}`,
            text: lines,
          };

          retrieved = [{ chunk: realtimeChunk, score: 1e9 }, ...retrieved];
        }
      } catch {
        // silencieux : on retombe sur RAG normal
      }
    }

    const sys = systemPrompt(process.env.RAG_SITE_NAME);
    const user = buildUserPrompt(q, lang, retrieved);

    const apiKey = process.env.OPENROUTER_API_KEY ?? "";
    if (!apiKey) {
      return new NextResponse(
        JSON.stringify({
          error:
            "OPENROUTER_API_KEY manquant. Ajoutez-le aux variables d’environnement du projet Vercel.",
        }),
        { status: 500, headers }
      );
    }

    const { answer } = await answerWithFallback(apiKey, sys, user);
    return new NextResponse(JSON.stringify({ answer }), { status: 200, headers });
  } catch (e: unknown) {
    console.error("[faq-ai] error:", e);
    const message = e instanceof Error ? e.message : "Unknown error";
    return new NextResponse(JSON.stringify({ error: message }), { status: 500, headers });
  }
}
