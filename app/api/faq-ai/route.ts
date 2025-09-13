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
  text: string; // contenu brut du chunk
  tokens?: number;
};

type IndexFile = {
  chunks: IndexChunk[];
};

type Retrieved = {
  chunk: IndexChunk;
  score: number;
};

type EventItem = {
  title?: string;
  start?: string; // ISO date
  end?: string;   // ISO date
  location?: string;
  url?: string;
};

/* ==== Types GraphQL Shopify (réponse Metaobjects) ==== */
type ShopifyField = { key: string; value: string };

type ShopifyMetaobject = {
  id?: string;
  handle?: string;
  fields?: ShopifyField[];
};

type ShopifyMetaobjectsResponse = {
  data?: {
    metaobjects?: {
      edges?: Array<{ node?: ShopifyMetaobject }>;
    };
  };
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
   Helpers URLs + Markdown→HTML
========================= */

function publicBase(): string {
  const raw = process.env.SHOPIFY_PUBLIC_BASE?.trim() || "";
  if (!raw) return "";
  // retire www. et slash final
  let s = raw.replace(/^https?:\/\/www\./i, "https://").replace(/\/+$/, "");
  if (!/^https?:\/\//i.test(s)) s = "https://" + s;
  return s;
}

function autoLinkMarkdown(s: string): string {
  // [label](url) -> <a href="url">label</a>
  const withMd = s.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_m, label, url) =>
      `<a href="${url}" target="_blank" rel="noopener">${label}</a>`
  );
  // liens bruts -> <a>
  return withMd.replace(
    /(https?:\/\/[^\s)]+)(?![^<]*>)/g,
    (m) => `<a href="${m}" target="_blank" rel="noopener">${m}</a>`
  );
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
    idf.set(k, Math.log((1 + N) / (1 + v)) + 1); // IDF lissé
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
   OpenRouter (fallback multi-modèles)
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
      "HTTP-Referer": process.env.OPENROUTER_SITE_URL ?? "https://example.com",
      "X-Title": process.env.OPENROUTER_SITE_NAME ?? "Kinko FAQ AI",
      "Cache-Control": "no-store",
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

/* =========================
   TEMPS RÉEL : metaobjects "event" (prochaine compétition)
========================= */

function shopDomainFromBase(): string | null {
  const base = publicBase();
  if (!base) return null;
  try {
    const u = new URL(base);
    return u.hostname; // ex: qfxdmn-i3.myshopify.com
  } catch {
    return null;
  }
}

// Tente de repérer les clés probables dans tes metaobjects
function parseEventFields(fields: ShopifyField[]): EventItem {
  const map: Record<string, string> = {};
  for (const f of fields) map[f.key.toLowerCase()] = f.value;

  const title =
    map["title"] || map["name"] || map["nom"] || map["event"] || map["evenement"];
  const start =
    map["start"] || map["date"] || map["start_date"] || map["date_debut"] || map["debut"];
  const end =
    map["end"] || map["end_date"] || map["date_fin"] || map["fin"];
  const location = map["location"] || map["lieu"] || map["city"] || map["ville"];
  const url = map["url"] || map["register_url"] || map["inscription"] || map["lien"];

  return { title, start, end, location, url };
}

async function fetchNextEvent(): Promise<EventItem | null> {
  const shopHost = shopDomainFromBase();
  const token = process.env.SHOPIFY_ADMIN_TOKEN;
  if (!shopHost || !token) return null;

  const apiVersion = "2024-10";
  const endpoint = `https://${shopHost}/admin/api/${apiVersion}/graphql.json`;

  const query = `
    query Events($type: String!, $first: Int!) {
      metaobjects(type: $type, first: $first) {
        edges {
          node {
            id
            handle
            fields { key value }
          }
        }
      }
    }`;

  const resp = await fetch(endpoint, {
    method: "POST",
    headers: {
      "X-Shopify-Access-Token": token,
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
    body: JSON.stringify({ query, variables: { type: "event", first: 100 } }),
  });

  if (!resp.ok) return null;

  const dataUnknown = await resp.json().catch(() => null) as unknown;
  const data = dataUnknown as ShopifyMetaobjectsResponse;

  const edges = data?.data?.metaobjects?.edges ?? [];
  if (!Array.isArray(edges) || edges.length === 0) return null;

  // Convertit et filtre pour l’événement à venir le plus proche
  const now = new Date();
  const upcoming: Array<{ item: EventItem; sortKey: number }> = [];

  for (const e of edges) {
    const fields = e?.node?.fields ?? [];
    if (!Array.isArray(fields) || fields.length === 0) continue;
    const it = parseEventFields(fields);
    if (!it.start) continue;
    const start = new Date(it.start);
    if (Number.isNaN(+start)) continue;
    if (start >= now) {
      upcoming.push({ item: it, sortKey: +start });
    }
  }

  if (upcoming.length === 0) return null;
  upcoming.sort((a, b) => a.sortKey - b.sortKey);
  return upcoming[0].item;
}

function formatDateFR(iso?: string): string | undefined {
  if (!iso) return;
  const d = new Date(iso);
  if (Number.isNaN(+d)) return;
  return d.toLocaleDateString("fr-CA", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function buildNextEventAnswerHTML(ev: EventItem): string {
  const start = formatDateFR(ev.start);
  const end = formatDateFR(ev.end);
  const base = publicBase();
  const cal = base ? `${base}/pages/calendrier` : "";

  const parts: string[] = [];
  parts.push(`<strong>Prochaine compétition :</strong>`);
  if (ev.title) parts.push(`<div>• ${ev.title}</div>`);
  if (start && end) parts.push(`<div>• Dates : du ${start} au ${end}</div>`);
  else if (start) parts.push(`<div>• Date : ${start}</div>`);
  if (ev.location) parts.push(`<div>• Lieu : ${ev.location}</div>`);
  if (ev.url) parts.push(`<div>• Inscriptions : <a href="${ev.url}" target="_blank" rel="noopener">${ev.url}</a></div>`);
  if (cal) parts.push(`<div>• Calendrier : <a href="${cal}" target="_blank" rel="noopener">${cal}</a></div>`);
  return parts.join("");
}

function looksLikeNextEventQuestion(q: string): boolean {
  const s = q.toLowerCase();
  return /(prochain(e)?|date).*(comp(é|e)tition|tournoi|év(é|e)nement)/i.test(s);
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

    // 1) Réponse TEMPS RÉEL pour "prochaine compétition"
    if (looksLikeNextEventQuestion(q)) {
      const ev = await fetchNextEvent();
      if (ev) {
        const html = buildNextEventAnswerHTML(ev);
        return new NextResponse(JSON.stringify({ answer: html }), { status: 200, headers });
      }
      // si rien trouvé, on continue vers le RAG/LLM
    }

    // 2) RAG indexé
    const index = await loadIndex();
    const retrieved = topK(index, q, 6);

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

    // 3) Post-traitement : corriger host (enlever www) + liens cliquables
    const base = publicBase();
    const sanitized = answer
      .replace(/https?:\/\/www\./gi, "https://") // retire www.
      .replace(/https?:\/\/[^)\s]+myshopify\.com/gi, (m) => {
        // force le host vers SHOPIFY_PUBLIC_BASE si dispo
        return base ? base : m.replace(/\/+$/, "");
      });

    const htmlOut = autoLinkMarkdown(sanitized);

    return new NextResponse(JSON.stringify({ answer: htmlOut }), { status: 200, headers });
  } catch (e: unknown) {
    console.error("[faq-ai] error:", e);
    const message = e instanceof Error ? e.message : "Unknown error";
    return new NextResponse(JSON.stringify({ error: message }), { status: 500, headers });
  }
}
