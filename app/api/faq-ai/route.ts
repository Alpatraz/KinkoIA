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

type IndexFile = { chunks: IndexChunk[] };
type Retrieved = { chunk: IndexChunk; score: number };

type ShopifyField = { key: string; value: string };
type ShopifyMetaobject = { id?: string; handle?: string; type?: string; fields?: ShopifyField[] };

type ShopifyMetaobjectDefinitionsResp = {
  data?: {
    metaobjectDefinitions?: {
      edges?: Array<{ node?: { type?: string } }>;
    };
  };
  errors?: unknown;
};

type ShopifyMetaobjectsResp = {
  data?: {
    metaobjects?: {
      edges?: Array<{ node?: ShopifyMetaobject }>;
    };
  };
  errors?: unknown;
};

type EventItem = {
  title?: string;
  start?: string; // ISO
  end?: string;   // ISO
  location?: string;
  url?: string;
  organizer?: string;
};

/* =========================
   CORS
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
  const raw = (process.env.SHOPIFY_PUBLIC_BASE || "").trim();
  if (!raw) return "";
  let s = raw.replace(/^https?:\/\/www\./i, "https://").replace(/\/+$/, "");
  if (!/^https?:\/\//i.test(s)) s = "https://" + s;
  return s;
}

function autoLinkMarkdown(s: string): string {
  const withMd = s.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_m, label, url) => `<a href="${url}" target="_blank" rel="noopener">${label}</a>`
  );
  return withMd.replace(
    /(https?:\/\/[^\s)]+)(?![^<]*>)/g,
    (m) => `<a href="${m}" target="_blank" rel="noopener">${m}</a>`
  );
}

/* =========================
   Lecture index
========================= */

let cachedIndex: IndexFile | null = null;

async function loadIndex(): Promise<IndexFile> {
  if (cachedIndex) return cachedIndex;
  const root = process.cwd();
  const indexPath = path.join(root, "ingested", "index.json");
  const ingestedDir = path.join(root, "ingested");

  try {
    const buf = await fs.readFile(indexPath, "utf8");
    const parsed = JSON.parse(buf) as { chunks?: unknown[] };
    if (parsed?.chunks && Array.isArray(parsed.chunks)) {
      const chunks: IndexChunk[] = parsed.chunks.map((c, i) => {
        const obj = c as Record<string, unknown>;
        return {
          id: String(obj.id ?? i),
          url: String(obj.url ?? ""),
          title: obj.title ? String(obj.title) : undefined,
          text: String(obj.text ?? (obj as Record<string, unknown>).content ?? ""),
          tokens: typeof obj.tokens === "number" ? obj.tokens : undefined,
        };
      });
      cachedIndex = { chunks };
      return cachedIndex;
    }
  } catch { /* fallback */ }

  const chunks: IndexChunk[] = [];
  try {
    const files = await fs.readdir(ingestedDir);
    for (const file of files) {
      if (!file.toLowerCase().endsWith(".md")) continue;
      const full = path.join(ingestedDir, file);
      const content = await fs.readFile(full, "utf8");
      const guessedUrl = file.replaceAll("_", "/").replace(/\.md$/i, "").replace(/^https?:\/\//i, "");
      chunks.push({ id: file, url: "https://" + guessedUrl, title: file, text: content });
    }
  } catch { /* vide */ }

  cachedIndex = { chunks };
  return cachedIndex;
}

/* =========================
   Récupération: TF-IDF simple
========================= */

function tokenize(s: string): string[] {
  return s.toLowerCase().normalize("NFKD").replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter(Boolean);
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
  df.forEach((v, k) => idf.set(k, Math.log((1 + N) / (1 + v)) + 1));
  return idf;
}

function scoreChunk(qt: string[], ch: IndexChunk, idf: Map<string, number>): number {
  if (!ch.text) return 0;
  const tokens = tokenize(ch.text);
  if (!tokens.length) return 0;
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  let score = 0;
  for (const q of qt) score += (tf.get(q) ?? 0) * (idf.get(q) ?? 0);
  if (ch.title) {
    const titleSet = new Set(tokenize(ch.title));
    let hits = 0;
    for (const q of qt) if (titleSet.has(q)) hits++;
    score *= 1 + Math.min(0.3, hits * 0.05);
  }
  return score;
}

function topK(context: IndexFile, q: string, k = 6): Retrieved[] {
  const chunks = context.chunks ?? [];
  if (!chunks.length) return [];
  const qTokens = tokenize(q).slice(0, 24);
  const idf = buildIdfMap(chunks);
  return chunks
    .map((ch) => ({ chunk: ch, score: scoreChunk(qTokens, ch, idf) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

/* =========================
   OpenRouter (multi-modèles)
========================= */

const DEFAULT_MODEL_LIST = "qwen/qwen-2.5-72b-instruct:free,google/gemma-2-9b-it:free,mistralai/mistral-nemo:free";
function parseModelList(s: string): string[] { return s.split(",").map((x) => x.trim()).filter(Boolean); }
const MODEL_LIST: string[] = process.env.RAG_MODEL ? parseModelList(process.env.RAG_MODEL) : parseModelList(DEFAULT_MODEL_LIST);
const DEFAULT_MODEL: string = MODEL_LIST[0] ?? "google/gemma-2-9b-it:free";

function systemPrompt(siteName?: string): string {
  const tag = siteName ? ` pour ${siteName}` : "";
  return [
    `Tu es “Sempaï Kinko”, un assistant d’aide et de vente${tag}.`,
    `Réponds clairement et brièvement.`,
    `- Réponds dans la langue demandée (FR par défaut).`,
    `- Pas d’invention : si l’info manque, dis-le et oriente.`,
    `- Ajoute une courte section "Sources" avec 1–3 liens pertinents.`,
  ].join("\n");
}

function buildUserPrompt(q: string, lang: string | undefined, retrieved: Retrieved[]): string {
  const ctx = retrieved.map((r, i) => {
    const head = r.chunk.title ? `${r.chunk.title} — ${r.chunk.url}` : r.chunk.url;
    const body = r.chunk.text.slice(0, 4000);
    return `[#${i + 1}] ${head}\n${body}`;
  }).join("\n\n---\n\n");
  const sources = Array.from(new Set(retrieved.map((r) => r.chunk.url))).slice(0, 3);
  return [
    `Question: ${q}`,
    lang ? `Langue attendue: ${lang}` : `Langue attendue: fr`,
    `Contexte :\n${ctx || "(aucun extrait pertinent trouvé)"}`,
    `Consignes :`,
    `- Réponds directement, sans méta-commentaires.`,
    `- Termine par "Sources" avec ces liens uniquement :`,
    sources.length ? sources.map((u) => `- ${u}`).join("\n") : `- (Aucune source disponible)`,
  ].join("\n");
}

async function callOpenRouterChat(apiKey: string, model: string, system: string, user: string): Promise<string> {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": process.env.OPENROUTER_SITE_URL ?? "https://example.com",
      "X-Title": process.env.OPENROUTER_SITE_NAME ?? "Kinko FAQ AI",
      "Cache-Control": "no-store",
    },
    body: JSON.stringify({ model, messages: [{ role: "system", content: system }, { role: "user", content: user }], temperature: 0.3 }),
  });
  if (!res.ok) throw new Error(`OpenRouter error ${res.status}: ${await res.text().catch(() => "")}`);
  const json = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
  const content = json.choices?.[0]?.message?.content?.trim();
  if (!content) throw new Error("Réponse vide du modèle.");
  return content;
}

async function answerWithFallback(apiKey: string, system: string, user: string): Promise<string> {
  const tried: string[] = [];
  for (const m of MODEL_LIST) {
    try { return await callOpenRouterChat(apiKey, m, system, user); }
    catch { tried.push(m); }
  }
  if (!tried.includes(DEFAULT_MODEL)) return callOpenRouterChat(apiKey, DEFAULT_MODEL, system, user);
  throw new Error(`Tous les modèles ont échoué: ${tried.join(", ")}`);
}

/* =========================
   TEMPS RÉEL — Metaobjects
========================= */

function shopHostFromBase(): string | null {
  const base = publicBase();
  if (!base) return null;
  try { return new URL(base).hostname; } catch { return null; }
}

function parseEvent(fields: ShopifyField[], typeHint?: string): EventItem {
  const map: Record<string, string> = {};
  for (const f of fields) map[f.key.toLowerCase()] = f.value;

  const title = map["title"] || map["name"] || map["nom"] || map["event"] || map["evenement"] || map["competition"];
  const start = map["start"] || map["date"] || map["start_date"] || map["date_debut"] || map["debut"] || map["from"];
  const end   = map["end"]   || map["end_date"]   || map["date_fin"]   || map["fin"]   || map["to"];
  const location  = map["location"] || map["lieu"] || map["city"] || map["ville"];
  const url   = map["url"] || map["register_url"] || map["inscription"] || map["lien"];
  const organizer = map["organizer"] || map["organisateur"] || map["federation"] || map["club"] || map["brand"] || typeHint;
  return { title, start, end, location, url, organizer };
}

function formatDateFR(iso?: string): string | undefined {
  if (!iso) return;
  const d = new Date(iso);
  if (Number.isNaN(+d)) return;
  return d.toLocaleDateString("fr-CA", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

function buildNextEventHTML(ev: EventItem): string {
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

function detectOrganizerToken(q: string): string | null {
  const s = q.toLowerCase();
  const tokens = ["sunfuki", "wkc", "naska", "jga", "kenpo", "studios unis"];
  for (const t of tokens) if (s.includes(t)) return t;
  return null;
}

function looksLikeNextEventQuestion(q: string): boolean {
  const s = q.toLowerCase();
  return /(prochain(e)?|date).*(comp(é|e)tition|tournoi|év(é|e)nement)/i.test(s);
}

/* ----- Admin GraphQL ----- */
async function adminFetch<T>(query: string, variables: Record<string, unknown> = {}): Promise<T | null> {
  const host = shopHostFromBase();
  const token = (process.env.SHOPIFY_ADMIN_TOKEN || "").trim();
  if (!host || !token) return null;
  const apiVersion = "2024-10";
  const resp = await fetch(`https://${host}/admin/api/${apiVersion}/graphql.json`, {
    method: "POST",
    headers: { "X-Shopify-Access-Token": token, "Content-Type": "application/json", "Cache-Control": "no-store" },
    body: JSON.stringify({ query, variables }),
  });
  if (!resp.ok) {
    console.error("[AdminGraphQL] HTTP", resp.status, await resp.text().catch(() => ""));
    return null;
  }
  return (await resp.json().catch(() => null)) as T | null;
}

/* ----- Storefront GraphQL ----- */
async function storefrontFetch<T>(query: string, variables: Record<string, unknown> = {}): Promise<T | null> {
  const host = shopHostFromBase();
  const token = (process.env.SHOPIFY_STOREFRONT_TOKEN || "").trim();
  if (!host || !token) return null;
  const apiVersion = "2024-10";
  const resp = await fetch(`https://${host}/api/${apiVersion}/graphql.json`, {
    method: "POST",
    headers: { "X-Shopify-Storefront-Access-Token": token, "Content-Type": "application/json", "Cache-Control": "no-store" },
    body: JSON.stringify({ query, variables }),
  });
  if (!resp.ok) {
    console.error("[StorefrontGraphQL] HTTP", resp.status, await resp.text().catch(() => ""));
    return null;
  }
  return (await resp.json().catch(() => null)) as T | null;
}

/* ----- Cherche la prochaine compétition (Admin puis Storefront) ----- */
async function fetchNextEventSmart(organizerHint: string | null): Promise<EventItem | null> {
  // 1) ADMIN: liste des types puis lecture de chaque type
  const defQ = `
    query { metaobjectDefinitions(first: 50) { edges { node { type } } } }
  `;
  const defs = await adminFetch<ShopifyMetaobjectDefinitionsResp>(defQ);
  const types: string[] =
    defs?.data?.metaobjectDefinitions?.edges?.map((e) => e.node?.type).filter((t): t is string => !!t) ?? [];

  const uniqueTypes = Array.from(new Set([...types]));
  const now = new Date();
  const pool: Array<{ ev: EventItem; t: number }> = [];

  const moQ = `
    query ReadMetaobjects($type: String!, $first: Int!) {
      metaobjects(type: $type, first: $first) {
        edges { node { type fields { key value } } }
      }
    }
  `;

  if (uniqueTypes.length) {
    for (const t of uniqueTypes) {
      const data = await adminFetch<ShopifyMetaobjectsResp>(moQ, { type: t, first: 100 });
      const edges = data?.data?.metaobjects?.edges ?? [];
      for (const e of edges) {
        const node = e?.node;
        if (!node?.fields) continue;
        const parsed = parseEvent(node.fields, node.type);
        if (!parsed.start) continue;
        if (organizerHint && parsed.organizer && !parsed.organizer.toLowerCase().includes(organizerHint)) continue;
        const d = new Date(parsed.start);
        if (Number.isNaN(+d)) continue;
        if (d >= now) pool.push({ ev: parsed, t: +d });
      }
    }
  }

  // 2) Si Admin vide, tente Storefront sur quelques types probables
  if (!pool.length) {
    const probable = ["event", "competition", "evenement", "calendar"];
    const sfQ = `
      query ReadSf($type: String!, $first: Int!) {
        metaobjects(type: $type, first: $first) {
          edges { node { type: type fields { key value } } }
        }
      }
    `;
    for (const t of probable) {
      const data = await storefrontFetch<ShopifyMetaobjectsResp>(sfQ, { type: t, first: 100 });
      const edges = data?.data?.metaobjects?.edges ?? [];
      for (const e of edges) {
        const node = e?.node;
        if (!node?.fields) continue;
        const parsed = parseEvent(node.fields, node.type);
        if (!parsed.start) continue;
        if (organizerHint && parsed.organizer && !parsed.organizer.toLowerCase().includes(organizerHint)) continue;
        const d = new Date(parsed.start);
        if (Number.isNaN(+d)) continue;
        if (d >= now) pool.push({ ev: parsed, t: +d });
      }
    }
  }

  if (!pool.length) return null;
  pool.sort((a, b) => a.t - b.t);
  return pool[0].ev;
}

/* =========================
   Handler POST
========================= */

export async function POST(req: NextRequest) {
  const headers = corsHeaders(req);

  try {
    const { q, lang } = (await req.json()) as FaqAiRequest;
    if (!q || typeof q !== "string") {
      return new NextResponse(JSON.stringify({ error: "Paramètre 'q' manquant." }), { status: 400, headers });
    }

    // TEMPS RÉEL: prochaine compétition
    if (looksLikeNextEventQuestion(q)) {
      const org = detectOrganizerToken(q);
      const ev = await fetchNextEventSmart(org);
      if (ev) {
        const html = buildNextEventHTML(ev);
        return new NextResponse(JSON.stringify({ answer: html }), { status: 200, headers });
      }
    }

    // RAG
    const index = await loadIndex();
    const retrieved = topK(index, q, 6);
    const sys = systemPrompt(process.env.RAG_SITE_NAME);
    const user = buildUserPrompt(q, lang, retrieved);

    const apiKey = process.env.OPENROUTER_API_KEY ?? "";
    if (!apiKey) {
      return new NextResponse(JSON.stringify({ error: "OPENROUTER_API_KEY manquant." }), { status: 500, headers });
    }

    const raw = await answerWithFallback(apiKey, sys, user);

    // Liens cliquables + remap du domaine
    const base = publicBase();
    const sanitized = raw
      .replace(/https?:\/\/www\./gi, "https://")
      .replace(/https?:\/\/[^)\s]+myshopify\.com/gi, (m) => (base ? base : m.replace(/\/+$/, "")));
    const htmlOut = autoLinkMarkdown(sanitized);

    return new NextResponse(JSON.stringify({ answer: htmlOut }), { status: 200, headers });
  } catch (e) {
    console.error("[faq-ai] error:", e);
    const message = e instanceof Error ? e.message : "Unknown error";
    return new NextResponse(JSON.stringify({ error: message }), { status: 500, headers });
  }
}
