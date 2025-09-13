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

type NextEvent = {
  title: string;
  start: string; // ISO
  end?: string | null;
  location?: string | null;
  url?: string | null;
  organizer?: string | null;
  source: "admin" | "storefront" | "env";
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
   Helpers URLs / ENV
========================= */

function baseUrl(): string {
  // ex: https://qfxdmn-i3.myshopify.com (sans /final)
  const b = (process.env.SHOPIFY_PUBLIC_BASE ?? "").replace(/\/+$/, "");
  return b.replace(/^https?:\/\/www\./, "https://");
}

function shopDomain(): string | null {
  try {
    const u = new URL(baseUrl());
    return u.host; // qfxdmn-i3.myshopify.com
  } catch {
    return null;
  }
}

function sanitizeUrl(u?: string | null): string | undefined {
  if (!u) return undefined;
  const b = baseUrl();
  return u
    .replace(/^https?:\/\/www\./, "https://")
    .replace(/^https?:\/\/qfxdmn-i3\.myshopify\.com/i, b);
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
   Récupération : TF-IDF light
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

  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);

  let score = 0;
  for (const qt of queryTokens) {
    const f = tf.get(qt) ?? 0;
    const w = idf.get(qt) ?? 0;
    score += f * w;
  }

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
  return chunks
    .map((ch) => ({ chunk: ch, score: scoreChunk(qTokens, ch, idf) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
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
   Shopify GraphQL helpers
========================= */

async function shopifyGraphqlAdmin<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
  const domain = shopDomain();
  const token = process.env.SHOPIFY_ADMIN_TOKEN;
  if (!domain || !token) throw new Error("SHOPIFY_ADMIN_TOKEN ou SHOPIFY_PUBLIC_BASE manquant.");

  const res = await fetch(`https://${domain}/admin/api/2024-10/graphql.json`, {
    method: "POST",
    headers: {
      "X-Shopify-Access-Token": token,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, variables }),
  });

  const json = (await res.json()) as T;
  return json;
}

async function shopifyGraphqlStorefront<T>(
  query: string,
  variables?: Record<string, unknown>
): Promise<T> {
  const domain = shopDomain();
  const token = process.env.SHOPIFY_STOREFRONT_TOKEN;
  if (!domain || !token) throw new Error("SHOPIFY_STOREFRONT_TOKEN ou SHOPIFY_PUBLIC_BASE manquant.");

  const res = await fetch(`https://${domain}/api/2024-10/graphql.json`, {
    method: "POST",
    headers: {
      "X-Shopify-Storefront-Access-Token": token,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, variables }),
  });

  const json = (await res.json()) as T;
  return json;
}

/* =========================
   Prochaine compétition (Admin / Storefront / ENV)
========================= */

const EVENT_TYPES = ["event", "evenement", "competition"];

function readNextEventFromEnv(): NextEvent | null {
  try {
    const raw = process.env.NEXT_EVENT_JSON;
    if (!raw) return null;
    const j = JSON.parse(raw) as Record<string, unknown>;
    const startRaw = String(j.start ?? "");
    if (!j.title || !startRaw) return null;
    const start = new Date(startRaw);
    if (isNaN(start.getTime())) return null;
    return {
      title: String(j.title),
      start: start.toISOString(),
      end: j.end ? new Date(String(j.end)).toISOString() : null,
      location: j.location ? String(j.location) : null,
      url: sanitizeUrl((j.url as string) || `${baseUrl()}/pages/calendrier`),
      organizer: j.organizer ? String(j.organizer) : null,
      source: "env",
    };
  } catch {
    return null;
  }
}

function toEventFromFields(fields: Array<{ key: string; value: string }>, source: NextEvent["source"]): NextEvent | null {
  const m = new Map(fields.map((f) => [f.key.toLowerCase(), f.value]));
  const title = m.get("title") ?? m.get("nom") ?? m.get("name");
  const start = m.get("start") ?? m.get("date") ?? m.get("debut");
  if (!title || !start) return null;
  return {
    title,
    start: new Date(start).toISOString(),
    end: m.get("end") ? new Date(String(m.get("end"))).toISOString() : null,
    location: m.get("location") ?? m.get("lieu") ?? null,
    url: sanitizeUrl(m.get("url") ?? undefined) ?? `${baseUrl()}/pages/calendrier`,
    organizer: m.get("organizer") ?? m.get("organisateur") ?? null,
    source,
  };
}

async function fetchNextEventAdmin(): Promise<NextEvent | null> {
  type Node = { type: string; fields: Array<{ key: string; value: string }> };
  type AdminResp = { data?: { metaobjects?: { nodes: Node[] } } };

  for (const t of EVENT_TYPES) {
    const q = `
      query($t: String!) {
        metaobjects(type: $t, first: 50) {
          nodes { type fields { key value } }
        }
      }`;
    const r = await shopifyGraphqlAdmin<AdminResp>(q, { t }).catch(() => null);
    const nodes = r?.data?.metaobjects?.nodes ?? [];
    const events = nodes
      .map((n) => toEventFromFields(n.fields, "admin"))
      .filter((e): e is NextEvent => !!e);
    const now = Date.now();
    const upcoming = events.filter((e) => new Date(e.start).getTime() >= now);
    if (upcoming.length) {
      upcoming.sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime());
      return upcoming[0];
    }
  }
  return null;
}

async function fetchNextEventStorefront(): Promise<NextEvent | null> {
  type Edge = { node: { type: string; fields: Array<{ key: string; value: string }> } };
  type SFResp = { data?: { metaobjects?: { edges: Edge[] } } };

  for (const t of EVENT_TYPES) {
    const q = `
      {
        metaobjects(type: "${t}", first: 50) {
          edges { node { type fields { key value } } }
        }
      }`;
    const r = await shopifyGraphqlStorefront<SFResp>(q).catch(() => null);
    const edges = r?.data?.metaobjects?.edges ?? [];
    const events = edges
      .map((e) => toEventFromFields(e.node.fields, "storefront"))
      .filter((e): e is NextEvent => !!e);
    const now = Date.now();
    const upcoming = events.filter((e) => new Date(e.start).getTime() >= now);
    if (upcoming.length) {
      upcoming.sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime());
      return upcoming[0];
    }
  }
  return null;
}

async function fetchNextEventSmart(): Promise<NextEvent | null> {
  // 1) Essai Admin
  try {
    const admin = await fetchNextEventAdmin();
    if (admin) return admin;
  } catch {
    // ignore
  }
  // 2) Essai Storefront
  try {
    const sf = await fetchNextEventStorefront();
    if (sf) return sf;
  } catch {
    // ignore
  }
  // 3) Fallback ENV
  const env = readNextEventFromEnv();
  if (env) return env;

  return null;
}

function htmlAnswerForEvent(ev: NextEvent): string {
  const d = new Date(ev.start);
  const fmt = d.toLocaleDateString("fr-CA", { year: "numeric", month: "long", day: "numeric" });
  const pieces: string[] = [];
  pieces.push(`La prochaine compétition est <b>${ev.title}</b>.`);
  pieces.push(`📅 <b>Date</b> : ${fmt}`);
  if (ev.location) pieces.push(`📍 <b>Lieu</b> : ${ev.location}`);
  if (ev.organizer) pieces.push(`🏷️ <b>Organisateur</b> : ${ev.organizer}`);
  const link = ev.url ?? `${baseUrl()}/pages/calendrier`;
  const src = `${baseUrl()}/pages/calendrier`;

  return [
    pieces.join("<br/>"),
    "",
    `👉 <a href="${link}" target="_blank" rel="noopener">Détails / Inscription</a>`,
    "",
    `<div style="margin-top:8px"><b>Sources :</b><br/>- <a href="${src}" target="_blank" rel="noopener">Calendrier des compétitions et des événements</a></div>`,
  ].join("\n");
}

/* =========================
   Handler POST
========================= */

const NEXT_EVENT_RE = /(prochain|prochaine).*(comp[eé]tition|tournoi|[eé]v[ée]nement)|quand.*(comp[eé]tition|tournoi|[eé]v[ée]nement)/i;

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

    // 0) Question "prochaine compétition" → réponse déterministe en priorité
    if (NEXT_EVENT_RE.test(q.toLowerCase())) {
      const ev = await fetchNextEventSmart();
      if (ev) {
        const answer = htmlAnswerForEvent(ev);
        return new NextResponse(JSON.stringify({ answer }), { status: 200, headers });
      }
      // sinon, on continue vers le RAG + LLM
    }

    // 1) RAG classique
    const index = await loadIndex();
    const retrieved = topK(index, q, 6);

    // 2) Prompt & LLM
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
