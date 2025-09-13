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
   Utils
========================= */

function normalizeUrl(u: string): string {
  return u.replace(/^https?:\/\/(www\.)?/i, "https://");
}

function fmtDateISO(d: string | Date): string {
  const dt = typeof d === "string" ? new Date(d) : d;
  if (Number.isNaN(dt.getTime())) return "";
  return dt.toLocaleDateString("fr-FR", { day: "2-digit", month: "long", year: "numeric" });
}

function isEventQuestion(q: string): boolean {
  const s = q.toLowerCase();
  return /(prochain|prochaine|date|quand).*(comp(é|e)tition|tournoi|év(é|e)nement|event)/i.test(s) ||
         /(comp(é|e)tition|tournoi|év(é|e)nement|event).*(prochain|prochaine|date|quand)/i.test(s);
}

/* =========================
   Lecture de l'index
========================= */

let cachedIndex: IndexFile | null = null;

async function loadIndex(): Promise<IndexFile> {
  if (cachedIndex) return cachedIndex;

  const root = process.cwd();
  const indexPath = path.join(root, "ingested", "index.json");
  const ingestedDir = path.join(root, "ingested");

  // 1) index.json
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
        const url = normalizeUrl(String(obj.url ?? ""));
        return {
          id: String(obj.id ?? i.toString()),
          url,
          title: obj.title ? String(obj.title) : undefined,
          text: String(obj.text ?? (obj as Record<string, unknown>).content ?? ""),
          tokens: typeof obj.tokens === "number" ? obj.tokens : undefined,
        } satisfies IndexChunk;
      });
      cachedIndex = { chunks };
      return cachedIndex;
    }
  } catch {
    /* pass */
  }

  // 2) fallback .md
  const chunks: IndexChunk[] = [];
  try {
    const files = await fs.readdir(ingestedDir);
    for (const file of files) {
      if (!file.toLowerCase().endsWith(".md")) continue;
      const full = path.join(ingestedDir, file);
      const content = await fs.readFile(full, "utf8");
      const guessedUrl = normalizeUrl(
        "https://" +
          file.replaceAll("_", "/").replace(/\.md$/i, "").replace(/^https?:\/\//i, "")
      );
      chunks.push({ id: file, url: guessedUrl, title: file, text: content });
    }
  } catch {
    /* pass */
  }

  cachedIndex = { chunks };
  return cachedIndex;
}

/* =========================
   Retrieval (TF-IDF simple)
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
    const seen = new Set(tokenize(ch.text));
    for (const t of seen) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const idf = new Map<string, number>();
  df.forEach((v, k) => idf.set(k, Math.log((1 + N) / (1 + v)) + 1));
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
   Chunk "prochaine compétition" (temps réel via env)
========================= */

function nextEventChunkFromEnv(): IndexChunk | null {
  const v = process.env.NEXT_EVENT_JSON;
  if (!v) return null;
  try {
    const parsed = JSON.parse(v) as {
      title?: string;
      start?: string;
      end?: string;
      location?: string;
      url?: string;
      organizer?: string;
    };

    const title = parsed.title ?? "Prochaine compétition";
    const dateStr = parsed.start ? fmtDateISO(parsed.start) : "";
    const endStr = parsed.end ? fmtDateISO(parsed.end) : "";
    const where = parsed.location ? ` — Lieu : ${parsed.location}` : "";
    const org = parsed.organizer ? ` — Organisateur : ${parsed.organizer}` : "";
    const url = parsed.url ? normalizeUrl(parsed.url) : "";

    const when =
      dateStr && endStr && dateStr !== endStr
        ? `${dateStr} → ${endStr}`
        : dateStr || endStr || "(date à confirmer)";

    const textLines = [
      `Prochaine compétition : ${title}`,
      `Date : ${when}${where}${org}`,
      url ? `Lien d'information / inscription : ${url}` : ``,
    ].filter(Boolean);

    return {
      id: "next_event",
      url: url || "https://qfxdmn-i3.myshopify.com/pages/calendrier",
      title: "Prochaine compétition",
      text: textLines.join("\n"),
    };
  } catch {
    return null;
  }
}

/* =========================
   Modèles OpenRouter (fallback)
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
   Prompt & OpenRouter
========================= */

function systemPrompt(siteName?: string): string {
  const tag = siteName ? ` pour ${siteName}` : "";
  return [
    `Tu es “Sempaï Kinko”, un assistant d’aide et de vente${tag}.`,
    `Parle clairement et utilement.`,
    `Si l’info n’est pas dans le contexte, dis-le et propose une alternative concrète (page, contact, etc.).`,
    `Quand tu cites des pages, mets des liens HTML cliquables (<a href="...">texte</a>).`,
  ].join("\n");
}

function buildUserPrompt(q: string, lang: string | undefined, retrieved: Retrieved[]): string {
  const ctx = retrieved
    .map((r, i) => {
      const head = r.chunk.title ? `${r.chunk.title} — ${r.chunk.url}` : r.chunk.url;
      const body = r.chunk.text.slice(0, 3500);
      return `[#${i + 1}] ${head}\n${body}`;
    })
    .join("\n\n---\n\n");

  const dedup = Array.from(new Set(retrieved.map((r) => r.chunk.url))).slice(0, 3);
  const sourcesHtml = dedup
    .map((u) => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`)
    .join("\n- ");

  return [
    `Question: ${q}`,
    `Langue attendue: ${lang || "fr"}`,
    ``,
    `Contexte (extraits) :`,
    ctx || "(aucun extrait pertinent trouvé)",
    ``,
    `Consignes de réponse :`,
    `- Réponds directement, sans méta-commentaires.`,
    `- Termine par un bloc "Sources :" listant 1–3 liens pertinents (HTML <a>) tirés du contexte.`,
    `Sources (pré-sélection) :`,
    sourcesHtml ? `- ${sourcesHtml}` : `- (Aucune source disponible)`,
  ].join("\n");
}

type ORChoice = { message: { role: "assistant"; content: string } };
type ORResp = { choices?: ORChoice[] };

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

  const json = (await res.json()) as ORResp;
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
   POST
========================= */

export async function POST(req: NextRequest) {
  const headers = corsHeaders(req);

  try {
    const body = (await req.json()) as unknown;
    const { q, lang } = (body as FaqAiRequest) ?? {};
    if (!q || typeof q !== "string") {
      return new NextResponse(JSON.stringify({ error: "Paramètre 'q' manquant." }), {
        status: 400,
        headers,
      });
    }

    const index = await loadIndex();
    let retrieved = topK(index, q, 6);

    // 🔸 Injection temps réel "prochaine compétition"
    if (isEventQuestion(q)) {
      const nextEvt = nextEventChunkFromEnv();
      if (nextEvt) {
        retrieved = [{ chunk: nextEvt, score: 999 }, ...retrieved];
      }
    }

    const sys = systemPrompt(process.env.RAG_SITE_NAME);
    const user = buildUserPrompt(q, lang, retrieved);

    const apiKey = process.env.OPENROUTER_API_KEY ?? "";
    if (!apiKey) {
      return new NextResponse(
        JSON.stringify({ error: "OPENROUTER_API_KEY manquant (Vercel → Settings → Environment Variables)." }),
        { status: 500, headers }
      );
    }

    const { answer } = await answerWithFallback(apiKey, sys, user);
    return new NextResponse(JSON.stringify({ answer }), { status: 200, headers });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "Unknown error";
    return new NextResponse(JSON.stringify({ error: message }), { status: 500, headers });
  }
}
