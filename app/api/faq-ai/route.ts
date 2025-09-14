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

type EventItem = {
  title: string;
  start: string; // ISO
  end?: string;
  location?: string;
  url?: string;
  organizer?: string;
  tags?: string[];
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

  // 1) Essayer index.json (supporte 'chunks' OU 'entries')
  try {
    const buf = await fs.readFile(indexPath, "utf8");
    const parsed = JSON.parse(buf) as unknown;

    if (parsed && typeof parsed === "object") {
      const p = parsed as { chunks?: unknown; entries?: unknown };
      const fromChunks = Array.isArray(p.chunks) ? (p.chunks as unknown[]) : null;
      const fromEntries = Array.isArray(p.entries) ? (p.entries as unknown[]) : null;
      const arr: unknown[] = fromChunks ?? fromEntries ?? [];

      if (arr.length > 0) {
        const chunks = arr.map((c, i) => {
          const obj = (c ?? {}) as Record<string, unknown>;
          const text =
            typeof obj.text === "string"
              ? obj.text
              : typeof obj.content === "string"
              ? (obj.content as string)
              : typeof obj.body === "string"
              ? (obj.body as string)
              : "";
        return {
            id: String(obj.id ?? i.toString()),
            url: String(obj.url ?? ""),
            title: typeof obj.title === "string" ? (obj.title as string) : undefined,
            text,
            tokens: typeof obj.tokens === "number" ? (obj.tokens as number) : undefined,
          } satisfies IndexChunk;
        });
        cachedIndex = { chunks };
        return cachedIndex;
      }
    }
  } catch {
    // ignore
  }

  // 2) Fallback: lire les .md de /ingested
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
    // index vide si rien
  }

  cachedIndex = { chunks };
  return cachedIndex;
}

/* =========================
   Récupération : scoring TF-IDF-ish
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
   Événements en temps réel
========================= */

function readEventsFromEnv(): EventItem[] {
  const plural = process.env.NEXT_EVENTS_JSON;
  if (plural) {
    try {
      const arr = JSON.parse(plural) as unknown;
      if (Array.isArray(arr)) {
        return arr
          .map((x): EventItem | null => {
            if (!x || typeof x !== "object") return null;
            const o = x as Record<string, unknown>;
            if (typeof o.title !== "string" || typeof o.start !== "string") return null;
            return {
              title: o.title,
              start: o.start,
              end: typeof o.end === "string" ? o.end : undefined,
              location: typeof o.location === "string" ? o.location : undefined,
              url: typeof o.url === "string" ? o.url : undefined,
              organizer: typeof o.organizer === "string" ? o.organizer : undefined,
              tags: Array.isArray(o.tags) ? (o.tags.filter((t) => typeof t === "string") as string[]) : undefined,
            };
          })
          .filter((e): e is EventItem => !!e);
      }
    } catch {
      // ignore
    }
  }
  // fallback: NEXT_EVENT_JSON (singulier)
  const singular = process.env.NEXT_EVENT_JSON;
  if (singular) {
    try {
      const o = JSON.parse(singular) as Record<string, unknown>;
      if (o && typeof o === "object" && typeof o.title === "string" && typeof o.start === "string") {
        return [
          {
            title: String(o.title),
            start: String(o.start),
            end: typeof o.end === "string" ? o.end : undefined,
            location: typeof o.location === "string" ? o.location : undefined,
            url: typeof o.url === "string" ? o.url : undefined,
            organizer: typeof o.organizer === "string" ? o.organizer : undefined,
            tags: Array.isArray(o.tags) ? (o.tags.filter((t) => typeof t === "string") as string[]) : undefined,
          },
        ];
      }
    } catch {
      // ignore
    }
  }
  return [];
}

function detectOrganizer(q: string): string | null {
  const m = q.toLowerCase();
  const table: Record<string, string> = {
    "sunfuki": "sunfuki",
    "studios unis": "studios unis",
    "studiosunis": "studios unis",
    "wkc": "wkc",
    "naska": "naska",
    "wako": "wako",
    "jga kenpo": "jga kenpo",
    "jgakenpo": "jga kenpo",
  };
  for (const k of Object.keys(table)) {
    if (m.includes(k)) return table[k];
  }
  return null;
}

function normalizeUrl(u: string): string {
  try {
    const x = new URL(u);
    x.hostname = x.hostname.replace(/^www\./, "");
    return x.toString();
  } catch {
    return u.replace("://www.", "://");
  }
}

function pickNextEvent(events: EventItem[], org?: string | null): EventItem | null {
  const now = Date.now();
  const list = events.filter((e) => {
    const t = Date.parse(e.start);
    if (Number.isNaN(t) || t < now) return false;
    if (!org) return true;
    const orga = (e.organizer ?? "").toLowerCase();
    const tags = (e.tags ?? []).map((x) => x.toLowerCase());
    return orga.includes(org) || tags.includes(org);
  });
  if (list.length === 0) return null;
  list.sort((a, b) => Date.parse(a.start) - Date.parse(b.start));
  return list[0];
}

function formatEventLine(e: EventItem): string {
  const d = new Date(e.start);
  const date = d.toLocaleDateString("fr-CA", { year: "numeric", month: "long", day: "numeric" });
  const parts = [
    `**${e.title}** — ${date}`,
    e.location ? `à ${e.location}` : null,
    e.organizer ? `(organisateur : ${e.organizer})` : null,
  ].filter(Boolean);
  const head = parts.join(" ");
  const link = e.url ? `\n→ [Calendrier des compétitions](${normalizeUrl(e.url)})` : "";
  return head + link;
}

/* =========================
   Résultats passés (JSON)
========================= */

async function loadResultsChunks(): Promise<IndexChunk[]> {
  const root = process.cwd();
  const dir = path.join(root, "public", "results");
  const out: IndexChunk[] = [];
  try {
    const files = await fs.readdir(dir);
    for (const f of files) {
      if (!f.toLowerCase().endsWith(".json")) continue;
      const full = path.join(dir, f);
      const raw = await fs.readFile(full, "utf8");
      // Résumé "safe" pour retrieval
      let title = f.replace(/\.json$/i, "");
      const url = `${process.env.SHOPIFY_PUBLIC_BASE ?? "https://qfxdmn-i3.myshopify.com"}/results/${encodeURIComponent(f)}`;
      let summary = raw.slice(0, 2000);
      try {
        const parsed = JSON.parse(raw) as Record<string, unknown>;
        const t = parsed["title"];
        const date = parsed["date"] ?? parsed["when"];
        const org = parsed["organizer"];
        const loc = parsed["location"] ?? parsed["city"];
        const head = [
          typeof t === "string" ? t : null,
          typeof date === "string" ? date : null,
          typeof org === "string" ? `(${org})` : null,
          typeof loc === "string" ? `— ${loc}` : null,
        ]
          .filter(Boolean)
          .join(" ");
        if (head) title = head;
        summary = JSON.stringify(parsed).slice(0, 2000);
      } catch {
        // garde le raw slice
      }
      out.push({
        id: `result:${f}`,
        url,
        title,
        text: summary,
      });
    }
  } catch {
    // aucun fichier => ok
  }
  return out;
}

/* =========================
   Prompt & appel OpenRouter
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
    // Identité
    `Tu es “Sempaï Kinko”, assistant officiel de la marque Kinko (karaté & armes d’entraînement)${tag}.`,
    `Rôle : conseiller d’équipement & coach d’orientation (tailles, modèles, niveaux de ceinture, sécurité, entretien) et guide dans la boutique Shopify avec parfois une teinte d'humour.`,
    `Gag récurrent: si l’utilisateur ne t’appelle pas “Sempaï”, tu peux gentiment le taquiner avec une seule phrase (ex: “On dit Sempaï 😉 sinon: 20 push-up !”) — mais tu réponds quand même normalement.`,

    // Style & ton
    `Style : expert, bienveillant, direct, sans jargon inutile. Tutoiement chaleureux et drôle au Québec. FR par défaut; propose EN si besoin. Tu peux ponctuellement faire un trait d’humour (jamais lourd).`,
    `Émojis : au plus un, seulement si utile (🥋, 🥇, 🛠).`,
    `Longueur : 3–5 lignes max + puces quand c’est plus clair.`,

    // Vérité & limites
    `Ne jamais inventer stock, prix, délais ou remises : renvoie vers les données Shopify si non présentes dans le contexte.`,
    `Sécurité d’abord : pas de conseils médicaux ni de techniques dangereuses/illégales.`,
    `Si la demande sort du périmètre (SAV complexe, médical, etc.), propose de passer à un humain et de nous contacter.`,

    // RAG & “Sources”
    `Tu t’appuies PRIORITAIREMENT sur le contexte fourni (extraits du site).`,
    `Si l’info n’y est pas, dis-le simplement (pas d’invention) et propose une alternative concrète (page à visiter, contact).`,
    `Termine par une section "Sources" listant 1–3 liens du contexte (aucun autre lien). S’il n’y en a pas : « (Aucune source disponible) ».`,
  ].join("\n");
}

function buildUserPrompt(
  q: string,
  lang: string | undefined,
  retrieved: Retrieved[],
  realtimeNotes: string[]
): string {
  const ctxRag = retrieved
    .map((r, i) => {
      const head = r.chunk.title ? `${r.chunk.title} — ${r.chunk.url}` : r.chunk.url;
      const body = r.chunk.text.slice(0, 4000);
      return `[#RAG-${i + 1}] ${head}\n${body}`;
    })
    .join("\n\n---\n\n");

  const ctxRealtime = realtimeNotes.length
    ? `[#REALTIME]\n${realtimeNotes.join("\n")}`
    : "";

  return [
    `Question: ${q}`,
    lang ? `Langue attendue: ${lang}` : `Langue attendue: fr`,
    ``,
    `Contexte (extraits provenant du site et données temps réel) :`,
    ctxRealtime,
    ctxRag || "(aucun extrait pertinent trouvé)",
    ``,
    `Consignes de réponse :`,
    `- Réponds directement, sans méta-commentaires.`,
    `- Si l'info n'est pas dans le contexte, dis-le et propose une alternative concrète (page à visiter, contact…).`,
    `- Termine par une section "Sources" (1–3 liens) en Markdown [texte](url), sans afficher l’URL brute.`,
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
   Post-traitement de liens
========================= */

function labelForUrl(u: string): string {
  try {
    const x = new URL(u);
    const p = x.pathname;
    if (p.includes("/pages/calendrier")) return "Calendrier des compétitions";
    if (p.includes("/blogs/")) return "Article du blog";
    if (p.includes("/products/")) return "Voir le produit";
    if (p.includes("/pages/")) return "Voir la page";
    return x.hostname.replace(/^www\./, "");
  } catch {
    return "Lien";
  }
}

function tidyLinks(markdown: string): string {
  // 1) supprime www. dans les href
  let txt = markdown.replace(/\((https?:\/\/)www\./g, "($1");

  // 2) Si le texte du lien est l’URL brute, remplace par un libellé
  //    pattern: [https://domaine/...](https://domaine/...)
  txt = txt.replace(
    /\[https?:\/\/[^\]]+\]\((https?:\/\/[^\)]+)\)/g,
    (_m, href: string) => `[${labelForUrl(href)}](${normalizeUrl(href)})`
  );

  // 3) Sur les liens déjà corrects, normalise juste le href (sans www)
  txt = txt.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g,
    (_m, label: string, href: string) => `[${label}](${normalizeUrl(href)})`
  );

  return txt;
}

/* =========================
   Handler POST
========================= */

export async function POST(req: NextRequest) {
  const headers = corsHeaders(req);

  try {
    const body = (await req.json()) as unknown;
    const { q, lang } = (body ?? {}) as FaqAiRequest;

    // endpoint debug interne
    if (q === "__events__") {
      const events = readEventsFromEnv();
      const next = pickNextEvent(events, null);
      const nextSunfuki = pickNextEvent(events, "sunfuki");
      return new NextResponse(
        JSON.stringify({ count: events.length, next, nextSunfuki }),
        { status: 200, headers }
      );
    }

    if (!q || typeof q !== "string") {
      return new NextResponse(JSON.stringify({ error: "Paramètre 'q' manquant." }), {
        status: 400,
        headers,
      });
    }

    // 1) RAG
    const index = await loadIndex();
    const retrieved = topK(index, q, 6);

    // 2) Données temps réel (événements)
    const realtimeNotes: string[] = [];
    const org = detectOrganizer(q);
    const events = readEventsFromEnv();

    // prochaine globale
    const nextEvent = pickNextEvent(events, null);
    if (nextEvent) {
      realtimeNotes.push(`Prochaine compétition (tous organisateurs) :\n${formatEventLine(nextEvent)}`);
    }

    // prochaine par organisateur si demandé
    if (org) {
      const nextOrg = pickNextEvent(events, org);
      if (nextOrg) {
        realtimeNotes.push(`Prochaine compétition **${org}** :\n${formatEventLine(nextOrg)}`);
      }
    }

    // 3) Résultats passés (JSON) → injectés comme pseudo-chunks
    const pastChunks = await loadResultsChunks();
    const retrievedPlusPast: Retrieved[] = [
      ...retrieved,
      ...pastChunks.map((c, i) => ({ chunk: c, score: 0.5 - i * 0.01 })), // un petit poids fixe
    ];

    // 4) Prompt + appel modèle
    const sys = systemPrompt(process.env.RAG_SITE_NAME);
    const user = buildUserPrompt(q, lang, retrievedPlusPast, realtimeNotes);

    const apiKey = process.env.OPENROUTER_API_KEY ?? "";
    if (!apiKey) {
      return new NextResponse(
        JSON.stringify({
          error:
            "OPENROUTER_API_KEY manquant. Ajoutez-le aux variables d’environnement Vercel.",
        }),
        { status: 500, headers }
      );
    }

    const { answer } = await answerWithFallback(apiKey, sys, user);
    const pretty = tidyLinks(answer);

    return new NextResponse(JSON.stringify({ answer: pretty }), { status: 200, headers });
  } catch (e: unknown) {
    console.error("[faq-ai] error:", e);
    const message = e instanceof Error ? e.message : "Unknown error";
    return new NextResponse(JSON.stringify({ error: message }), { status: 500, headers });
  }
}
