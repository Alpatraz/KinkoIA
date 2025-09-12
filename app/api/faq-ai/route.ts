// app/api/faq-ai/route.ts
export const runtime = "nodejs";           // Embeddings locaux => runtime Node
export const dynamic = "force-dynamic";

/**
 * Variables attendues (.env.local / Vercel):
 *
 *  OPENROUTER_API_KEY=sk-or-v1_...
 *  OPENROUTER_SITE_URL=https://kinko-karate.com
 *  OPENROUTER_APP_NAME=Kinko FAQ Bot
 *
 *  SHOPIFY_STORE_DOMAIN=xxx.myshopify.com
 *  SHOPIFY_STOREFRONT_TOKEN=...
 *
 *  SUPABASE_URL=...
 *  SUPABASE_SERVICE_ROLE=...
 *
 * Supabase: table `faq_chunks` en vector(384) + fonction match_faq_chunks(vector(384), int)
 */

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

export async function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const qRaw = (body?.q ?? "").toString();
    const q = qRaw.trim();
    if (!q) return json({ error: "Missing q" }, 400);

    // 1) Essaie d'abord la FAQ interne (métaobjets Shopify)
    const faqs = await fetchFaqItems();
    const scored = faqs
      .map((f) => ({ ...f, s: sim(q, `${f.question || ""} ${f.answerPlain || ""}`) }))
      .sort((a, b) => b.s - a.s);

    if (scored[0]?.s > 0.78) {
      const top = scored[0];
      return json({
        answer: bulletify(top.answerPlain),
        sources: [{ id: 1, url: `/pages/faq#${top.handle}`, label: `FAQ : ${top.question}` }],
        confidence: "high",
        mode: "faq",
      });
    }

    // 2) Fallback RAG (Supabase + embeddings locaux MiniLM 384d)
    const chunks = await ragSearch(q, 6);

    if (!chunks?.length) {
      const noData = `Je n’ai pas trouvé d’élément pertinent dans la base pour ta question.

Sources à consulter :
- Calendrier des compétitions (notre site)
- Page de l’organisateur (Uventex/Fitofan) → Règlement / Informations
- Fédérations (WKC/WAKO) : sections “Rules” / “Equipment”

Si tu me précises l’événement/organisateur, je peux cibler les bonnes sources.`;
      return json({ answer: noData, sources: [], confidence: "low", mode: "rag" });
    }

    const context = chunks
      .map((c: any, i: number) => `[${i + 1}] ${c.content}\nURL: ${c.url}`)
      .join("\n\n");

    const prompt = `Tu es l’assistant FAQ pour karatékas.
- Réponds UNIQUEMENT à partir des contextes ci-dessous; si une info manque, dis-le et propose où chercher (fiche Calendrier/organisateur/contact).
- Si une règle varie selon l’organisateur (WKC/WAKO/organisateur local), dis-le explicitement.
- Réponse concise, en FR sauf si la question est en anglais.
- Termine par "Sources : [#,#]".

Question: ${q}

Contextes:
${context}`;

    const completion = await chat(prompt);

    const confident =
      /Sources\s*:\s*\[[0-9 ,]+\]/i.test(completion) &&
      (chunks?.[0]?.score ?? 0) > 0.72;

    return json({
      answer: (completion || "").trim(),
      sources: (chunks || []).slice(0, 3).map((c: any, i: number) => ({
        id: i + 1,
        url: c.url,
        label: c.source || c.url,
        score: c.score,
      })),
      confidence: confident ? "high" : "low",
      mode: "rag",
    });
  } catch (err: any) {
    return json({ error: err?.message || "Unexpected error" }, 500);
  }
}

/* ----------------- utils ----------------- */

function json(obj: any, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

const SHOPIFY_DOMAIN = process.env.SHOPIFY_STORE_DOMAIN!;
const STOREFRONT_TOKEN = process.env.SHOPIFY_STOREFRONT_TOKEN!;
async function fetchFaqItems() {
  if (!SHOPIFY_DOMAIN || !STOREFRONT_TOKEN) return [];
  const query = `
    query {
      metaobjects(type: "faq_item", first: 200) {
        nodes { handle fields { key value } }
      }
    }`;
  const r = await fetch(`https://${SHOPIFY_DOMAIN}/api/2024-07/graphql.json`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
    },
    body: JSON.stringify({ query }),
  });
  const j = await r.json();
  const nodes = j?.data?.metaobjects?.nodes || [];
  return nodes.map((n: any) => {
    const get = (k: string) => n.fields?.find((f: any) => f.key === k)?.value;
    const answerPlain = richTextToPlain(get("answer"));
    return { handle: n.handle, question: get("question") || "", answerPlain };
  });
}

function richTextToPlain(rt?: string) {
  if (!rt) return "";
  try {
    const obj = JSON.parse(rt);
    return (obj.children || [])
      .map((p: any) => (p.children || []).map((t: any) => t?.value || "").join(""))
      .join("\n\n");
  } catch {
    return rt;
  }
}

function sim(a: string, b: string) {
  const norm = (s: string) =>
    s.toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .split(/\W+/)
      .filter((w) => w.length > 3);
  const A = new Set(norm(a));
  const B = new Set(norm(b));
  const inter = [...A].filter((x) => B.has(x)).length;
  const uni = new Set([...A, ...B]).size || 1;
  return inter / uni;
}

function bulletify(text: string) {
  const lines = (text || "").split(/\n+/).map((s) => s.trim()).filter(Boolean);
  return lines.length > 1 ? lines.map((l) => `- ${l}`).join("\n") : text;
}

/* ---- RAG (Supabase + embeddings locaux MiniLM 384d) ---- */

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE!;

// Embeddings locaux
import { pipeline } from "@xenova/transformers";
let _fe: any;
async function getEmbedder() {
  if (!_fe) {
    // @ts-ignore
    _fe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2"); // 384 dims
  }
  return _fe;
}
async function embedLocal(text: string): Promise<number[]> {
  const fe = await getEmbedder();
  const out = await fe(text, { pooling: "mean", normalize: true });
  return Array.from(out.data); // Float32Array -> number[]
}

async function ragSearch(question: string, top = 6) {
  if (!SUPABASE_URL || !SUPABASE_KEY) return [];
  const qEmb = await embedLocal(question);

  const { createClient } = await import("@supabase/supabase-js");
  const supa = createClient(SUPABASE_URL, SUPABASE_KEY);

  const { data, error } = await supa.rpc("match_faq_chunks", {
    query_embedding: qEmb,
    match_count: top,
  });
  if (error) throw error;
  return data;
}

/* ------------------- Chat via OpenRouter ------------------- */

async function chat(prompt: string) {
  const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
      "HTTP-Referer": process.env.OPENROUTER_SITE_URL || "https://kinko-karate.com",
      "X-Title": process.env.OPENROUTER_APP_NAME || "Kinko FAQ Bot",
    },
    body: JSON.stringify({
      // Tu peux passer à "meta-llama/llama-3.3-70b-instruct:free" si dispo
      model: "meta-llama/llama-3.1-8b-instruct:free",
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content:
            "Assistant FAQ karatékas : n’utiliser que les contextes; citer Sources:[#]; préciser variations par organisateur; FR/EN selon la question; si info absente -> le dire + proposer lien/contact; jamais d’affirmation sans source.",
        },
        { role: "user", content: prompt },
      ],
    }),
  });
  const j = await r.json().catch(async () => ({ choices: [] }));
  return j?.choices?.[0]?.message?.content || "";
}
