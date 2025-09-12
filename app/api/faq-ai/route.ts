// app/api/faq-ai/route.ts
import type { NextRequest } from "next/server";

export const runtime = "edge"; // ou "nodejs", comme tu veux

const SENSEI = "Sempaï Kinko";

function systemPrompt(shopName: string) {
  return [
    `Tu es ${SENSEI}, assistant de la boutique ${shopName}.`,
    `Réponds en français par défaut. Sois clair, utile, concis.`,
    `Si tu cites une info du site ou d'une FAQ, mentionne la source sous forme courte (ex: "Source : FAQ - Inscriptions").`,
    `Si le contexte ne suffit pas, demande une précision ou propose un lien utile (catalogue, FAQ, contact).`,
    `N’affiche pas de métadonnées techniques (lang, fetched_at, source_url brut, etc.).`,
  ].join(" ");
}

type RagContext = {
  chunks: { text: string; source?: string }[];
};

async function askOpenRouter(prompt: string) {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error("OPENROUTER_API_KEY manquante");

  const model =
    process.env.OPENROUTER_MODEL ??
    "meta-llama/llama-3.1-8b-instruct:free";

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${key}`,
      "Content-Type": "application/json",
      // Ces 2 en-têtes sont utiles côté navigateur; côté API c’est optionnel
      "HTTP-Referer": process.env.APP_URL || "https://kinko-ia.vercel.app",
      "X-Title": "Kinko IA",
    },
    body: JSON.stringify({
      model,
      temperature: 0.3,
      messages: [
        { role: "system", content: systemPrompt("Kinko") },
        { role: "user", content: prompt },
      ],
    }),
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`OpenRouter ${res.status}: ${t}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() || "";
}

// Petit helper pour fabriquer un prompt RAG propre
function buildRagPrompt(question: string, ctx?: RagContext) {
  const contextBlock = (ctx?.chunks ?? [])
    .slice(0, 6) // limite de sécurité
    .map((c, i) => `#${i + 1} ${c.source ? `[${c.source}] ` : ""}${c.text}`)
    .join("\n\n");

  return [
    `Question utilisateur : ${question}`,
    contextBlock ? `\n\nContexte fiable :\n${contextBlock}` : "",
    `\n\nConsignes : Si le contexte ne contient pas la réponse, explique-le brièvement, puis oriente vers la FAQ ou le contact.`,
  ].join("");
}

function allowCors(origin: string | null) {
  const allow = (process.env.CORS_ALLOW_ORIGINS || "")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
  return origin && allow.includes(origin) ? origin : "";
}

export async function OPTIONS(req: NextRequest) {
  const origin = req.headers.get("origin");
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": allowCors(origin),
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "86400",
    },
  });
}

export async function POST(req: NextRequest) {
  try {
    const origin = req.headers.get("origin");
    const { q, context } = (await req.json()) as { q: string; context?: RagContext };

    if (!q || typeof q !== "string") {
      return new Response(JSON.stringify({ error: "Question manquante." }), {
        status: 400,
        headers: {
          "Access-Control-Allow-Origin": allowCors(origin),
          "Content-Type": "application/json",
        },
      });
    }

    const prompt = buildRagPrompt(q, context);
    const answer = await askOpenRouter(prompt);

    return new Response(JSON.stringify({ answer, who: "Sempaï Kinko" }), {
      status: 200,
      headers: {
        "Access-Control-Allow-Origin": allowCors(origin),
        "Content-Type": "application/json",
      },
    });
  } catch (err: any) {
    return new Response(
      JSON.stringify({ error: "LLM indisponible, réessaie dans un instant." }),
      {
        status: 500,
        headers: {
          "Access-Control-Allow-Origin": allowCors(req.headers.get("origin")),
          "Content-Type": "application/json",
        },
      }
    );
  }
}
