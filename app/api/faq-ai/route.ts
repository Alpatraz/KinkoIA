// app/api/faq-ai/route.ts
import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs"; // nécessaire pour utiliser fs

type FaqRequest = { q: string; lang?: string };
type Source = { file: string };
type FaqResponse = { answer: string; sources?: Source[] };

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const DATA_DIR = path.join(process.cwd(), "ingested");

let INDEX: { file: string; text: string }[] | null = null;

function loadIndex(): void {
  if (INDEX) return;
  INDEX = [];
  if (!fs.existsSync(DATA_DIR)) return;
  const files = fs.readdirSync(DATA_DIR).filter((f) => /\.(md|txt|html)$/i.test(f));
  for (const file of files) {
    const text = fs.readFileSync(path.join(DATA_DIR, file), "utf8");
    INDEX.push({ file, text });
  }
}

function bestSnippet(q: string): { snippet: string; file: string } | null {
  loadIndex();
  if (!INDEX || INDEX.length === 0) return null;

  const query = q.toLowerCase().replace(/\s+/g, " ").trim();
  const terms = query.split(" ").filter(Boolean);

  let bestScore = 0;
  let best: { text: string; file: string } | null = null;

  for (const doc of INDEX) {
    const lower = doc.text.toLowerCase();
    let score = 0;
    for (const t of terms) {
      const m = lower.match(new RegExp(`\\b${escapeRegExp(t)}\\b`, "g"));
      score += m ? m.length : 0;
    }
    if (score > bestScore) {
      bestScore = score;
      best = { text: doc.text, file: doc.file };
    }
  }

  if (!best || bestScore === 0) return null;

  // extrait autour de la première occurrence
  const lower = best.text.toLowerCase();
  let i = Infinity;
  for (const t of terms) {
    const p = lower.indexOf(t);
    if (p !== -1 && p < i) i = p;
  }
  if (i === Infinity) i = 0;
  const start = Math.max(0, i - 200);
  const end = Math.min(best.text.length, i + 400);
  const raw = best.text.slice(start, end).replace(/\s+/g, " ").trim();

  const snippet = (start > 0 ? "… " : "") + raw + (end < best.text.length ? " …" : "");
  return { snippet, file: best.file };
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: corsHeaders });
}

export async function POST(req: Request) {
  try {
    const { q } = (await req.json()) as FaqRequest;
    if (!q || typeof q !== "string") {
      return NextResponse.json<FaqResponse>(
        { answer: "Question vide. Peux-tu préciser ?" },
        { headers: corsHeaders, status: 400 }
      );
    }

    const hit = bestSnippet(q);
    const answer =
      hit?.snippet ??
      "Je n’ai pas trouvé d’information correspondante dans la documentation. Essaie de reformuler ta question ou visite la page Espace Kinko.";

    const payload: FaqResponse = hit
      ? { answer, sources: [{ file: hit.file }] }
      : { answer };

    return NextResponse.json<FaqResponse>(payload, { headers: corsHeaders });
  } catch (err) {
    return NextResponse.json<FaqResponse>(
      { answer: "Oups, une erreur est survenue. Réessaie dans un instant." },
      { headers: corsHeaders, status: 500 }
    );
  }
}
