// app/api/faq-ai/route.ts
export const runtime = 'nodejs';

type RAGRequest = {
  q: string;
  lang?: string;
  topK?: number;
};

type Source = { title: string; url: string; snippet?: string };
type RAGResponse = { answer: string; sources: Source[] };
type ErrorPayload = { error: string };

const ALLOWED_ORIGINS: string[] =
  (process.env.CORS_ORIGINS ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

// Util: CORS
function corsHeaders(origin: string | null): Record<string, string> {
  const allowAll = ALLOWED_ORIGINS.length === 0 || ALLOWED_ORIGINS.includes('*');
  const allowOrigin =
    allowAll || (origin && ALLOWED_ORIGINS.includes(origin))
      ? origin ?? '*'
      : ALLOWED_ORIGINS[0] ?? '*';

  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

function json<T>(data: T, status: number, origin: string | null): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...corsHeaders(origin),
    },
  });
}

export async function OPTIONS(req: Request): Promise<Response> {
  const origin = req.headers.get('origin');
  return new Response(null, { headers: corsHeaders(origin) });
}

export async function POST(req: Request): Promise<Response> {
  const origin = req.headers.get('origin');

  let body: RAGRequest;
  try {
    body = (await req.json()) as RAGRequest;
  } catch {
    return json<ErrorPayload>({ error: 'Bad JSON' }, 400, origin);
  }

  const q = (body.q ?? '').trim();
  const lang = body.lang ?? 'fr';
  if (!q) {
    return json<ErrorPayload>({ error: 'Missing "q"' }, 400, origin);
  }

  // TODO: brancher ici ton vrai RAG (embeddings + recherche).
  // Réponse factice pour valider l’intégration Shopify :
  const payload: RAGResponse = {
    answer: `Sempaï Kinko : j’ai bien reçu ta question « ${q} ». (lang=${lang})`,
    sources: [],
  };

  return json<RAGResponse>(payload, 200, origin);
}
