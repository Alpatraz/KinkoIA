// scripts/ingest_faq_items.js
// Ingestion des métaobjets Shopify (faq_item) vers Supabase avec embeddings locaux (MiniLM 384d)

import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import { createClient } from "@supabase/supabase-js";
import { pipeline } from "@xenova/transformers";

// ------- ENV -------
const {
  SHOPIFY_STORE_DOMAIN,
  SHOPIFY_STOREFRONT_TOKEN,
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE,
} = process.env;

if (!SHOPIFY_STORE_DOMAIN || !SHOPIFY_STOREFRONT_TOKEN || !SUPABASE_URL || !SUPABASE_SERVICE_ROLE) {
  console.error("❌ Variables manquantes. Vérifie .env.local : SHOPIFY_STORE_DOMAIN, SHOPIFY_STOREFRONT_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_ROLE");
  process.exit(1);
}

// ------- Clients -------
const supa = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE);

// ------- Helpers -------
function richTextToPlain(rt) {
  if (!rt) return "";
  try {
    const obj = JSON.parse(rt);
    return (obj.children || [])
      .map((p) => (p.children || []).map((t) => t?.value || "").join(""))
      .join("\n\n")
      .trim();
  } catch {
    return String(rt || "");
  }
}

function chunk(text, max = 900) {
  // Découpage par paragraphes pour garder du sens
  const raw = (text || "").split(/\n+/).map((s) => s.trim()).filter(Boolean);
  const out = [];
  let buf = [];
  let size = 0;
  for (const line of raw) {
    if (size + line.length + 1 > max && buf.length) {
      out.push(buf.join("\n"));
      buf = [];
      size = 0;
    }
    buf.push(line);
    size += line.length + 1;
  }
  if (buf.length) out.push(buf.join("\n"));
  return out.length ? out : [text.slice(0, max)];
}

// ------- Embeddings locaux (MiniLM 384d) -------
let _fe; // cache du pipeline
async function getEmbedder() {
  if (!_fe) {
    _fe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2"); // 384 dims
  }
  return _fe;
}

async function embed(texts) {
  const fe = await getEmbedder();
  const vectors = [];
  for (const t of texts) {
    // pooling = mean, normalize = true -> vecteur unitaire
    const out = await fe(t, { pooling: "mean", normalize: true });
    vectors.push(Array.from(out.data)); // Float32Array -> Array<number>
  }
  return vectors;
}

// ------- Shopify : lire les metaobjects faq_item -------
async function fetchFaqItems() {
  const query = `
    query {
      metaobjects(type: "faq_item", first: 200) {
        nodes {
          handle
          fields { key value }
        }
      }
    }`;

  const r = await fetch(`https://${SHOPIFY_STORE_DOMAIN}/api/2024-07/graphql.json`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Shopify-Storefront-Access-Token": SHOPIFY_STOREFRONT_TOKEN,
    },
    body: JSON.stringify({ query }),
  });

  const j = await r.json().catch(async () => {
    const txt = await r.text();
    throw new Error("Réponse non-JSON de Shopify: " + txt.slice(0, 200));
  });

  if (j.errors) {
    console.error("⚠️ GraphQL errors:", j.errors);
  }

  const nodes = j?.data?.metaobjects?.nodes || [];
  return nodes.map((n) => {
    const get = (k) => n.fields?.find((f) => f.key === k)?.value;

    return {
      handle: n.handle,
      question: get("question") || "",
      answer: richTextToPlain(get("answer") || ""),
      centers: get("centers") || "", // optionnel selon ta définition
      tags: get("tags") || "",       // optionnel
      lang: (get("lang") || "fr").toLowerCase().startsWith("en") ? "en" : "fr",
    };
  });
}

// ------- Ingestion principale -------
async function ingest() {
  console.log("⏳ Lecture des FAQ Shopify…");
  const items = await fetchFaqItems();
  if (!items.length) {
    console.log("Aucun faq_item trouvé.");
    return;
  }
  console.log(`→ ${items.length} items.`);

  let inserted = 0;

  for (const it of items) {
    const base = `${it.question}\n\n${it.answer}`.trim();
    if (!base) continue;

    const parts = chunk(base, 900);
    const embs = await embed(parts);

    const rows = parts.map((content, i) => ({
      content,
      url: `/pages/faq#${it.handle}`,
      source: "faq_item",
      organizer: it.centers || it.tags || null,
      lang: it.lang || "fr",
      embedding: embs[i], // <- vector(384)
    }));

    const { error } = await supa.from("faq_chunks").insert(rows);
    if (error) {
      console.error(`❌ Insert error pour ${it.handle}:`, error.message);
    } else {
      inserted += rows.length;
      console.log(`✔️ ${it.handle} (+${rows.length})`);
    }
  }

  console.log(`\n✅ Ingestion terminée. Chunks insérés: ${inserted}`);
}

// ------- Run -------
ingest().catch((e) => {
  console.error("❌ Ingestion échouée:", e.message || e);
  process.exit(1);
});
