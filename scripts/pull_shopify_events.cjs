#!/usr/bin/env node
// pull_shopify_events.cjs
// Récupère les metaobjects "Event" (ou autres types) depuis la Storefront API
// et écrit public/events.json pour que la route API puisse les lire.
//
// ENV acceptées :
// - SHOPIFY_DOMAIN  (⇒ "qfxdmn-i3.myshopify.com")            ← préféré
//   (alias: SHOPIFY_STORE_DOMAIN, SHOPIFY_SHOP, SHOPIFY_SHOP_DOMAIN)
// - SHOPIFY_STOREFRONT_TOKEN (Storefront access token)
//   (alias: SHOPIFY_STOREFRONT_ACCESS_TOKEN, NEXT_PUBLIC_SHOPIFY_STOREFRONT_TOKEN)
// - SHOPIFY_API_VERSION (par défaut "2024-10")
// - SHOPIFY_EVENT_METAOBJECT_TYPES (ex: "Event,competition"; défaut "Event")
// - SHOPIFY_CALENDAR_URL (facultatif; mis dans chaque item.url si absent)

const fs = require("fs");
const path = require("path");

// Charge .env.local si présent
try {
  const dotenvPath = process.env.DOTENV_PATH || path.resolve(process.cwd(), ".env.local");
  if (fs.existsSync(dotenvPath)) {
    require("dotenv").config({ path: dotenvPath });
  }
} catch { /* noop */ }

function pickEnv(...keys) {
  for (const k of keys) {
    const v = process.env[k];
    if (v && String(v).trim()) return v.trim();
  }
  return "";
}

const DOMAIN = pickEnv("SHOPIFY_DOMAIN", "SHOPIFY_STORE_DOMAIN", "SHOPIFY_SHOP", "SHOPIFY_SHOP_DOMAIN", "SHOP");
const TOKEN = pickEnv("SHOPIFY_STOREFRONT_TOKEN", "SHOPIFY_STOREFRONT_ACCESS_TOKEN", "NEXT_PUBLIC_SHOPIFY_STOREFRONT_TOKEN");
const API_VERSION = process.env.SHOPIFY_API_VERSION?.trim() || "2024-10";
const TYPES = (process.env.SHOPIFY_EVENT_METAOBJECT_TYPES?.trim() || "Event")
  .split(",")
  .map(s => s.trim())
  .filter(Boolean);
const DEFAULT_URL = process.env.SHOPIFY_CALENDAR_URL?.trim() || "";

if (!DOMAIN || !TOKEN) {
  console.error(`❌ SHOPIFY_DOMAIN ou SHOPIFY_STOREFRONT_TOKEN manquant(s).
  Vu: DOMAIN="${DOMAIN || "(vide)"}", TOKEN=${TOKEN ? "(présent)" : "(vide)"}
  Indice: ajoute dans .env.local
    SHOPIFY_DOMAIN=ton-boutique.myshopify.com
    SHOPIFY_STOREFRONT_TOKEN=xxxxxxxxxxxxxxxx
`);
  process.exit(1);
}

console.log(`Shopify: domain=${DOMAIN}, version=${API_VERSION}, types=[${TYPES.join(", ")}]`);

const ENDPOINT = `https://${DOMAIN}/api/${API_VERSION}/graphql.json`;

async function gql(query, variables = {}) {
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Shopify-Storefront-Access-Token": TOKEN,
      "Accept": "application/json",
    },
    body: JSON.stringify({ query, variables }),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  const j = await res.json();
  if (j.errors) throw new Error("GraphQL errors: " + JSON.stringify(j.errors));
  return j.data;
}

const QUERY = `
  query Events($type: String!, $cursor: String) {
    metaobjects(type: $type, first: 250, after: $cursor) {
      nodes {
        type
        handle
        updatedAt
        fields { key value }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
`;

function field(obj, key) {
  const f = obj.fields?.find(x => x.key === key);
  return f?.value?.trim() || "";
}

function toEvent(node) {
  // clés possibles selon ton modèle: title, start, end, location, organizer, tags, url
  // on tolère aussi date/when, city
  const title = field(node, "title") || field(node, "name") || node.handle;
  const start = field(node, "start") || field(node, "date") || field(node, "when");
  const end = field(node, "end") || "";
  const location = field(node, "location") || field(node, "city") || "";
  const organizer = field(node, "organizer") || field(node, "host") || "";
  const url = field(node, "url") || DEFAULT_URL || "";
  let tags = field(node, "tags");
  try {
    // si tags est JSON: ["a","b"]
    const t = JSON.parse(tags);
    if (Array.isArray(t)) tags = t.join(",");
  } catch {}
  return {
    title,
    start,
    end: end || undefined,
    location: location || undefined,
    url: url || undefined,
    organizer: organizer || undefined,
    tags: tags ? String(tags).split(/[;,]/).map(s => s.trim()).filter(Boolean) : undefined,
    _sourceType: node.type,
    _updatedAt: node.updatedAt,
    _handle: node.handle,
  };
}

async function fetchAllForType(type) {
  const out = [];
  let cursor = null;
  let page = 1;
  while (true) {
    const data = await gql(QUERY, { type, cursor });
    const m = data?.metaobjects;
    const nodes = m?.nodes || [];
    console.log(` - ${type} p${page}: ${nodes.length} item(s)`);
    for (const n of nodes) out.push(toEvent(n));
    if (m?.pageInfo?.hasNextPage) {
      cursor = m.pageInfo.endCursor;
      page++;
    } else {
      break;
    }
  }
  return out;
}

(async () => {
  const all = [];
  for (const t of TYPES) {
    try {
      const items = await fetchAllForType(t);
      all.push(...items);
    } catch (e) {
      console.warn(`⚠️ Type "${t}" échoué: ${e.message}`);
    }
  }

  // Tri par date de début si possible
  all.sort((a, b) => {
    const ta = Date.parse(a.start || "") || 0;
    const tb = Date.parse(b.start || "") || 0;
    return ta - tb;
  });

  const outPath = path.join(process.cwd(), "public", "events.json");
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(all, null, 2), "utf8");
  console.log(`✅ Écrit: ${outPath} (${all.length} évènement(s))`);
})().catch(err => {
  console.error("❌ Fatal:", err?.stack || err?.message || String(err));
  process.exit(1);
});
