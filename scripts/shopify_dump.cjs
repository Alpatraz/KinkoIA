#!/usr/bin/env node
/* Dump Shopify content to ./ingested as Markdown.
 * Reads .env.local for SHOPIFY_SHOP and SHOPIFY_ADMIN_TOKEN.
 * Works even when the storefront is password-protected (uses Admin API).
 */

"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");

// Load env (prefer .env.local; fall back to default .env)
const dotenv = require("dotenv");
dotenv.config({ path: path.join(process.cwd(), ".env.local") });
dotenv.config(); // fallback if some vars are only in .env

const SHOP = process.env.SHOPIFY_SHOP || process.env.SHOPIFY_STORE_DOMAIN;
const ADMIN_TOKEN = process.env.SHOPIFY_ADMIN_TOKEN;
const API_VERSION = process.env.SHOPIFY_API_VERSION || "2024-07";
const OUT_DIR = path.join(process.cwd(), "ingested");

if (!SHOP || !ADMIN_TOKEN) {
  console.error("❌ SHOPIFY_SHOP ou SHOPIFY_ADMIN_TOKEN manquant dans .env.local");
  process.exit(1);
}

const ADMIN_BASE = `https://${SHOP}/admin/api/${API_VERSION}`;
const PUBLIC_BASE =
  (process.env.SHOPIFY_PUBLIC_BASE && process.env.SHOPIFY_PUBLIC_BASE.replace(/\/+$/, "")) ||
  `https://${SHOP}`;

function adminHeaders() {
  return {
    "Content-Type": "application/json",
    "X-Shopify-Access-Token": ADMIN_TOKEN,
  };
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// Shopify "Link" header → page_info (for pagination)
function nextPageInfoFromLink(linkHeader) {
  if (!linkHeader) return null;
  // Example: <https://shop/admin/api/2024-07/products.json?limit=250&page_info=XXXX>; rel="next"
  const parts = linkHeader.split(",");
  for (const p of parts) {
    if (p.includes('rel="next"')) {
      const m = p.match(/<([^>]+)>/);
      if (!m) continue;
      const url = new URL(m[1]);
      return url.searchParams.get("page_info");
    }
  }
  return null;
}

async function adminGetAll(pathname, query = {}) {
  const results = [];
  let pageInfo = null;
  do {
    const url = new URL(ADMIN_BASE + pathname);
    // default limit 250
    url.searchParams.set("limit", String(query.limit ?? 250));
    for (const [k, v] of Object.entries(query)) {
      if (k !== "limit") url.searchParams.set(k, String(v));
    }
    if (pageInfo) url.searchParams.set("page_info", pageInfo);

    const res = await fetch(url, { headers: adminHeaders() });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`GET ${pathname} ${res.status}: ${text}`);
    }
    const json = await res.json();

    // top-level key depends on resource:
    // pages: { pages: [...] }, blogs: { blogs: [...] }, articles: { articles: [...] }, etc.
    const key = Object.keys(json).find((k) => Array.isArray(json[k]));
    if (key) results.push(...json[key]);

    pageInfo = nextPageInfoFromLink(res.headers.get("Link"));
    // be nice to API
    if (pageInfo) await sleep(200);
  } while (pageInfo);

  return results;
}

function sanitizeFileName(s) {
  return s
    .toLowerCase()
    .replace(/https?:\/\//g, "")
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

async function writeMarkdown({ url, title, body, kind, id }) {
  const safe = sanitizeFileName(`${url || title || id || kind}.md`);
  const file = path.join(OUT_DIR, safe);
  const fm = [
    "---",
    `url: ${url || ""}`,
    `title: "${(title || "").replace(/"/g, '\\"')}"`,
    `kind: ${kind}`,
    `source: shopify_admin`,
    "---",
    "",
  ].join("\n");

  const md = `${fm}${body || ""}\n`;
  await fs.writeFile(file, md, "utf8");
  return file;
}

async function dumpPages() {
  const pages = await adminGetAll("/pages.json");
  const files = [];
  for (const p of pages) {
    const url = `${PUBLIC_BASE}/pages/${p.handle}`;
    const body = (p.body_html || "")
      .replace(/<\/?script[^>]*>/gi, "")
      .replace(/\r/g, "");
    files.push(
      await writeMarkdown({
        url,
        title: p.title,
        body,
        kind: "page",
        id: p.id,
      })
    );
  }
  console.log(` - pages: ${files.length} fichier(s)`);
  return files;
}

async function dumpBlogsAndArticles() {
  const blogs = await adminGetAll("/blogs.json");
  let count = 0;
  for (const b of blogs) {
    const blogBase = `${PUBLIC_BASE}/blogs/${b.handle}`;
    const articles = await adminGetAll(`/blogs/${b.id}/articles.json`);
    for (const a of articles) {
      const url = `${blogBase}/${a.handle}`;
      const body = (a.body_html || "")
        .replace(/<\/?script[^>]*>/gi, "")
        .replace(/\r/g, "");
      await writeMarkdown({
        url,
        title: a.title,
        body,
        kind: "article",
        id: a.id,
      });
      count++;
    }
  }
  console.log(` - articles: ${count} fichier(s)`);
}

async function dumpProducts() {
  const products = await adminGetAll("/products.json", { fields: "id,title,handle,body_html,tags,vendor" });
  let count = 0;
  for (const p of products) {
    const url = `${PUBLIC_BASE}/products/${p.handle}`;
    const body = [
      `# ${p.title}\n`,
      p.body_html || "",
      "",
      p.vendor ? `**Vendor:** ${p.vendor}` : "",
      p.tags ? `**Tags:** ${p.tags}` : "",
    ]
      .filter(Boolean)
      .join("\n");
    await writeMarkdown({
      url,
      title: p.title,
      body,
      kind: "product",
      id: p.id,
    });
    count++;
  }
  console.log(` - products: ${count} fichier(s)`);
}

async function dumpCollections() {
  const custom = await adminGetAll("/custom_collections.json");
  const smart = await adminGetAll("/smart_collections.json");
  const all = [...custom, ...smart];
  let count = 0;
  for (const c of all) {
    const url = `${PUBLIC_BASE}/collections/${c.handle}`;
    const body = [`# ${c.title}\n`, c.body_html || ""].join("\n");
    await writeMarkdown({
      url,
      title: c.title,
      body,
      kind: "collection",
      id: c.id,
    });
    count++;
  }
  console.log(` - collections: ${count} fichier(s)`);
}

(async function main() {
  console.log(`🔓 Shopify dump from ${SHOP} (${API_VERSION}) → ${OUT_DIR}`);
  await fs.mkdir(OUT_DIR, { recursive: true });

  try {
    await dumpPages();
    await dumpBlogsAndArticles();
    await dumpProducts();
    await dumpCollections();
    console.log("✅ Terminé.");
    process.exit(0);
  } catch (err) {
    console.error("❌ Échec:", err?.message || err);
    process.exit(1);
  }
})();
