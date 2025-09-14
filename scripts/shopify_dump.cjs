/* scripts/shopify_dump.cjs
 * Dump Shopify Pages, Blogs/Articles et Metaobjects vers ./ingested/*.md
 * Utilise l'Admin GraphQL API.
 */
const fs = require("node:fs/promises");
const path = require("node:path");
const { setTimeout: delay } = require("node:timers/promises");

// Charge .env.local si non préchargé par -r dotenv/config
try { require("dotenv").config({ path: process.env.dotenv_config_path || ".env.local" }); } catch {}

const SHOP = process.env.SHOPIFY_SHOP;
const ADMIN_TOKEN = process.env.SHOPIFY_ADMIN_TOKEN;
const API_VERSION = process.env.SHOPIFY_API_VERSION || "2024-07";
const PUBLIC_BASE = (process.env.SHOPIFY_PUBLIC_BASE || "").replace(/\/+$/,"");
const META_TYPES = (process.env.SHOPIFY_METAOBJECT_TYPES || "")
  .split(",").map(s => s.trim()).filter(Boolean);
const CALENDAR_URL = process.env.SHOPIFY_CALENDAR_URL || "";

if (!SHOP || !ADMIN_TOKEN) {
  console.error("❌ SHOPIFY_SHOP ou SHOPIFY_ADMIN_TOKEN manquant dans .env.local");
  process.exit(1);
}

const GQL_URL = `https://${SHOP}/admin/api/${API_VERSION}/graphql.json`;
const OUT_DIR = path.join(process.cwd(), "ingested");

async function adminGQL(query, variables) {
  const res = await fetch(GQL_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Shopify-Access-Token": ADMIN_TOKEN,
    },
    body: JSON.stringify({ query, variables }),
  });
  const json = await res.json();
  if (!res.ok || json.errors) {
    throw new Error(`Shopify GraphQL error: ${res.status} ${JSON.stringify(json.errors || json)}`);
  }
  return json.data;
}

function slugify(s) {
  return String(s || "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/(^-|-$)/g, "");
}

async function writeMD(name, lines) {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const file = path.join(OUT_DIR, `${name}.md`);
  await fs.writeFile(file, lines.join("\n"), "utf8");
  console.log(`   ✔️  Saved: ${path.relative(process.cwd(), file)} (${(lines.join("\n").length)} chars)`);
}

/* --------------------
   Dump Pages
-------------------- */
async function dumpPages() {
  console.log("⏳ Pages…");
  const q = `
    query Pages($cursor: String) {
      pages(first: 100, after: $cursor, sortKey: UPDATED_AT) {
        pageInfo { hasNextPage endCursor }
        nodes { id handle title bodyHtml updatedAt onlineStoreUrl }
      }
    }`;
  let cursor = null;
  let count = 0;

  while (true) {
    const data = await adminGQL(q, { cursor });
    const { nodes, pageInfo } = data.pages;
    for (const p of nodes) {
      const url = p.onlineStoreUrl || (PUBLIC_BASE ? `${PUBLIC_BASE}/pages/${p.handle}` : "");
      const md = [
        `# ${p.title || p.handle}`,
        ``,
        `Source: ${url}`,
        `Updated: ${p.updatedAt}`,
        ``,
        // bodyHtml -> texte brut simple
        String(p.bodyHtml || "")
          .replace(/<br\s*\/?>/gi, "\n")
          .replace(/<[^>]+>/g, "")
      ];
      await writeMD(`shopify_page_${p.handle}`, md);
      count++;
    }
    if (!pageInfo.hasNextPage) break;
    cursor = pageInfo.endCursor;
    await delay(150); // respirer un peu
  }
  console.log(`✅ Pages: ${count}`);
}

/* --------------------
   Dump Blogs + Articles
-------------------- */
async function dumpBlogs() {
  console.log("⏳ Blogs & Articles…");
  const qBlogs = `
    query Blogs($cursor:String) {
      blogs(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { id handle title }
      }
    }`;
  const qArticles = `
    query Articles($id:ID!, $cursor:String) {
      blog(id:$id) {
        articles(first: 100, after:$cursor, sortKey: UPDATED_AT) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id handle title contentHtml excerpt tags publishedAt onlineStoreUrl
          }
        }
      }
    }`;

  let cursor = null;
  let total = 0;
  while (true) {
    const data = await adminGQL(qBlogs, { cursor });
    const { nodes: blogs, pageInfo } = data.blogs;

    for (const b of blogs) {
      let ac = null;
      while (true) {
        const ad = await adminGQL(qArticles, { id: b.id, cursor: ac });
        const art = ad.blog.articles;
        for (const a of art.nodes) {
          const url =
            a.onlineStoreUrl || (PUBLIC_BASE ? `${PUBLIC_BASE}/blogs/${b.handle}/${a.handle}` : "");
          const text = (a.contentHtml || a.excerpt || "")
            .replace(/<br\s*\/?>/gi, "\n")
            .replace(/<[^>]+>/g, "");
          const md = [
            `# ${a.title || a.handle}`,
            ``,
            `Blog: ${b.title} (${b.handle})`,
            `Source: ${url}`,
            `Published: ${a.publishedAt || ""}`,
            `Tags: ${(a.tags || []).join(", ")}`,
            ``,
            text
          ];
          await writeMD(`shopify_blog_${b.handle}__${a.handle}`, md);
          total++;
        }
        if (!art.pageInfo.hasNextPage) break;
        ac = art.pageInfo.endCursor;
        await delay(150);
      }
    }
    if (!pageInfo.hasNextPage) break;
    cursor = pageInfo.endCursor;
    await delay(150);
  }
  console.log(`✅ Articles: ${total}`);
}

/* --------------------
   Dump Metaobjects
-------------------- */
async function dumpMetaobjects() {
  if (META_TYPES.length === 0) {
    console.log("ℹ️  Aucun type dans SHOPIFY_METAOBJECT_TYPES — étape ignorée.");
    return;
  }
  console.log(`⏳ Metaobjects (${META_TYPES.join(", ")})…`);

  const q = `
    query Meta($type:String!, $cursor:String) {
      metaobjects(type:$type, first:100, after:$cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id handle type updatedAt
          fields { key value }
        }
      }
    }`;

  let total = 0;

  for (const type of META_TYPES) {
    let cursor = null;
    while (true) {
      const data = await adminGQL(q, { type, cursor });
      const { nodes, pageInfo } = data.metaobjects;

      for (const m of nodes) {
        // Map fields -> objet clé/valeur
        const fm = {};
        for (const f of m.fields || []) fm[f.key] = f.value;

        // Champs "classiques" si présents : title/name/date/location/url/description
        const title = fm.title || fm.name || m.handle || `${m.type} ${m.id}`;
        const date =
          fm.date || fm.start_date || fm.start || fm.when || fm.date_time || "";
        const location = fm.location || fm.city || fm.place || "";
        const link = fm.url || fm.link || fm.registration || "";
        const desc = fm.description || fm.note || fm.notes || "";

        // Lien public de repli : page calendrier si fournie
        const publicUrl = link || CALENDAR_URL || "";

        // Construire un corps texte bien indexable
        const lines = [
          `# ${title}`,
          ``,
          `Type: ${m.type}`,
          `Handle: ${m.handle}`,
          `Updated: ${m.updatedAt}`,
          publicUrl ? `Source: ${publicUrl}` : `Source: (metaobject ${m.type})`,
          ``,
          date ? `Date: ${date}` : ``,
          location ? `Lieu: ${location}` : ``,
          ``,
          desc || ``,
          ``,
          `--- Champs complets ---`,
          ...Object.entries(fm).map(([k, v]) => `- ${k}: ${v}`)
        ].filter(Boolean);

        await writeMD(`shopify_meta_${type}__${slugify(m.handle)}`, lines);
        total++;
      }

      if (!pageInfo.hasNextPage) break;
      cursor = pageInfo.endCursor;
      await delay(150);
    }
  }
  console.log(`✅ Metaobjects: ${total}`);
}

/* --------------------
   main
-------------------- */
(async () => {
  console.log("▶️  Shopify dump → ./ingested");
  await dumpPages().catch(e => { console.error("Pages error:", e); });
  await dumpBlogs().catch(e => { console.error("Blogs error:", e); });
  await dumpMetaobjects().catch(e => { console.error("Metaobjects error:", e); });
  console.log("✅ Terminé.");
})();
