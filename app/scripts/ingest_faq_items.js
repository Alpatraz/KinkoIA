import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const {
  OPENAI_API_KEY,
  SHOPIFY_STORE_DOMAIN,
  SHOPIFY_STOREFRONT_TOKEN,
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE
} = process.env;

if (!OPENAI_API_KEY || !SHOPIFY_STORE_DOMAIN || !SHOPIFY_STOREFRONT_TOKEN || !SUPABASE_URL || !SUPABASE_SERVICE_ROLE) {
  console.error("Missing env vars. Check your .env.local");
  process.exit(1);
}

const supa = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE);

function richTextToPlain(rt){
  try{
    const obj = JSON.parse(rt);
    return (obj.children||[])
      .map(p => (p.children||[]).map(t=>t?.value||"").join(""))
      .join("\n\n");
  }catch{return rt||"";}
}

function chunk(text, size=1000, overlap=150){
  const parts = [];
  for (let i=0;i<text.length;i+= (size - overlap)) parts.push(text.slice(i, i+size));
  return parts;
}

async function embed(texts){
  const r = await fetch("https://api.openai.com/v1/embeddings",{
    method:"POST",
    headers:{ "Content-Type":"application/json","Authorization":`Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model:"text-embedding-3-large", input:texts })
  });
  const j = await r.json();
  return j.data.map(d=>d.embedding);
}

async function fetchFaqItems(){
  const query = `
    query {
      metaobjects(type: "faq_item", first: 200) {
        nodes { handle fields { key value } }
      }
    }`;
  const r = await fetch(`https://${SHOPIFY_STORE_DOMAIN}/api/2024-07/graphql.json`,{
    method:"POST",
    headers:{
      "Content-Type":"application/json",
      "X-Shopify-Storefront-Access-Token": SHOPIFY_STOREFRONT_TOKEN
    },
    body: JSON.stringify({ query })
  });
  const j = await r.json();
  return (j?.data?.metaobjects?.nodes||[]).map(n=>{
    const get = k => n.fields?.find(f=>f.key===k)?.value;
    return {
      handle: n.handle,
      question: get("question")||"",
      answer: richTextToPlain(get("answer")||"")
    };
  });
}

async function run(){
  const items = await fetchFaqItems();
  for (const it of items) {
    const base = `${it.question}\n\n${it.answer}`.trim();
    if (!base) continue;
    const parts = chunk(base, 1000, 150);
    const embs = await embed(parts);
    const rows = parts.map((content, i)=> ({
      content,
      url: `/pages/faq#${it.handle}`,
      source: "FAQ interne",
      organizer: null,
      lang: "fr",
      embedding: embs[i]
    }));
    const { error } = await supa.from("faq_chunks").insert(rows);
    if (error) console.error("Insert error", error);
    else console.log(`Indexed FAQ: ${it.handle} (+${rows.length})`);
  }
  console.log("Done.");
}

run().catch(err=>{ console.error(err); process.exit(1); });
