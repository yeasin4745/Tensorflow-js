/**
 * ============================================================
 *  embedding-similarity.js
 *  ------------------------------------------------------------
 *  Category : Embeddings & Semantic Similarity
 *  Purpose  : Ollama-এর লোকাল embedding model দিয়ে টেক্সটকে
 *             vector-এ রূপান্তর করা, তারপর দুটি টেক্সটের
 *             cosine similarity ও তাদের মধ্যকার angle (θ)
 *             বের করা। Semantic search-এর ভিত্তি এখানেই।
 *
 *  Setup    : ollama pull embeddinggemma
 *  Run      : npm run embed -- "text 1" "text 2"
 *             npm run embed:test   (built-in demo)
 *
 *  Math     : cos(θ) = (A·B) / (|A| × |B|)
 *             θ      = acos(cos θ) × (180 / π)
 * ============================================================
 */

import tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";

const OLLAMA_EMBED_URL =
  process.env.OLLAMA_EMBED_URL || "http://127.0.0.1:11434/api/embed";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "embeddinggemma";

/* ---------- 1. Ollama Embedding API Call ---------- */

async function createEmbeddings(texts) {
  const input = Array.isArray(texts) ? texts : [texts];

  const response = await fetch(OLLAMA_EMBED_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama API error (${response.status}): ${errorText}`);
  }

  const data = await response.json();
  if (!Array.isArray(data.embeddings)) {
    throw new Error("Ollama response does not contain embeddings.");
  }
  return data.embeddings;
}

/* ---------- 2. Cosine Similarity + Angle ---------- */

function similarity(v1, v2) {
  return tf.tidy(() => {
    const a = tf.tensor1d(v1, "float32");
    const b = tf.tensor1d(v2, "float32");

    const dotProduct = tf.sum(tf.mul(a, b));
    const magA = tf.sqrt(tf.sum(tf.square(a)));
    const magB = tf.sqrt(tf.sum(tf.square(b)));

    const cosine = dotProduct.div(magA.mul(magB)).clamp(-1, 1); // float এরর থেকে বাঁচতে
    return cosine;
  });
}

function angleDegrees(cosine) {
  // θ = acos(cos) × (180/π)
  return Math.acos(Math.min(1, Math.max(-1, cosine))) * (180 / Math.PI);
}

/* ---------- 3. Compare Two Texts ---------- */

async function compareTexts(text1, text2) {
  log("Text 1:", text1);
  log("Text 2:", text2);
  log("Embedding started...");

  const embeddings = await createEmbeddings([text1, text2]);
  if (embeddings.length !== 2) {
    throw new Error("Expected two embedding vectors.");
  }

  log(`Embedding done — vector dimension: ${embeddings[0].length}`);

  const cosineTensor = similarity(embeddings[0], embeddings[1]);
  const cosine = (await cosineTensor.data())[0];
  cosineTensor.dispose();

  const theta = angleDegrees(cosine);

  log("─".repeat(40));
  log(`Cosine similarity : ${cosine.toFixed(6)}`);
  log(`Angle (θ)         : ${theta.toFixed(2)}°`);
  log(
    `Interpretation    : ${
      cosine > 0.85
        ? "🟢 প্রায় একই অর্থ (semantically identical)"
        : cosine > 0.5
        ? "🟡 কাছাকাছি অর্থ (related)"
        : cosine > 0
        ? "🟠 দুর্বল সম্পর্ক"
        : "🔴 ভিন্ন অর্থ"
    }`
  );
}

/* ---------- 4. Semantic Search Demo ---------- */
// একটি query আর কয়েকটি candidate ডকুমেন্টের মধ্যে
// সবচেয়ে কাছের ম্যাচ বের করা — vector search-এর মিনি সংস্করণ।

async function semanticSearchDemo() {
  const query = "How do I learn programming?";
  const documents = [
    "The best way to start coding is by building small projects.",
    "Cooking biryani requires basmati rice and spices.",
    "Practice JavaScript every day to improve your skills.",
    "The stock market fluctuated heavily this quarter.",
  ];

  const [queryEmb, ...docEmbs] = await createEmbeddings([query, ...documents]);

  const results = await Promise.all(
    docEmbs.map(async (doc) => {
      const t = similarity(queryEmb, doc);
      const cos = (await t.data())[0];
      t.dispose();
      return cos;
    })
  );

  log("\n=== Semantic Search Demo ===");
  log(`Query: "${query}"\n`);
  results
    .map((cos, i) => ({ doc: documents[i], cos, theta: angleDegrees(cos) }))
    .sort((x, y) => y.cos - x.cos)
    .forEach((r, rank) => {
      log(`${rank + 1}. (${r.cos.toFixed(4)} | ${r.theta.toFixed(1)}°) ${r.doc}`);
    });
}

/* ---------- Entry Point ---------- */

async function main() {
  await tf.setBackend("wasm");
  await tf.ready();
  log("TensorFlow backend:", tf.getBackend(), "\n");

  const [, , t1, t2] = process.argv;

  if (t1 && t2) {
    await compareTexts(t1, t2);
  } else {
    // কোনো argument না দিলে demo চলবে
    await compareTexts(
      "I love learning about neural networks",
      "Deep learning models are fascinating to study"
    );
    await semanticSearchDemo();
  }
}

main().catch((error) => {
  console.error("Failed:", error.message);
  process.exitCode = 1;
});
