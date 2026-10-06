'use strict';

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-wasm');

const DEFAULT_OLLAMA_EMBED_URL =
  process.env.OLLAMA_EMBED_URL || 'http://127.0.0.1:11434/api/embed';
const DEFAULT_EMBEDDING_MODEL =
  process.env.EMBEDDING_MODEL || 'embeddinggemma';

function validateTexts(texts) {
  const input = Array.isArray(texts) ? texts : [texts];

  if (input.length === 0 || input.some((text) => typeof text !== 'string' || !text.trim())) {
    throw new TypeError('Provide one or more non-empty text strings.');
  }

  return input;
}

async function createEmbeddings(
  texts,
  {
    url = DEFAULT_OLLAMA_EMBED_URL,
    model = DEFAULT_EMBEDDING_MODEL,
    fetchImpl = globalThis.fetch,
    signal,
  } = {},
) {
  const input = validateTexts(texts);
  if (typeof fetchImpl !== 'function') {
    throw new Error('A Fetch API implementation is required (Node.js 18 or newer).');
  }

  const response = await fetchImpl(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, input }),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama API error (${response.status}): ${errorText}`);
  }

  const data = await response.json();
  if (!Array.isArray(data.embeddings) || data.embeddings.length !== input.length) {
    throw new Error(`Ollama must return exactly ${input.length} embedding vectors.`);
  }

  const dimensions = data.embeddings[0]?.length;
  if (!Number.isInteger(dimensions) || dimensions === 0) {
    throw new Error('Ollama returned an empty or invalid embedding vector.');
  }

  for (const [index, vector] of data.embeddings.entries()) {
    if (
      !Array.isArray(vector) ||
      vector.length !== dimensions ||
      vector.some((value) => !Number.isFinite(value))
    ) {
      throw new Error(`Ollama returned an invalid embedding at index ${index}.`);
    }
  }

  return data.embeddings;
}

async function cosineSimilarity(vector1, vector2) {
  if (!Array.isArray(vector1) || !Array.isArray(vector2) || vector1.length === 0) {
    throw new TypeError('Cosine similarity expects two non-empty numeric arrays.');
  }
  if (vector1.length !== vector2.length) {
    throw new RangeError(`Vector size mismatch: ${vector1.length} !== ${vector2.length}`);
  }
  if (
    [...vector1, ...vector2].some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError('Embedding vectors must contain only finite numbers.');
  }

  await tf.ready();
  return tf.tidy(() => {
    const a = tf.tensor1d(vector1, 'float32');
    const b = tf.tensor1d(vector2, 'float32');
    const dotProduct = tf.sum(tf.mul(a, b));
    const magnitudeA = tf.sqrt(tf.sum(tf.square(a)));
    const magnitudeB = tf.sqrt(tf.sum(tf.square(b)));
    const denominator = tf.mul(magnitudeA, magnitudeB).dataSync()[0];

    if (denominator === 0) {
      throw new RangeError('Cosine similarity is undefined for a zero vector.');
    }

    const similarity = tf.div(dotProduct, tf.mul(magnitudeA, magnitudeB)).dataSync()[0];
    return Math.max(-1, Math.min(1, similarity));
  });
}

async function compareTexts(text1, text2, options = {}) {
  const [embedding1, embedding2] = await createEmbeddings([text1, text2], options);
  const similarity = await cosineSimilarity(embedding1, embedding2);
  const angleDegrees = (Math.acos(similarity) * 180) / Math.PI;

  return {
    similarity,
    angleDegrees,
    dimensions: embedding1.length,
  };
}

async function main(args = process.argv.slice(2)) {
  if (args.length < 2) {
    throw new Error('Usage: node index.js "first text" "second text"');
  }

  await tf.setBackend('wasm');
  await tf.ready();
  console.log(`TensorFlow.js backend: ${tf.getBackend()}`);

  const [text1, ...remainingText] = args;
  const text2 = remainingText.join(' ');
  console.log('Generating embeddings with Ollama...');

  const result = await compareTexts(text1, text2);
  console.log(`Embedding dimensions: ${result.dimensions}`);
  console.log(`Cosine similarity: ${result.similarity.toFixed(6)}`);
  console.log(`Angle: ${result.angleDegrees.toFixed(2)}°`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { compareTexts, cosineSimilarity, createEmbeddings, validateTexts };
