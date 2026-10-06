'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  compareTexts,
  cosineSimilarity,
  createEmbeddings,
  validateTexts,
} = require('../index');

test('cosineSimilarity returns 1 for identical vectors', async () => {
  assert.ok(Math.abs((await cosineSimilarity([1, 2, 3], [1, 2, 3])) - 1) < 1e-6);
});

test('cosineSimilarity returns 0 for orthogonal vectors', async () => {
  assert.ok(Math.abs(await cosineSimilarity([1, 0], [0, 1])) < 1e-6);
});

test('cosineSimilarity rejects vectors with different dimensions', async () => {
  await assert.rejects(cosineSimilarity([1], [1, 2]), /Vector size mismatch/);
});

test('cosineSimilarity rejects zero vectors and non-finite values', async () => {
  await assert.rejects(cosineSimilarity([0, 0], [1, 2]), /zero vector/);
  await assert.rejects(cosineSimilarity([1, Number.NaN], [1, 2]), /finite numbers/);
});

test('validateTexts accepts a single string and rejects empty input', () => {
  assert.deepEqual(validateTexts('hello'), ['hello']);
  assert.throws(() => validateTexts(['hello', '  ']), /non-empty text strings/);
});

test('createEmbeddings validates the Ollama response shape', async () => {
  const fetchImpl = async () => ({
    ok: true,
    json: async () => ({ embeddings: [[0.25, 0.75]] }),
  });

  assert.deepEqual(
    await createEmbeddings('hello', { fetchImpl }),
    [[0.25, 0.75]],
  );

  await assert.rejects(
    createEmbeddings(['one', 'two'], { fetchImpl }),
    /exactly 2 embedding vectors/,
  );
});

test('compareTexts returns cosine similarity and angle without a live Ollama server', async () => {
  const fetchImpl = async () => ({
    ok: true,
    json: async () => ({ embeddings: [[1, 0], [0, 1]] }),
  });
  const result = await compareTexts('first', 'second', { fetchImpl });

  assert.ok(Math.abs(result.similarity) < 1e-6);
  assert.ok(Math.abs(result.angleDegrees - 90) < 1e-4);
  assert.equal(result.dimensions, 2);
});
