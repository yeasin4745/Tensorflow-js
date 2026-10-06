'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { trainLinearRegression } = require('../tf');

test('trainLinearRegression learns y = 5x + 2', async () => {
  const prediction = await trainLinearRegression(2);
  assert.ok(Math.abs(prediction - 12) < 0.1, `expected about 12, got ${prediction}`);
});
