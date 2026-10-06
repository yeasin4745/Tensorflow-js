'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { trainAndPredict } = require('../model');

test('trainAndPredict learns all three matrix-defined outputs', async () => {
  const prediction = await trainAndPredict([2, 3]);
  const expected = [14, 9, 10];

  prediction.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) < 0.1, `output ${index + 1}: expected ${expected[index]}, got ${value}`);
  });
});
