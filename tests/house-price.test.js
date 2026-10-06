'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  getNormalizationStats,
  normalize,
  parseCsv,
  parsePredictionInput,
} = require('../house-price-predction');

test('parseCsv maps the documented feature and label columns', () => {
  const csv = [
    'Size_sqft,Bedrooms,Year_Built,Price',
    '800,2,2001,120000',
    '1000,3,2010,180000',
  ].join('\n');
  assert.deepEqual(parseCsv(csv), [[800, 2, 2001, 120000], [1000, 3, 2010, 180000]]);
});

test('parseCsv rejects unexpected headers and non-numeric rows', () => {
  assert.throws(() => parseCsv('Size,Bedrooms,Year,Price\n1,2,3,4'), /must contain columns/);
  assert.throws(() => parseCsv('Size_sqft,Bedrooms,Year_Built,Price\nno,2,3,4'), /invalid numeric value/);
});

test('constant features normalize safely instead of dividing by zero', () => {
  const stats = getNormalizationStats([7, 7, 7]);
  assert.equal(normalize(7, stats), 0);
});

test('prediction input accepts exactly three finite values', () => {
  assert.deepEqual(parsePredictionInput(['900', '3', '2012']), [900, 3, 2012]);
  assert.throws(() => parsePredictionInput(['900', '3']), /Usage:/);
});
