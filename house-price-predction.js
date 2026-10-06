'use strict';

const tf = require('@tensorflow/tfjs');

const DATASET_URL =
  process.env.HOUSING_CSV_URL ||
  'https://raw.githubusercontent.com/yeasin4745/csv-datasets/main/datasets/regression/housing_market_data.csv';
const FEATURE_COLUMNS = ['Size_sqft', 'Bedrooms', 'Year_Built'];
const LABEL_COLUMN = 'Price';
const DEFAULT_INPUT = [1020, 4, 2019];

function parseCsv(text) {
  const [headerLine, ...dataLines] = text.trim().split(/\r?\n/);
  if (!headerLine || dataLines.length === 0) {
    throw new Error('The housing CSV is empty or has no data rows.');
  }

  const headers = headerLine.split(',').map((header) => header.trim());
  const requiredColumns = [...FEATURE_COLUMNS, LABEL_COLUMN];
  const indexes = requiredColumns.map((column) => headers.indexOf(column));
  if (indexes.some((index) => index < 0)) {
    throw new Error(`CSV must contain columns: ${requiredColumns.join(', ')}.`);
  }

  const rows = dataLines.filter((line) => line.trim()).map((line, rowIndex) => {
    const cells = line.split(',').map((cell) => cell.trim());
    const values = indexes.map((index) => Number(cells[index]));
    if (values.some((value) => !Number.isFinite(value))) {
      throw new Error(`CSV row ${rowIndex + 2} contains an invalid numeric value.`);
    }
    return values;
  });

  if (rows.length < 2) {
    throw new Error('At least two valid housing rows are required for training.');
  }
  return rows;
}

async function loadDataSet(url = DATASET_URL, fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== 'function') {
    throw new Error('A Fetch API implementation is required (Node.js 18 or newer).');
  }
  const response = await fetchImpl(url);
  if (!response.ok) {
    throw new Error(`Could not download housing CSV (${response.status}): ${url}`);
  }
  return parseCsv(await response.text());
}

function getNormalizationStats(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const standardDeviation = Math.sqrt(variance) || 1;
  return { mean, standardDeviation };
}

function normalize(value, stats) {
  return (value - stats.mean) / stats.standardDeviation;
}

function parsePredictionInput(args) {
  if (args.length === 0) return DEFAULT_INPUT;
  const values = args.map(Number);
  if (values.length !== FEATURE_COLUMNS.length || values.some((value) => !Number.isFinite(value))) {
    throw new TypeError('Usage: node house-price-predction.js <size_sqft> <bedrooms> <year_built>');
  }
  return values;
}

async function trainAndPredict(inputRaw = DEFAULT_INPUT) {
  if (!Array.isArray(inputRaw) || inputRaw.length !== FEATURE_COLUMNS.length || inputRaw.some((value) => !Number.isFinite(value))) {
    throw new TypeError('Prediction input must contain three finite numbers: size, bedrooms, and year built.');
  }

  const rows = await loadDataSet();
  const featureStats = FEATURE_COLUMNS.map((_, column) =>
    getNormalizationStats(rows.map((row) => row[column])),
  );
  const priceStats = getNormalizationStats(rows.map((row) => row[3]));
  const normalizedInputs = rows.map((row) =>
    row.slice(0, 3).map((value, column) => normalize(value, featureStats[column])),
  );
  const normalizedPrices = rows.map((row) => normalize(row[3], priceStats));
  const xTrain = tf.tensor2d(normalizedInputs);
  const yTrain = tf.tensor2d(normalizedPrices, [normalizedPrices.length, 1]);
  const model = tf.sequential();

  try {
    model.add(tf.layers.dense({ units: 1, inputShape: [FEATURE_COLUMNS.length], activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    console.log(`Training on ${rows.length} CSV rows...`);
    await model.fit(xTrain, yTrain, {
      epochs: 200,
      batchSize: Math.min(8, rows.length),
      validationSplit: rows.length >= 5 ? 0.2 : 0,
      verbose: 0,
    });

    const normalizedInput = inputRaw.map((value, index) => normalize(value, featureStats[index]));
    const inputTensor = tf.tensor2d([normalizedInput]);
    const prediction = model.predict(inputTensor);
    try {
      const [normalizedPrice] = await prediction.data();
      const predictedPrice = normalizedPrice * priceStats.standardDeviation + priceStats.mean;
      console.log(`Predicted price for ${inputRaw[0]} sqft, ${inputRaw[1]} bedrooms, built ${inputRaw[2]}: ${predictedPrice.toFixed(2)}`);
      return predictedPrice;
    } finally {
      inputTensor.dispose();
      prediction.dispose();
    }
  } finally {
    xTrain.dispose();
    yTrain.dispose();
    model.dispose();
  }
}

if (require.main === module) {
  Promise.resolve()
    .then(() => trainAndPredict(parsePredictionInput(process.argv.slice(2))))
    .catch((error) => {
    console.error(`Failed: ${error.message}`);
    process.exitCode = 1;
    });
}

module.exports = {
  getNormalizationStats,
  loadDataSet,
  normalize,
  parseCsv,
  parsePredictionInput,
  trainAndPredict,
};
