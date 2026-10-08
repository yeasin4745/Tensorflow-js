/**
 * ============================================================
 *  house-price-prediction.js
 *  ------------------------------------------------------------
 *  Category : Practical Regression Project
 *  Purpose  : Multi-feature Linear Regression — বাড়ির Size,
 *             Bedrooms আর YearBuilt থেকে Price প্রেডিক্ট করা।
 *             CSV লোড → z-score normalization → training →
 *             prediction → denormalize।
 *
 *  Run      : npm run house-price
 *  Backend  : WASM (tfjs-node ছাড়াই Termux-এ চলে)
 * ============================================================
 */

import tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";

const { log } = console;

const CSV_URL =
  "https://raw.githubusercontent.com/yeasin4745/csv-datasets/main/HousingMarketData.csv";

/* ---------- 1. Load CSV Dataset ---------- */

async function loadDataSet() {
  const dataset = tf.data.csv(CSV_URL, {
    hasHeader: true,
    columnConfigs: { Price: { isLabel: true } },
  });

  const size = [], bedrooms = [], year = [], price = [];

  await dataset.forEachAsync((row) => {
    size.push(parseFloat(row.Size));
    bedrooms.push(parseFloat(row.Bedrooms));
    year.push(parseFloat(row.YearBuilt));
    price.push(parseFloat(row.Price));
  });

  return { size, bedrooms, year, price };
}

/* ---------- 2. Z-score Normalization ---------- */
// z = (x - mean) / std
// Feature-গুলোর scale আলাদা হলে training unstable হয় —
// তাই সবকে একই রেঞ্জে (mean 0, std 1) আনা হয়।

function zNormalize(array) {
  const tensor = tf.tensor1d(array);
  const mean = tensor.mean().arraySync();
  const variance = tensor.sub(mean).square().mean().arraySync();
  const std = Math.sqrt(variance);
  const norm = array.map((n) => (n - mean) / std);
  return { norm, mean, std };
}

/* ---------- 3. Model ---------- */

function createModel() {
  const model = tf.sequential();
  // 3 features → 1 output, linear activation = linear regression
  model.add(tf.layers.dense({ inputShape: [3], units: 1, activation: "linear" }));
  return model;
}

/* ---------- 4. Main Pipeline ---------- */

async function main(inputRaw = [1020, 4, 2019]) {
  await tf.setBackend("wasm");
  await tf.ready();
  log("Backend:", tf.getBackend());

  const { size, bedrooms, year, price } = await loadDataSet();
  log(`Dataset loaded: ${size.length} rows`);

  const sizeStats = zNormalize(size);
  const roomStats = zNormalize(bedrooms);
  const yearStats = zNormalize(year);
  const priceStats = zNormalize(price);

  // Feature matrix তৈরি
  const inputs = [];
  const labels = [];
  for (let i = 0; i < size.length; i++) {
    inputs.push([sizeStats.norm[i], roomStats.norm[i], yearStats.norm[i]]);
    labels.push(priceStats.norm[i]);
  }

  const xTrain = tf.tensor2d(inputs);
  const yTrain = tf.tensor2d(labels, [labels.length, 1]);

  const model = createModel();
  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(0.001),
    metrics: ["mse"],
  });

  log("Training started...");
  await model.fit(xTrain, yTrain, {
    epochs: 100,
    batchSize: 4,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 10 === 0) {
          log(
            `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`
          );
        }
      },
    },
  });

  log("\n✅ Training finished!\n");

  // নতুন ইনপুট normalize → predict → denormalize
  const inputNorm = [
    (inputRaw[0] - sizeStats.mean) / sizeStats.std,
    (inputRaw[1] - roomStats.mean) / roomStats.std,
    (inputRaw[2] - yearStats.mean) / yearStats.std,
  ];

  const prediction = tf.tidy(() => {
    const inputTensor = tf.tensor2d([inputNorm]);
    return model.predict(inputTensor);
  });

  const predNormValue = (await prediction.data())[0];
  prediction.dispose();

  // y = z·σ + μ  (denormalize)
  const predictedPrice = predNormValue * priceStats.std + priceStats.mean;

  log(
    `Predicted Price for ${inputRaw[0]} sqft, ${inputRaw[1]} beds, built ${inputRaw[2]}: ৳${Math.round(predictedPrice)}`
  );

  tf.dispose([xTrain, yTrain]);
  model.dispose();
}

main();
