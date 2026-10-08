/**
 * ============================================================
 *  models.js
 *  ------------------------------------------------------------
 *  Category : Neural Network Models (Training & Prediction)
 *  Purpose  : দুটি মডেল একসাথে —
 *             1) Simple Linear Regression: y = 5x + 2 শেখা
 *             2) Matrix-based Dense Model: ম্যানুয়ালি W·x + b
 *                দিয়ে আউটপুট বানিয়ে সেই ম্যাপিং ট্রেইন করা
 *
 *  Run      : npm run models
 *  Backend  : WASM (Termux/Android-বান্ধব)
 * ============================================================
 */

import tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";

const { log } = console;

/* ============================================================
 * Model 1 — Simple Linear Regression (y = 5x + 2)
 * ============================================================ */

// Dataset: ৩০টি র‍্যান্ডম X, আর Y = 5X + 2
const x = tf.truncatedNormal([30], 0, 10, "int32").reshape([30, 1]);
const y = tf.mul(x, 5).add(2);

async function trainLinearModel(testValue = 2) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.sgd(0.05),
  });

  await model.fit(x, y, { epochs: 200 });
  log("✅ Linear model trained (expected: y = 5x + 2)");

  const pred = model.predict(tf.tensor([[testValue]]));
  pred.print();
  log(`Actual value for x=${testValue}: ${5 * testValue + 2}`);

  model.dispose();
}

/* ============================================================
 * Model 2 — Matrix-based Dense Network
 * ------------------------------------------------------------
 * ম্যানুয়ালি নির্ধারিত W আর b দিয়ে টার্গেট আউটপুট বানাই, তারপর
 * একটা dense layer ট্রেইন করে ওই ম্যাপিং শিখতে দিই।
 *
 *   y₁ = 2x₁ + 3x₂ + 1
 *   y₂ = 5x₁ -  x₂ + 2
 *   y₃ = -x₁ + 4x₂
 *
 * অর্থাৎ W (3×2) আর b (3×1) — dense layer নিজেই এগুলো recover করবে।
 * ============================================================ */

const W = tf.tensor2d([
  [2, 3],
  [5, -1],
  [-1, 4],
]);
const inputs = tf.tensor([
  [1, 1],
  [2, 1],
  [3, 2],
]);
const bias = tf.tensor([
  [1],
  [2],
  [0],
]);

async function generateTargets() {
  // একবারে পুরো batch: Xᵀ (2×3) হিসেবে matMul করলে লুপ লাগে না
  // Y = (W · Xᵀ + b)ᵀ  → shape (3 samples × 3 outputs)
  const targets = W.matMul(inputs.transpose())
    .add(bias) // broadcasting: (3×3) + (3×1)
    .transpose();
  return targets;
}

async function trainMatrixModel(newInput) {
  const targets = await generateTargets();

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 3, inputShape: [2], activation: "linear" }));

  model.compile({ optimizer: tf.train.sgd(0.1), loss: "meanSquaredError" });

  await model.fit(inputs, targets, { epochs: 300, verbose: 0 });
  log("✅ Matrix model trained");

  const predict = model.predict(newInput);
  predict.print();
  log("Expected for [1, 2]: [7, 5, 7]");

  model.dispose();
}

/* ============================================================ */

async function main() {
  await tf.setBackend("wasm");
  await tf.ready();
  log("Backend:", tf.getBackend(), "\n");

  log("--- Model 1: Linear Regression ---");
  await trainLinearModel(2);

  log("\n--- Model 2: Matrix Dense Model ---");
  await trainMatrixModel(tf.tensor([[1, 2]]));
}

main();

