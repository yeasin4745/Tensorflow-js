/**
 * ============================================================
 *  tensor-basics.js
 *  ------------------------------------------------------------
 *  Category : Core Tensor Operations
 *  Purpose  : TensorFlow.js-এর বেসিক থেকে অ্যাডভান্সড টেনসর
 *             অপারেশনগুলোর হ্যান্ডস-অন রেফারেন্স — tensor তৈরি,
 *             গাণিতিক অপারেশন, রিশেপ, dtype কাস্ট, buffer এক্সেস,
 *             normalization এবং CSV ডেটা পাইপলাইন।
 *
 *  Run      : npm run tensor-basics
 *  Backend  : WASM (Termux/Android-বান্ধব)
 * ============================================================
 */

import tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";

const { log } = console;

function print(...datas) {
  datas.forEach((d) => d.print());
}

/* ---------- 1. Tensor Creation ---------- */
const a = tf.fill([4, 4], 10);
const b = tf.fill([4, 4], 5);
const identity = tf.eye(5, 5);
const linspace = tf.linspace(9, 7, 10);
const random = tf.truncatedNormal([8, 5], 6, 2); // নরমাল ডিস্ট্রিবিউশনের চেয়ে efficient
const variable = tf.variable(tf.randomNormal([2, 2], 5, 3));

// variable-এ নতুন ভ্যালু assign করা যায়
// variable.assign(random);

/* ---------- 2. Element-wise Math ---------- */
const add = tf.add(a, b);
const sub = tf.sub(a, b);
const div = tf.div(a, b);
const scalarMul = tf.mul(a, tf.scalar(5));

// ম্যাট্রিক্স মাল্টিপ্লিকেশন (element-wise নয়!)
const matMul = tf.matMul(a, b);

// A² - 5A + 6I  (matrix polynomial)
const A = tf.tensor([
  [4, 2],
  [3, 5],
]);
const I = tf.eye(2, 2);
const poly = tf
  .matMul(A, A)
  .sub(A.mul(tf.scalar(5)))
  .add(I.mul(tf.scalar(6)));

// activation: sigmoid
const sigmoid = A.sigmoid();

/* ---------- 3. Reduction / Statistics ---------- */
const t1d = tf.tensor1d([4, 5, 6, 4]);
const mean = tf.mean(t1d);
const max = tf.max(t1d);
const sum = tf.sum(t1d);

/* ---------- 4. Shape Ops ---------- */
const reshaped = t1d.reshape([2, 2]); // 1D -> 2D
const stacked = tf.stack([a, b, tf.fill([4, 4], 1)]); // একই shape-এর tensor জোড়া
const cloned = A.clone();
const transposed = cloned.transpose();

/* ---------- 5. dtype Conversion ---------- */
const asBool = A.cast("bool");
const oneHot = tf.oneHot(tf.tensor([2, 1, 0], [3], "int32"), 3);

/* ---------- 6. Buffer (low-level, sync access) ---------- */
const buf = tf.tensor([
  [6, -8, -1],
  [-2, 8, 1],
  [8, 4, 1],
]).bufferSync();
// log(buf.get(1, 2));     // নির্দিষ্ট সেল পড়া
// log(buf.values);         // flat array

const writable = tf.buffer([2, 2]);
writable.set(55, 0, 0);
const fromBuffer = writable.toTensor();

/* ---------- 7. Normalization ---------- */

// Z-score: z = (x - mean) / std
function zScore(arr) {
  const t = tf.tensor1d(arr);
  const mean = t.mean().arraySync();
  const std = tf.sqrt(t.sub(mean).square().mean()).arraySync();
  return arr.map((n) => (n - mean) / std);
}

// Min-Max: x' = (x - min) / (max - min)
function minMax(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  return arr.map((n) => (n - min) / (max - min));
}

/* ---------- 8. CSV Data Pipeline (tf.data) ---------- */

async function loadCsvDataset() {
  const csvUrl =
    "https://raw.githubusercontent.com/yeasin4745/csv-datasets/refs/heads/main/data.csv";

  const dataset = tf.data.csv(csvUrl, {
    hasHeader: true,
    columnConfigs: { Weight: { isLabel: true }, Height: {} },
    configuredColumnsOnly: true,
  });

  // batch করে পড়লে মেমরি-efficient — বড় ডেটাসেটে sequential লোড হয়
  const batched = dataset.batch(2);
  await batched.forEachAsync((batch) => {
    log(batch.xs, "=>", batch.ys);
  });
}

/* ---------- Demo Output ---------- */

async function main() {
  await tf.setBackend("wasm");
  await tf.ready();
  log("Backend:", tf.getBackend());

  print(add, sub, div, scalarMul, matMul, poly, sigmoid);
  print(mean, max, sum, reshaped, identity, stacked, transposed, asBool, oneHot, fromBuffer);

  log("zScore([5,2,4,1,7]):", zScore([5, 2, 4, 1, 7]));
  log("minMax([5,2,4,1,7]):", minMax([5, 2, 4, 1, 7]));

  // ডেটাসেট দরকার হলে আনকমেন্ট করো:
  // await loadCsvDataset();

  tf.dispose([add, sub, div, scalarMul, matMul, poly, sigmoid, mean, max, sum]);
  log("Memory tensors (after dispose):", tf.memory().numTensors);
}

main();

