'use strict';

const tf = require('@tensorflow/tfjs');

function runTensorOperations() {
  console.log('\n--- Tensor operations ---');
  tf.tidy(() => {
    const a = tf.fill([2, 2], 10);
    const b = tf.fill([2, 2], 5);
    const sample = tf.tensor1d([4, 5, 6, 4]);

    console.log('Addition:');
    tf.add(a, b).print();
    console.log('Subtraction:');
    tf.sub(a, b).print();
    console.log('Element-wise division:');
    tf.div(a, b).print();
    console.log('Matrix multiplication:');
    tf.matMul(a, b).print();
    console.log('Mean / max / sum:');
    tf.stack([tf.mean(sample), tf.max(sample), tf.sum(sample)]).print();
    console.log('Reshape to 2 × 2:');
    sample.reshape([2, 2]).print();
    console.log('Identity matrix:');
    tf.eye(3).print();
  });
}

async function trainLinearRegression(value = 2) {
  if (!Number.isFinite(value)) {
    throw new TypeError('Prediction input must be a finite number.');
  }

  const values = Array.from({ length: 21 }, (_, index) => index - 10);
  const xTrain = tf.tensor2d(values, [values.length, 1]);
  const yTrain = tf.tensor2d(values.map((x) => 5 * x + 2), [values.length, 1]);
  const model = tf.sequential();

  try {
    model.add(tf.layers.dense({
      units: 1,
      inputShape: [1],
      kernelInitializer: 'zeros',
      biasInitializer: 'zeros',
    }));
    model.compile({ optimizer: tf.train.adam(0.05), loss: 'meanSquaredError' });
    await model.fit(xTrain, yTrain, { epochs: 250, verbose: 0 });

    const input = tf.tensor2d([[value]]);
    const prediction = model.predict(input);
    try {
      const [predicted] = await prediction.data();
      console.log(`\nLearned y = 5x + 2; prediction for x=${value}: ${predicted.toFixed(3)}`);
      return predicted;
    } finally {
      input.dispose();
      prediction.dispose();
    }
  } finally {
    xTrain.dispose();
    yTrain.dispose();
    model.dispose();
  }
}

async function main() {
  await tf.ready();
  console.log(`TensorFlow.js backend: ${tf.getBackend()}`);
  runTensorOperations();

  const value = process.argv[2] === undefined ? 2 : Number(process.argv[2]);
  await trainLinearRegression(value);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { runTensorOperations, trainLinearRegression };
