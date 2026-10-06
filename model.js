'use strict';

const tf = require('@tensorflow/tfjs');

const WEIGHTS = [
  [2, 3],
  [5, -1],
  [-1, 4],
];
const BIAS = [1, 2, 0];

function parseInput(args) {
  const input = args.length === 0 ? [1, 2] : args.map(Number);
  if (input.length !== 2 || input.some((value) => !Number.isFinite(value))) {
    throw new TypeError('Usage: node model.js [x1 x2] (both values must be finite numbers)');
  }
  return input;
}

async function trainAndPredict(newInput = [1, 2]) {
  const inputValues = parseInput(newInput.map(String));
  const trainingInputs = [];
  const trainingOutputs = [];
  for (let x1 = -5; x1 <= 5; x1 += 1) {
    for (let x2 = -5; x2 <= 5; x2 += 1) {
      trainingInputs.push([x1, x2]);
      trainingOutputs.push(
        WEIGHTS.map(([weight1, weight2], index) =>
          weight1 * x1 + weight2 * x2 + BIAS[index],
        ),
      );
    }
  }

  const xTrain = tf.tensor2d(trainingInputs);
  const yTrain = tf.tensor2d(trainingOutputs);
  const model = tf.sequential();

  try {
    model.add(tf.layers.dense({ units: 3, inputShape: [2], activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    await model.fit(xTrain, yTrain, { epochs: 1000, verbose: 0 });

    const inputTensor = tf.tensor2d([inputValues]);
    const prediction = model.predict(inputTensor);
    try {
      const output = Array.from(await prediction.data());
      console.log(`Input [${inputValues.join(', ')}] → predicted [${output.map((n) => n.toFixed(3)).join(', ')}]`);
      console.log('Expected from the example equations: [2x₁ + 3x₂ + 1, 5x₁ − x₂ + 2, −x₁ + 4x₂]');
      return output;
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
  trainAndPredict(process.argv.slice(2)).catch((error) => {
    console.error(`Failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { parseInput, trainAndPredict };
