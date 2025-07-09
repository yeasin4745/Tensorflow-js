let tf = require('@tensorflow/tfjs');

// Manual matrix operation to generate output
let w = tf.tensor2d([[2, 3], [5, -1], [-1, 4]]);
let input = tf.tensor([[1, 1], [2, 1], [3, 2]]);
let bias = tf.tensor([[1], [2], [0]]);

async function run() {
  let inputArray = await input.array();
  let output = [];

  for (let i of inputArray) {
    let data = tf.tensor(i).reshape([2, 1]);
    let result = w.matMul(data).add(bias).reshape([1, 3]);
    let array = await result.array();
    output.push(array[0]);

    data.dispose();
    result.dispose();
  }

  return tf.tensor(output);
}

(async function (newInput) {
  let output = await run();

  let model = tf.sequential();
  model.add(tf.layers.dense({
    units: 3,
    inputShape: [2],
    activation: 'linear'
  }));

  model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError'
  });

  await model.fit(input, output, {
    epochs: 300
  });

  let predict = model.predict(newInput);
  predict.print();

})(tf.tensor([[1, 2]]));
