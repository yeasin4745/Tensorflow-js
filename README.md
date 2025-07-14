<div>
  <img wiidth=600 height=500  src="https://www.gstatic.com/devrel-devsite/prod/v46d043083f27fa7361aea8506dabbd161e0b84f5a7c6df8d5e3cfad447dd4376/tensorflow/images/lockup.svg" alt="logo" />
</div>




# TensorFlow.js বেসিক  TensorFlow.js এর বেসিক টপিক নিয়ে বিস্তারিত আলোচনা করা হয়েছে। এখানে টেনসর অপারেশন থেকে শুরু করে একটি সহজ মডেল তৈরির উদাহরণ পর্যন্ত দেখানো হয়েছে।

---

## 📌 **টেনসর অপারেশন (Tensor Operation)**

- দুটি ম্যাট্রিক্সের উপর অ্যাড, সাবস্ট্রাকশন, ডিভিশন এবং মাল্টিপ্লিকেশন করা।
- স্কেলার ভ্যালু দিয়ে মাল্টিপ্লিকেশন।
- ম্যাট্রিক্স মাল্টিপ্লিকেশন (Matrix Multiplication)।
- টেনসর এর বিভিন্ন পরিসংখ্যানিক ফাংশন ব্যবহার যেমন `mean`, `max`, `sum`।
- টেনসর রিশেপ (Reshape) করা — 1D থেকে 2D।
- টেনসরসরকে অ্যারে বা ডেটাতে রূপান্তর করা।

---

## 🔄 **ডেটা টাইপ রূপান্তর ও ম্যাট্রিক্স অপারেশন**

- `cast()` ফাংশনের মাধ্যমে ফাংশনের মাধ্যমে ডেট টাইইপরিবর্তন (যেমন booleann)।
- টেনসর ক্লোন করা এবং ট্রান্সপোজnspose) করা।
- আইডেন্টিটি ম্যাট্রিক্স তৈরি (`eye()` ফাংশন)।
- একাধিক টেনসর একত্রে স্ট্যাক করা (`stack()` ফাংশন)।

---

# Simple Linear Regression Model with TensorFlow.js 🎯

## 📜 Description

এই প্রজেক্টে TensorFlow.js ব্যবহার করে একটি সিম্পল লিনিয়ার রিগ্রেশন মডেল তৈরি করা হয়েছে। মডেলটি `y = 5x + 2` সমীকরণ শেখার জন্য ট্রেন করা হয়েছে।

## 📊 Dataset

র্যান্ডমলি ডেটা তৈরি করা হয়েছে:

- **X:** 30টি র‍্যান্ডম সংখ্যা (Normal Distribution থেকে)  
- **Y:** সমীকরণ অনুযায়ী তৈরি → `y = 5x + 2`

```javascript
let x = tf.truncatedNormal([30], 0, 10, 'int32').reshape([30,1]);
let y = tf.mul(x, 5).add(2);
```


# [model.js](https://github.com/yeasin4745/Tensorflow-js/blob/main/model.js)



# 📐 ম্যাট্রিক্স ভিত্তিক কাস্টম নিউরাল নেটওয়ার্ক মডেল

এই প্রজেক্টে একটি ম্যানুয়ালি নির্ধারিত ম্যাট্রিক্স ও বাইয়াস ব্যবহার করে আউটপুট ডেটা তৈরি করা হয়েছে এবং তা ব্যবহার করে একটি TensorFlow.js মডেল প্রশিক্ষণ (training) করা হয়েছে।

---

## 🎯 সমীকরণ

আমরা নিচের সমীকরণ অনুযায়ী আউটপুট গণনা করেছি:
> ### 🧠 Equation Reference:
> $$
> y_1 = 2x_1 + 3x_2 + 1
> $$
> $$
> y_2 = 5x_1 - x_2 + 2
> $$
> $$
> y_3 = -x_1 + 4x_2
> $$
## ⚙️ প্রযুক্তি

- 📦 TensorFlow.js
- 🧠 Dense Neural Network (1 layer)
- 🖥️ Node.js (Runtime)

---

## 📄 model.js

নিচের `model.js` ফাইলে আমাদের পুরো কাজটি করা হয়েছে:

### 📌 `model.js` ফাইলে যা আছে:

- ইনপুট ডে টা → `x₁, x₂`
- ম্যাট্রিক্স অপারেশন করে ম্যানুয়ালি আউটপুট তৈরি (`y₁, y₂, y₃`)
- একটি Sequential মডেল তৈরি
- মডেল ট্রেইন করে নতুন ইনপুটের জন্য প্রেডিকশন

---

### 🔢 কোড:

```js
let tf = require('@tensorflow/tfjs');

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

// training and prediction 
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
