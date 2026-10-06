# TensorFlow.js — হাতে-কলমে JavaScript উদাহরণ

এই repository-তে Node.js দিয়ে TensorFlow.js শেখার কয়েকটি ছোট, runnable example আছে। Tensor operation, linear regression, multi-output model এবং Ollama embedding দিয়ে text similarity—প্রতিটি example আলাদা JavaScript file-এ রাখা হয়েছে, যেন একেকটি বিষয় আলাদাভাবে চালানো ও বোঝা যায়।

> **প্রয়োজনীয়তা:** Node.js 18 বা পরবর্তী version এবং npm। `index.js` চালাতে Ollama-ও চালু থাকতে হবে। House-price example চালাতে internet connection প্রয়োজন, কারণ এটি GitHub থেকে CSV dataset নেয়।

## Installation

```bash
git clone https://github.com/yeasin4745/Tensorflow-js.git
cd Tensorflow-js
npm install
```

## কোন ফাইলে কী আছে

| File | কাজ |
| --- | --- |
| `index.js` | Ollama থেকে দুইটি text-এর embedding নিয়ে TensorFlow.js-এ cosine similarity ও angle গণনা |
| `tf.js` | Tensor operations এবং `y = 5x + 2` শেখার linear regression demo |
| `model.js` | matrix-নির্ধারিত তিনটি equation থেকে multi-output regression model train ও predict |
| `house-price-predction.js` | CSV-এর size, bedroom ও build year feature দিয়ে house-price regression demo |
| `tests/index.test.js` | embedding validation ও cosine similarity-র offline unit tests |

## Examples চালানো

### 1. Tensor operations ও linear regression

```bash
node tf.js
node tf.js 3.5
```

প্রথম command tensor addition, subtraction, division, matrix multiplication, statistics, reshape ও identity matrix দেখায়। এরপর ছোট model train করে `y = 5x + 2`-এর জন্য default `x = 2` predict করে। দ্বিতীয় command-এ নিজের input দেওয়া যায়।

### 2. Multi-output model

```bash
node model.js
node model.js 2 3
```

Model-টি নিচের mapping শিখতে চেষ্টা করে:

- `y₁ = 2x₁ + 3x₂ + 1`
- `y₂ = 5x₁ − x₂ + 2`
- `y₃ = −x₁ + 4x₂`

### 3. House-price regression

```bash
node house-price-predction.js
node house-price-predction.js 1200 3 2015
```

ইনপুটের ক্রম হলো `size_sqft bedrooms year_built`। Default dataset URL হলো [`housing_market_data.csv`](https://github.com/yeasin4745/csv-datasets/blob/main/datasets/regression/housing_market_data.csv)। বিকল্প CSV দিতে চাইলে `HOUSING_CSV_URL` environment variable ব্যবহার করুন। CSV-তে `Size_sqft`, `Bedrooms`, `Year_Built`, এবং `Price` numeric column থাকতে হবে।

```bash
HOUSING_CSV_URL="https://example.com/housing.csv" node house-price-predction.js 1200 3 2015
```

এটি শেখার demo—বাস্তব property valuation-এর জন্য trained বা validated model নয়। Prediction-এর মান dataset ও training run অনুযায়ী বদলাতে পারে।

### 4. Ollama দিয়ে text similarity

প্রথমে [Ollama](https://ollama.com/) install ও চালু করে embedding model নামান:

```bash
ollama pull embeddinggemma
ollama serve
```

তারপর অন্য terminal-এ project directory থেকে চালান:

```bash
node index.js "I enjoy learning JavaScript" "I like programming in JavaScript"
```

`index.js` Ollama-র `/api/embed` API-তে দুইটি text পাঠায় এবং cosine similarity (−1 থেকে 1) ও angle দেখায়। সাধারণভাবে similarity যত বেশি, vector দুটি তত কাছাকাছি; নির্দিষ্ট threshold ব্যবহার করার আগে আপনার model ও use case দিয়ে যাচাই করুন।

ঐচ্ছিক configuration:

```bash
OLLAMA_EMBED_URL="http://127.0.0.1:11434/api/embed" \
EMBEDDING_MODEL="embeddinggemma" \
node index.js "first text" "second text"
```

`OLLAMA_EMBED_URL` এবং `EMBEDDING_MODEL`-এর default যথাক্রমে `http://127.0.0.1:11434/api/embed` ও `embeddinggemma`।

## Tests ও code check

```bash
npm test
npm run check
```

`npm test` Node.js-এর built-in test runner ব্যবহার করে; এই test চালাতে Ollama বা live network লাগে না।

## কীভাবে cosine similarity হিসাব হয়

দুই embedding vector `A` ও `B`-এর জন্য:

```text
cosineSimilarity = (A · B) / (||A|| × ||B||)
angleDegrees     = acos(cosineSimilarity) × 180 / π
```

`index.js` vector length, finite numeric values, Ollama response dimensions এবং zero vector পরীক্ষা করে। TensorFlow tensor-গুলো `tf.tidy()` দিয়ে scope শেষে dispose হয়। অন্যান্য training script-ও input, prediction tensor এবং model শেষ হলে dispose করে, যাতে অপ্রয়োজনীয় memory জমে না থাকে।

## Common issues

- **`fetch failed` / Ollama connection refused:** `ollama serve` চলছে কি না এবং `OLLAMA_EMBED_URL` ঠিক কি না যাচাই করুন।
- **Model not found:** `ollama pull embeddinggemma` চালান, অথবা `EMBEDDING_MODEL`-এ local Ollama model-এর নাম দিন।
- **CSV download error:** dataset URL reachable কি না দেখুন; দরকার হলে `HOUSING_CSV_URL` দিন। CSV header README-তে উল্লেখ করা নামগুলোর সঙ্গে মিলতে হবে।
- **`node: command not found`:** Node.js 18+ install করে terminal restart করুন।
- **Similarity-তে zero-vector error:** Ollama-র returned embedding ফাঁকা বা সব শূন্য হলে cosine similarity নির্ধারিত নয়; model/API response যাচাই করুন।

## Learning note

এই repository-র model-গুলো ছোট educational demo। Production application-এ dataset split, validation metrics, reproducible seeds, model persistence, input/data quality checks এবং বাস্তব use case-এ performance evaluation যুক্ত করুন।