<div align="center">
  <img width="600" src="https://www.gstatic.com/devrel-devsite/prod/v46d043083f27fa7361aea8506dabbd161e0b84f5a7c6df8d5e3cfad447dd4376/tensorflow/images/lockup.svg" alt="TensorFlow.js logo" />
</div>

# 🧠 TensorFlow.js Lab

TensorFlow.js দিয়ে বেসিক টেনসর অপারেশন থেকে শুরু করে লিনিয়ার রিগ্রেশন, মাল্টি-ফিচার প্রেডিকশন এবং Ollama-ভিত্তিক embedding similarity পর্যন্ত হ্যান্ডস-অন এক্সপেরিমেন্টের সংকলন।

**সব কোড Android/Termux (aarch64) পরিবেশের জন্য অপ্টিমাইজড — native `tfjs-node` এর বদলে WASM ব্যাকএন্ড ব্যবহার করা হয়েছে।**

---

## 📂 Project Structure

```
.
├── package.json
├── README.md
└── src/
    ├── tensor-basics.js          # Core tensor operations
    ├── models.js                 # Neural network training
    ├── house-price-prediction.js # Practical regression project
    └── embedding-similarity.js   # Ollama embeddings + vector search
```

| File | Category | কাজ |
|------|----------|-----|
| `tensor-basics.js` | Core Operations | Tensor তৈরি, গাণিতিক অপারেশন, reshape, dtype cast, buffer, normalization, CSV pipeline |
| `models.js` | Models | `y = 5x + 2` শেখা + matrix-ভিত্তিক dense network training |
| `house-price-prediction.js` | Regression | Size, Bedrooms, YearBuilt → Price prediction |
| `embedding-similarity.js` | Embeddings | Ollama embedding, cosine similarity, angle, semantic search demo |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js ≥ 18** (Termux-এ: `pkg install nodejs`)
- **Ollama** (শুধু embedding টেস্টের জন্য)

```bash
# Termux-এ Ollama ইন্সটল
pkg install curl
curl -fsSL https://ollama.com/install.sh | sh

# Embedding model ডাউনলোড
ollama pull embeddinggemma
```

### Installation

```bash
git clone https://github.com/yeasin4745/Tensorflow-js
cd Tensorflow-js
npm install
```

### Usage

```bash
npm run tensor-basics   # Tensor operations playground
npm run models          # Neural network training demo
npm run house-price     # House price prediction

npm run embed           # Embedding demo (built-in examples + semantic search)
npm run embed -- "text 1" "text 2"   # নিজের টেক্সট compare করা
```

**Environment variables** (optional):

| Variable | Default | কাজ |
|----------|---------|-----|
| `OLLAMA_EMBED_URL` | `http://127.0.0.1:11434/api/embed` | Ollama API endpoint |
| `EMBEDDING_MODEL` | `embeddinggemma` | Embedding model name |

---

## 📌 Topic Coverage

### 1️⃣ Tensor Operations (`tensor-basics.js`)

- Tensor তৈরি: `fill`, `eye`, `linspace`, `truncatedNormal`, `variable`
- Element-wise গাণিতিক অপারেশন: `add`, `sub`, `div`, `mul`
- Matrix multiplication: `matMul` (element-wise `mul` নয়)
- Statistics: `mean`, `max`, `sum`
- Shape ops: `reshape`, `stack`, `transpose`, `clone`
- dtype conversion: `cast('bool')`, `oneHot`
- Buffer: `bufferSync`, `tf.buffer` → low-level sync access
- Memory management: `dispose()`, `tf.memory()`

### 2️⃣ Neural Network Models (`models.js`)

- **Simple Linear Regression** — মডেলটি নিজে থেকেই `y = 5x + 2` সম্পর্কটি weight ও bias থেকে recover করে শেখে
- **Matrix-based Dense Model** — ম্যানুয়ালি তৈরি `W·x + b` ম্যাপিং একটি dense layer দিয়ে ট্রেইন করা

### 3️⃣ House Price Prediction (`house-price-prediction.js`)

- `tf.data.csv` দিয়ে remote CSV লোড
- ৩টি feature (Size, Bedrooms, YearBuilt) → ১টি output (Price)
- Training চেক করার জন্য `validationSplit`

### 4️⃣ Embedding Similarity (`embedding-similarity.js`)

- Ollama `/api/embed` endpoint দিয়ে টেক্সট → vector
- Cosine similarity ও angle (θ) গণনা
- Semantic search demo: একটি query-র সাথে সবচেয়ে মিল আছে এমন document খোঁজা

---

## 📐 Formulas

### Cosine Similarity

দুটি vector-এর মধ্যে **কোণের** ভিত্তিতে সম্পর্ক মাপা হয় — magnitude নয়, **দিক (direction)** গুরুত্বপূর্ণ:

$$
\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum A_i^2} \cdot \sqrt{\sum B_i^2}}
$$

Angle বের করতে:

$$
\theta = \arccos(\cos\theta) \times \frac{180}{\pi}
$$

| cos(θ) | θ | অর্থ |
|--------|---|------|
| `1.0` | `0°` | একই দিক — প্রায় একই অর্থ |
| `0.0` | `90°` | Orthogonal — কোনো semantic সম্পর্ক নেই |
| `-1.0` | `180°` | বিপরীত দিক |

### Z-score Normalization

$$
z = \frac{x - \mu}{\sigma}
$$

Feature-গুলোর scale আলাদা হলে (যেমন Size ≈ 2000, Bedrooms ≈ 3) gradient unstable হয় — তাই সব feature-কে mean 0, std 1-এ আনা হয়।

Denormalize (prediction ফিরে পেতে):

$$
y = z \cdot \sigma + \mu
$$

### Dense Layer (Neural Network)

$$
\vec{y} = W \cdot \vec{x} + \vec{b}
$$

`models.js`-এ ব্যবহৃত নির্দিষ্ট সমীকরণ:

$$
y_1 = 2x_1 + 3x_2 + 1
$$
$$
y_2 = 5x_1 - x_2 + 2
$$
$$
y_3 = -x_1 + 4x_2
$$

### Loss Function (Mean Squared Error)

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

## ⚙️ Why WASM Backend?

| Backend | Termux (Android) | ব্যাখ্যা |
|---------|------------------|---------|
| `tfjs-node` | ❌ | Native C++ build দরকার — aarch64/Termux-এ প্রায়ই fail করে |
| `cpu` (pure JS) | ✅ | চলে, কিন্তু ধীর |
| **`wasm`** | ✅ | প্রায় native speed, কোনো compilation লাগে না — **এই প্রজেক্টে ব্যবহৃত** |

---

## 📄 License

MIT © [MD Yeasin Ali](https://github.com/yeasin4745)
