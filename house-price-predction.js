const tf = require('@tensorflow/tfjs-node'); // Node.js এর জন্য tfjs-node ব্যবহার করা ভালো
const { log } = console;

/**
 * 1. GitHub থেকে CSV ডেটাসেট লোড করে।
 */
async function loadDataSet() {
    
    const csvUrl = 'https://raw.githubusercontent.com/yeasin4745/csv-datasets/main/HousingMarketData.csv';

    const dataset = tf.data.csv(csvUrl, {
        hasHeader: true,
        columnConfigs: {
            Price: { isLabel: true }
        }
    });

    const size = [], bedrooms = [], year = [], price = [];

    await dataset.forEachAsync(row => {
        size.push(parseFloat(row.Size));
        bedrooms.push(parseFloat(row.Bedrooms));
        year.push(parseFloat(row.YearBuilt));
        price.push(parseFloat(row.Price));
    });

    return { size, bedrooms, year, price };
}

/**
 * 2. ডেটা নরম্যালাইজ করে এবং গড় (mean) ও স্ট্যাডার্ড ডেভিয়েশন (std) রিটার্ন করে।
 */
function getNorm(array) {
    const tensor = tf.tensor1d(array);
    const mean = tensor.mean().arraySync();
    const variance = tensor.sub(mean).square().mean().arraySync();
    const std = Math.sqrt(variance);
    const norm = array.map(n => (n - mean) / std);
    return { norm, mean, std }; // mean এবং std রিটার্ন করা হয়েছে Prediction এর জন্য
}

/**
 * 3. একটি সিম্পল লিনিয়ার রিগ্রেশন মডেল তৈরি করে।
 */
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [3], // ইনপুট ফিচার সংখ্যা: Size, Bedrooms, YearBuilt
        units: 1,
        activation: 'linear'
    }));
    return model;
}

/**
 * 4. Main Function: ডেটা লোড, প্রসেস, মডেল ট্রেইন এবং প্রেডিক্ট করার জন্য।
 */
(async function main(inputRaw) {
    const data = await loadDataSet();
    const { size, bedrooms, year, price } = data;

    // প্রতিটি ফিচারের জন্য ডেটা নরম্যালাইজ করা
    const sizeStats = getNorm(size);
    const roomStats = getNorm(bedrooms);
    const yearStats = getNorm(year);
    const priceStats = getNorm(price);

    const inputs = [];
    const labels = [];

    // মডেলের জন্য ইনপুট এবং লেবেল প্রস্তুত করা
    for (let i = 0; i < size.length; i++) {
        inputs.push([
            sizeStats.norm[i],
            roomStats.norm[i],
            yearStats.norm[i]
        ]);
        labels.push(priceStats.norm[i]);
    }

    const xTrain = tf.tensor2d(inputs);
    const yTrain = tf.tensor2d(labels, [labels.length, 1]);

    const model = createModel();

    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(0.001),
        metrics: ['mse']
    });

    log('Training started...');
    await model.fit(xTrain, yTrain, {
        epochs: 100,
        batchSize: 4,
        validationSplit: 0.2, // ডেটাসেটের ২০% ভ্যালিডেশনের জন্য ব্যবহার করা হবে
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
            }
        }
    });

    log('\n✅ Training finished!\n');

    // নতুন বাড়ির দাম প্রেডিক্ট করা
    const inputNorm = [
        (inputRaw[0] - sizeStats.mean) / sizeStats.std,
        (inputRaw[1] - roomStats.mean) / roomStats.std,
        (inputRaw[2] - yearStats.mean) / yearStats.std
    ];

    const inputTensor = tf.tensor2d([inputNorm]);
    const predictionNorm = model.predict(inputTensor);
    const predNormValue = (await predictionNorm.data())[0];

    // নরম্যালাইজড প্রেডিকশনকে আসল দামে রূপান্তর করা
    const predictedPrice = predNormValue * priceStats.std + priceStats.mean;

    log(`Predicted Price for ${inputRaw[0]} sqft, ${inputRaw[1]} beds, built ${inputRaw[2]}: ৳${Math.round(predictedPrice)}`);

})([1020, 4, 2019]); // এখানে ইনপুট দিয়ে টেস্ট করুন: [Size, Bedrooms, YearBuilt]
