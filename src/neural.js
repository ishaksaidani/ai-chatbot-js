const fs = require('fs');
const path = require('path');

const MODEL_FILE = path.join(__dirname, '..', 'data', 'neural-model.json');

/**
 * Lightweight pure-JS neural network for intent classification and pattern learning.
 * Inspired by Brain.js but zero native dependencies.
 * Uses a simple feedforward network with backpropagation.
 */
class NeuralNetwork {
  constructor(options = {}) {
    this.learningRate = options.learningRate || 0.3;
    this.iterations = options.iterations || 2000;
    this.errorThresh = options.errorThresh || 0.005;
    this.hiddenSize = options.hiddenSize || 20;
    this.vocab = [];
    this.labels = [];
    this.weightsIH = null; // input -> hidden
    this.weightsHO = null; // hidden -> output
    this.biasH = null;
    this.biasO = null;
    this.trained = false;
    this.loadModel();
  }

  // --- Activation functions ---

  sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  sigmoidDerivative(x) {
    return x * (1 - x);
  }

  // --- Matrix helpers ---

  createMatrix(rows, cols, fillFn) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (fillFn ? fillFn() : 0))
    );
  }

  randomWeight() {
    return (Math.random() - 0.5) * 2;
  }

  // --- Text to vector ---

  buildVocab(trainingData) {
    const wordSet = new Set();
    const labelSet = new Set();
    for (const item of trainingData) {
      const words = this.tokenize(item.input);
      words.forEach((w) => wordSet.add(w));
      labelSet.add(item.output);
    }
    this.vocab = Array.from(wordSet);
    this.labels = Array.from(labelSet);
  }

  textToVector(text) {
    const words = this.tokenize(text);
    return this.vocab.map((v) => (words.includes(v) ? 1 : 0));
  }

  labelToVector(label) {
    return this.labels.map((l) => (l === label ? 1 : 0));
  }

  vectorToLabel(output) {
    let maxIdx = 0;
    let maxVal = output[0];
    for (let i = 1; i < output.length; i++) {
      if (output[i] > maxVal) {
        maxVal = output[i];
        maxIdx = i;
      }
    }
    return { label: this.labels[maxIdx], confidence: maxVal };
  }

  tokenize(text) {
    return (text || '')
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter((w) => w.length > 1);
  }

  // --- Forward pass ---

  forward(inputVec) {
    const inputSize = inputVec.length;
    const hiddenSize = this.biasH.length;
    const outputSize = this.biasO.length;

    // Input -> Hidden
    const hidden = new Array(hiddenSize);
    for (let h = 0; h < hiddenSize; h++) {
      let sum = this.biasH[h];
      for (let i = 0; i < inputSize; i++) {
        sum += inputVec[i] * this.weightsIH[i][h];
      }
      hidden[h] = this.sigmoid(sum);
    }

    // Hidden -> Output
    const output = new Array(outputSize);
    for (let o = 0; o < outputSize; o++) {
      let sum = this.biasO[o];
      for (let h = 0; h < hiddenSize; h++) {
        sum += hidden[h] * this.weightsHO[h][o];
      }
      output[o] = this.sigmoid(sum);
    }

    return { hidden, output };
  }

  // --- Training ---

  train(trainingData) {
    if (!trainingData || trainingData.length === 0) return;

    this.buildVocab(trainingData);

    const inputSize = this.vocab.length;
    const hiddenSize = Math.min(this.hiddenSize, Math.max(5, Math.floor(inputSize / 2)));
    const outputSize = this.labels.length;

    if (inputSize === 0 || outputSize === 0) return;

    // Initialize weights
    this.weightsIH = this.createMatrix(inputSize, hiddenSize, () => this.randomWeight());
    this.weightsHO = this.createMatrix(hiddenSize, outputSize, () => this.randomWeight());
    this.biasH = Array.from({ length: hiddenSize }, () => this.randomWeight());
    this.biasO = Array.from({ length: outputSize }, () => this.randomWeight());

    // Prepare training vectors
    const samples = trainingData.map((item) => ({
      input: this.textToVector(item.input),
      target: this.labelToVector(item.output),
    }));

    // Train with backpropagation
    for (let iter = 0; iter < this.iterations; iter++) {
      let totalError = 0;

      for (const sample of samples) {
        const { hidden, output } = this.forward(sample.input);

        // Calculate output errors
        const outputErrors = new Array(outputSize);
        for (let o = 0; o < outputSize; o++) {
          const error = sample.target[o] - output[o];
          outputErrors[o] = error * this.sigmoidDerivative(output[o]);
          totalError += error * error;
        }

        // Calculate hidden errors
        const hiddenErrors = new Array(hiddenSize);
        for (let h = 0; h < hiddenSize; h++) {
          let error = 0;
          for (let o = 0; o < outputSize; o++) {
            error += outputErrors[o] * this.weightsHO[h][o];
          }
          hiddenErrors[h] = error * this.sigmoidDerivative(hidden[h]);
        }

        // Update hidden->output weights
        for (let h = 0; h < hiddenSize; h++) {
          for (let o = 0; o < outputSize; o++) {
            this.weightsHO[h][o] += this.learningRate * outputErrors[o] * hidden[h];
          }
        }
        for (let o = 0; o < outputSize; o++) {
          this.biasO[o] += this.learningRate * outputErrors[o];
        }

        // Update input->hidden weights
        for (let i = 0; i < inputSize; i++) {
          for (let h = 0; h < hiddenSize; h++) {
            this.weightsIH[i][h] += this.learningRate * hiddenErrors[h] * sample.input[i];
          }
        }
        for (let h = 0; h < hiddenSize; h++) {
          this.biasH[h] += this.learningRate * hiddenErrors[h];
        }
      }

      const avgError = totalError / samples.length;
      if (avgError < this.errorThresh) {
        break;
      }
    }

    this.trained = true;
    this.saveModel();
  }

  // --- Prediction ---

  predict(text) {
    if (!this.trained || !this.weightsIH) {
      return { label: 'unknown', confidence: 0 };
    }

    const inputVec = this.textToVector(text);

    // Check if input has any known words
    if (inputVec.every((v) => v === 0)) {
      return { label: 'unknown', confidence: 0 };
    }

    const { output } = this.forward(inputVec);
    return this.vectorToLabel(output);
  }

  // --- Incremental learning ---

  addTrainingExample(input, output, existingData) {
    const newData = [...existingData, { input, output }];
    this.train(newData);
    return newData;
  }

  // --- Model persistence ---

  saveModel() {
    const model = {
      vocab: this.vocab,
      labels: this.labels,
      weightsIH: this.weightsIH,
      weightsHO: this.weightsHO,
      biasH: this.biasH,
      biasO: this.biasO,
      trained: this.trained,
    };
    try {
      fs.writeFileSync(MODEL_FILE, JSON.stringify(model));
    } catch (err) {
      console.error('Failed to save model:', err.message);
    }
  }

  loadModel() {
    try {
      if (fs.existsSync(MODEL_FILE)) {
        const model = JSON.parse(fs.readFileSync(MODEL_FILE, 'utf-8'));
        this.vocab = model.vocab || [];
        this.labels = model.labels || [];
        this.weightsIH = model.weightsIH;
        this.weightsHO = model.weightsHO;
        this.biasH = model.biasH;
        this.biasO = model.biasO;
        this.trained = model.trained || false;
      }
    } catch (err) {
      console.error('Failed to load model:', err.message);
    }
  }
}

module.exports = NeuralNetwork;
