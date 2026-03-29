const MemorySystem = require('./memory');
const IntelligenceLayer = require('./intelligence');
const NeuralNetwork = require('./neural');
const fs = require('fs');
const path = require('path');

const TRAINING_DATA_FILE = path.join(__dirname, '..', 'data', 'training-data.json');
const INTERACTION_LOG_FILE = path.join(__dirname, '..', 'data', 'interactions.json');

/**
 * Chat Engine — the main pipeline:
 * Input → Pattern Match → Memory Search → Neural Classification → Response Generation → Store → Output
 */
class ChatEngine {
  constructor() {
    this.memory = new MemorySystem();
    this.intelligence = new IntelligenceLayer();
    this.neural = new NeuralNetwork({ hiddenSize: 15, iterations: 500, errorThresh: 0.01 });
    this.initialized = false;

    this.initialize();
  }

  initialize() {
    // Skip training if model already loaded from disk
    if (this.neural.trained) {
      console.log('Neural network loaded from saved model — skipping training.');
      this.initialized = true;
      return;
    }

    // Train neural network on intent data
    const trainingData = this.intelligence.trainingData;
    if (trainingData.intents && trainingData.intents.length > 0) {
      const neuralTraining = [];
      for (const intent of trainingData.intents) {
        for (const pattern of intent.patterns) {
          neuralTraining.push({ input: pattern, output: intent.tag });
        }
      }
      console.log(`Training neural network on ${neuralTraining.length} patterns...`);
      this.neural.train(neuralTraining);
      console.log('Neural network trained successfully!');
    }
    this.initialized = true;
  }

  /**
   * Main chat processing pipeline
   */
  async processMessage(userInput) {
    const startTime = Date.now();
    const debugInfo = {
      input: userInput,
      steps: [],
      memoryHits: [],
      reasoning: null,
      neuralPrediction: null,
      strategyUsed: null,
      responseTime: 0,
    };

    // Step 1: Normalize input
    const normalizedInput = userInput.trim();
    debugInfo.steps.push('Input normalized');

    // Step 2: Check for user name extraction
    this.extractUserInfo(normalizedInput);

    // Step 3: Search memory for relevant past interactions
    const memoryHits = this.memory.searchLongTerm(normalizedInput);
    debugInfo.memoryHits = memoryHits;
    debugInfo.steps.push(`Found ${memoryHits.length} memory matches`);

    // Step 4: Search facts
    const factHits = this.memory.searchFacts(normalizedInput);
    debugInfo.steps.push(`Found ${factHits.length} relevant facts`);

    // Step 5: Run intelligence analysis (pattern matching + intent matching)
    const reasoning = this.intelligence.analyze(normalizedInput, memoryHits);
    debugInfo.reasoning = reasoning;
    debugInfo.steps.push('Intelligence analysis complete');

    // Step 6: Neural network prediction
    const neuralPrediction = this.neural.predict(normalizedInput);
    debugInfo.neuralPrediction = neuralPrediction;
    debugInfo.steps.push(`Neural prediction: ${neuralPrediction.label} (${(neuralPrediction.confidence * 100).toFixed(1)}%)`);

    // Step 7: Generate response
    const response = this.generateResponse(reasoning, neuralPrediction, memoryHits, factHits, normalizedInput);
    debugInfo.strategyUsed = response.strategy;
    debugInfo.steps.push(`Response generated via: ${response.strategy}`);

    // Step 8: Analyze sentiment
    const sentiment = this.intelligence.analyzeSentiment(normalizedInput);
    debugInfo.sentiment = sentiment;

    // Step 9: Store interaction in memory
    this.storeInteraction(normalizedInput, response.text, reasoning, sentiment);
    debugInfo.steps.push('Interaction stored in memory');

    // Step 10: Handle feedback for self-improvement
    if (reasoning.patternMatch && (reasoning.patternMatch.intent === 'feedback_positive' || reasoning.patternMatch.intent === 'feedback_negative')) {
      this.handleFeedback(reasoning.patternMatch.intent, normalizedInput);
      debugInfo.steps.push('Feedback processed for learning');
    }

    debugInfo.responseTime = Date.now() - startTime;

    return {
      response: response.text,
      debug: debugInfo,
    };
  }

  /**
   * Generate response based on combined analysis
   */
  generateResponse(reasoning, neuralPrediction, memoryHits, factHits, input) {
    const strategy = reasoning.strategy;

    // Strategy: Pattern match (highest priority for high-confidence matches)
    if (strategy.type === 'pattern') {
      const intent = strategy.source.intent;

      // Special handlers
      if (intent === 'math' && strategy.source.data) {
        const { a, op, b, result } = strategy.source.data;
        return {
          text: `${a} ${op} ${b} = ${result}`,
          strategy: 'pattern:math',
        };
      }

      if (intent === 'time') {
        const now = new Date();
        return {
          text: `The current date and time is: ${now.toLocaleString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })}`,
          strategy: 'pattern:time',
        };
      }

      // Look up response from training data
      const intentData = this.findIntentData(intent);
      if (intentData) {
        const response = this.pickResponse(intentData.responses, input);
        return { text: response, strategy: `pattern:${intent}` };
      }
    }

    // Strategy: Intent match from training data
    if (strategy.type === 'intent' && strategy.source) {
      const response = this.pickResponse(strategy.source.responses, input);
      return { text: response, strategy: `intent:${strategy.source.tag}` };
    }

    // Strategy: Memory-based response
    if (strategy.type === 'memory' && strategy.source) {
      const memEntry = strategy.source;
      return {
        text: `Based on our previous conversation, I recall discussing something similar. ${memEntry.output || "Let me think about this differently..."}`,
        strategy: 'memory',
      };
    }

    // Neural network as backup classifier
    if (neuralPrediction.confidence > 0.6 && neuralPrediction.label !== 'unknown') {
      const intentData = this.findIntentData(neuralPrediction.label);
      if (intentData) {
        const response = this.pickResponse(intentData.responses, input);
        return { text: response, strategy: `neural:${neuralPrediction.label}` };
      }
    }

    // Fact-based response
    if (factHits.length > 0) {
      return {
        text: `Here's what I know: ${factHits[0].text}`,
        strategy: 'facts',
      };
    }

    // Context-aware fallback using recent conversation
    const recentContext = this.memory.getRecentContext(3);
    if (recentContext.length > 0) {
      const lastTopic = this.intelligence.extractTopics(
        recentContext.map((c) => c.input || '').join(' ')
      );
      if (lastTopic.length > 0) {
        return {
          text: `That's an interesting point! We've been discussing topics related to ${lastTopic.slice(0, 3).join(', ')}. Could you elaborate or ask me something more specific? I'm always learning and want to give you the best answer I can.`,
          strategy: 'context_fallback',
        };
      }
    }

    // Ultimate fallback — intelligent responses
    const fallbacks = [
      "That's an interesting question! I don't have a specific answer for that yet, but I'm learning from every conversation. Could you rephrase it or give me more context?",
      "I'm not sure about that one, but I'm constantly improving! The more we chat, the smarter I get. Can you tell me more about what you're looking for?",
      "Great question! I haven't learned enough about that topic yet, but I'll remember this interaction for the future. Try asking me about science, technology, math, history, or philosophy!",
      "I'm still learning about that. My knowledge grows with every conversation! Feel free to teach me — if you tell me the answer, I'll remember it for next time.",
      "Hmm, that's outside my current knowledge. But here's the cool part — I improve over time! Ask me about topics like science, math, coding, health, or creativity, and I can really shine.",
    ];

    return {
      text: fallbacks[Math.floor(Math.random() * fallbacks.length)],
      strategy: 'fallback',
    };
  }

  /**
   * Find intent data from training dataset
   */
  findIntentData(tag) {
    const data = this.intelligence.trainingData;
    if (!data.intents) return null;
    return data.intents.find((i) => i.tag === tag) || null;
  }

  /**
   * Pick a response with some variety — avoid repeating the last response
   */
  pickResponse(responses, input) {
    if (!responses || responses.length === 0) return "I'm not sure what to say.";

    // Check recent responses to avoid repetition
    const recent = this.memory.getRecentContext(5);
    const recentResponses = recent.map((r) => r.output);

    // Filter out recently used responses
    const available = responses.filter((r) => !recentResponses.includes(r));
    const pool = available.length > 0 ? available : responses;

    // Pick random from pool
    let response = pool[Math.floor(Math.random() * pool.length)];

    // Replace placeholders
    response = response.replace('{datetime}', new Date().toLocaleString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    }));

    // Personalize with user name if known
    const prefs = this.memory.getUserPrefs();
    if (prefs.userName) {
      // Occasionally add personalization
      if (Math.random() > 0.7) {
        response = response.replace(/^(Hello|Hey|Hi)!/, `$1, ${prefs.userName}!`);
      }
    }

    return response;
  }

  /**
   * Extract user information from input
   */
  extractUserInfo(input) {
    const nameMatch = input.match(/(?:my name is|i'm called|call me|i am)\s+([a-zA-Z]+)/i);
    if (nameMatch) {
      const name = nameMatch[1].charAt(0).toUpperCase() + nameMatch[1].slice(1).toLowerCase();
      this.memory.setUserPref('userName', name);
      this.memory.addFact({ text: `The user's name is ${name}`, topic: 'user_info' });
    }
  }

  /**
   * Store interaction in both short-term and long-term memory
   */
  storeInteraction(input, output, reasoning, sentiment) {
    const topics = this.intelligence.extractTopics(input);

    // Short-term memory
    this.memory.addToShortTerm({
      input,
      output,
      topics,
      sentiment: sentiment.sentiment,
    });

    // Long-term memory (store meaningful interactions)
    if (input.split(' ').length > 2 || reasoning.strategy.type !== 'fallback') {
      this.memory.addToLongTerm({
        input,
        output,
        topics,
        intent: reasoning.patternMatch ? reasoning.patternMatch.intent : (reasoning.intentMatch ? reasoning.intentMatch.tag : 'unknown'),
        sentiment: sentiment.sentiment,
        quality: reasoning.strategy.type !== 'fallback' ? 'good' : 'needs_improvement',
      });
    }

    // Log interaction for learning
    this.logInteraction(input, output, reasoning);
  }

  /**
   * Handle user feedback for self-improvement
   */
  handleFeedback(type, input) {
    const recent = this.memory.getRecentContext(2);
    if (recent.length < 2) return;

    const lastInteraction = recent[recent.length - 2]; // The interaction before the feedback

    if (type === 'feedback_positive') {
      // Reinforce the successful pattern
      this.memory.addFact({
        text: `Good response to "${lastInteraction.input}": "${lastInteraction.output}"`,
        topic: 'successful_response',
      });
    } else if (type === 'feedback_negative') {
      // Mark as needs improvement
      this.memory.addFact({
        text: `Poor response to "${lastInteraction.input}" — needs improvement`,
        topic: 'failed_response',
      });
    }
  }

  /**
   * Log interaction for future retraining
   */
  logInteraction(input, output, reasoning) {
    try {
      let log = [];
      if (fs.existsSync(INTERACTION_LOG_FILE)) {
        log = JSON.parse(fs.readFileSync(INTERACTION_LOG_FILE, 'utf-8'));
      }
      log.push({
        input,
        output,
        intent: reasoning.patternMatch ? reasoning.patternMatch.intent : 'unknown',
        strategy: reasoning.strategy.type,
        timestamp: Date.now(),
      });
      // Keep last 500 interactions
      if (log.length > 500) {
        log = log.slice(-500);
      }
      fs.writeFileSync(INTERACTION_LOG_FILE, JSON.stringify(log, null, 2));
    } catch (err) {
      // Non-critical error
    }
  }

  /**
   * Get memory stats for debug panel
   */
  getMemoryStats() {
    return {
      ...this.memory.getStats(),
      neuralTrained: this.neural.trained,
      neuralVocabSize: this.neural.vocab.length,
      neuralLabels: this.neural.labels.length,
    };
  }

  /**
   * Get full memory dump for debug/API
   */
  getMemoryDump() {
    return {
      shortTerm: this.memory.getShortTermMemory(),
      longTerm: this.memory.getLongTermMemory(),
      facts: this.memory.getFacts(),
      userPrefs: this.memory.getUserPrefs(),
      stats: this.getMemoryStats(),
    };
  }

  /**
   * Clear conversation history
   */
  clearConversation() {
    this.memory.clearShortTerm();
    return { message: 'Short-term memory cleared. Long-term memory preserved.' };
  }
}

module.exports = ChatEngine;
