const fs = require('fs');
const path = require('path');

const TRAINING_DATA_FILE = path.join(__dirname, '..', 'data', 'training-data.json');

/**
 * Intelligence Layer — handles:
 * 1. Text similarity matching (TF-IDF inspired)
 * 2. Pattern matching with regex rules
 * 3. Context-aware response generation
 * 4. Rule-based augmentation
 */
class IntelligenceLayer {
  constructor() {
    this.patterns = this.loadPatterns();
    this.trainingData = this.loadTrainingData();
    this.stopWords = new Set([
      'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
      'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
      'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
      'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
      'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
      'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
      'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
      'so', 'than', 'too', 'very', 'just', 'but', 'and', 'or', 'if', 'it',
      'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
      'she', 'her', 'they', 'them', 'this', 'that', 'these', 'those',
      'what', 'which', 'who', 'whom',
    ]);
  }

  // --- Load training data ---

  loadTrainingData() {
    try {
      if (fs.existsSync(TRAINING_DATA_FILE)) {
        return JSON.parse(fs.readFileSync(TRAINING_DATA_FILE, 'utf-8'));
      }
    } catch (err) {
      console.error('Failed to load training data:', err.message);
    }
    return { intents: [], conversations: [] };
  }

  reloadTrainingData() {
    this.trainingData = this.loadTrainingData();
  }

  // --- Pattern rules (regex-based) ---

  loadPatterns() {
    return [
      {
        name: 'greeting',
        patterns: [/^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|howdy|sup|yo)\b/i],
        handler: (match, input) => ({
          intent: 'greeting',
          confidence: 0.95,
        }),
      },
      {
        name: 'farewell',
        patterns: [/^(bye|goodbye|see\s*you|farewell|take\s*care|later|gotta\s*go|good\s*night)\b/i],
        handler: () => ({ intent: 'farewell', confidence: 0.95 }),
      },
      {
        name: 'gratitude',
        patterns: [/(thank|thanks|thx|appreciate|grateful)\b/i],
        handler: () => ({ intent: 'gratitude', confidence: 0.9 }),
      },
      {
        name: 'identity',
        patterns: [/(who\s*(are|r)\s*you|what\s*(are|r)\s*you|your\s*name|tell\s*me\s*about\s*yourself)/i],
        handler: () => ({ intent: 'identity', confidence: 0.95 }),
      },
      {
        name: 'capability',
        patterns: [/(what\s*can\s*you\s*do|help\s*me|what\s*do\s*you\s*know|your\s*capabilities|what\s*are\s*you\s*capable)/i],
        handler: () => ({ intent: 'capability', confidence: 0.9 }),
      },
      {
        name: 'how_are_you',
        patterns: [/(how\s*(are|r)\s*you|how\s*(is|s)\s*it\s*going|how\s*do\s*you\s*feel|what'?s?\s*up)/i],
        handler: () => ({ intent: 'how_are_you', confidence: 0.9 }),
      },
      {
        name: 'math',
        patterns: [/(\d+)\s*([+\-*/x×÷])\s*(\d+)/],
        handler: (match) => {
          const a = parseFloat(match[1]);
          const op = match[2];
          const b = parseFloat(match[3]);
          let result;
          switch (op) {
            case '+': result = a + b; break;
            case '-': result = a - b; break;
            case '*': case 'x': case '×': result = a * b; break;
            case '/': case '÷': result = b !== 0 ? a / b : 'undefined (division by zero)'; break;
          }
          return { intent: 'math', confidence: 1.0, data: { a, op, b, result } };
        },
      },
      {
        name: 'definition',
        patterns: [/(what\s*(is|are|does)|define|explain|tell\s*me\s*(about|what))\s+(.+)/i],
        handler: (match) => ({
          intent: 'definition',
          confidence: 0.7,
          data: { topic: match[4] || match[0] },
        }),
      },
      {
        name: 'opinion',
        patterns: [/(do\s*you\s*(think|believe|like)|what\s*is\s*your\s*(opinion|view|thought)|how\s*do\s*you\s*feel\s*about)/i],
        handler: () => ({ intent: 'opinion', confidence: 0.8 }),
      },
      {
        name: 'joke',
        patterns: [/(tell\s*(me\s*)?(a\s*)?joke|make\s*me\s*laugh|something\s*funny|funny)/i],
        handler: () => ({ intent: 'joke', confidence: 0.9 }),
      },
      {
        name: 'time',
        patterns: [/(what\s*time|current\s*time|what\s*is\s*the\s*time|what\s*day|today'?s?\s*date|current\s*date)/i],
        handler: () => ({ intent: 'time', confidence: 0.95 }),
      },
      {
        name: 'feedback_positive',
        patterns: [/(good\s*(answer|job|response|bot)|well\s*done|nice|great|awesome|perfect|exactly|correct|right|yes\s*that)/i],
        handler: () => ({ intent: 'feedback_positive', confidence: 0.85 }),
      },
      {
        name: 'feedback_negative',
        patterns: [/(bad\s*(answer|response)|wrong|incorrect|no\s*that|not\s*right|terrible|awful|stupid|dumb|useless)/i],
        handler: () => ({ intent: 'feedback_negative', confidence: 0.85 }),
      },
    ];
  }

  // --- Core reasoning pipeline ---

  analyze(input, memoryContext = []) {
    const reasoning = {
      input,
      steps: [],
      patternMatch: null,
      similarCases: [],
      intentMatch: null,
      contextUsed: false,
    };

    // Step 1: Pattern matching
    reasoning.steps.push('Checking pattern rules...');
    const patternResult = this.matchPattern(input);
    if (patternResult) {
      reasoning.patternMatch = patternResult;
      reasoning.steps.push(`Pattern matched: ${patternResult.intent} (confidence: ${patternResult.confidence})`);
    }

    // Step 2: Search training data for similar inputs
    reasoning.steps.push('Searching training data...');
    const intentMatch = this.findBestIntent(input);
    if (intentMatch) {
      reasoning.intentMatch = intentMatch;
      reasoning.steps.push(`Intent match: ${intentMatch.tag} (score: ${intentMatch.score.toFixed(2)})`);
    }

    // Step 3: Search memory for similar past interactions
    reasoning.steps.push('Searching memory for context...');
    if (memoryContext && memoryContext.length > 0) {
      reasoning.contextUsed = true;
      reasoning.steps.push(`Found ${memoryContext.length} relevant memory entries`);
    }

    // Step 4: Combine signals to determine best response strategy
    reasoning.steps.push('Combining signals...');
    const strategy = this.determineStrategy(patternResult, intentMatch, memoryContext);
    reasoning.strategy = strategy;
    reasoning.steps.push(`Strategy: ${strategy.type}`);

    return reasoning;
  }

  // --- Pattern matching ---

  matchPattern(input) {
    for (const rule of this.patterns) {
      for (const pattern of rule.patterns) {
        const match = input.match(pattern);
        if (match) {
          return rule.handler(match, input);
        }
      }
    }
    return null;
  }

  // --- Intent matching from training data ---

  findBestIntent(input) {
    const data = this.trainingData;
    if (!data.intents || data.intents.length === 0) return null;

    const inputTokens = this.tokenize(input);
    let bestMatch = null;
    let bestScore = 0;

    for (const intent of data.intents) {
      for (const pattern of intent.patterns) {
        const patternTokens = this.tokenize(pattern);
        const score = this.cosineSimilarity(inputTokens, patternTokens);
        if (score > bestScore) {
          bestScore = score;
          bestMatch = {
            tag: intent.tag,
            score,
            responses: intent.responses,
            pattern,
          };
        }
      }
    }

    return bestScore > 0.15 ? bestMatch : null;
  }

  // --- Similarity computation ---

  cosineSimilarity(tokensA, tokensB) {
    const allTokens = [...new Set([...tokensA, ...tokensB])];
    if (allTokens.length === 0) return 0;

    const vecA = allTokens.map((t) => tokensA.filter((w) => w === t).length);
    const vecB = allTokens.map((t) => tokensB.filter((w) => w === t).length);

    let dotProduct = 0;
    let magA = 0;
    let magB = 0;
    for (let i = 0; i < allTokens.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magA += vecA[i] * vecA[i];
      magB += vecB[i] * vecB[i];
    }

    const magnitude = Math.sqrt(magA) * Math.sqrt(magB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  // --- Strategy determination ---

  determineStrategy(patternResult, intentMatch, memoryContext) {
    // Priority 1: High-confidence pattern match
    if (patternResult && patternResult.confidence >= 0.9) {
      return { type: 'pattern', source: patternResult };
    }

    // Priority 2: Good intent match from training data
    if (intentMatch && intentMatch.score >= 0.4) {
      return { type: 'intent', source: intentMatch };
    }

    // Priority 3: Memory-based response
    if (memoryContext && memoryContext.length > 0 && memoryContext[0].relevanceScore > 0.3) {
      return { type: 'memory', source: memoryContext[0] };
    }

    // Priority 4: Lower-confidence pattern
    if (patternResult) {
      return { type: 'pattern', source: patternResult };
    }

    // Priority 5: Lower-confidence intent
    if (intentMatch) {
      return { type: 'intent', source: intentMatch };
    }

    // Fallback
    return { type: 'fallback', source: null };
  }

  // --- Extract topics from user input ---

  extractTopics(input) {
    const tokens = this.tokenize(input);
    return tokens.filter((t) => !this.stopWords.has(t) && t.length > 2);
  }

  // --- Sentiment analysis (simple) ---

  analyzeSentiment(text) {
    const positive = [
      'good', 'great', 'awesome', 'excellent', 'amazing', 'wonderful',
      'fantastic', 'love', 'like', 'happy', 'glad', 'thanks', 'thank',
      'best', 'perfect', 'nice', 'cool', 'brilliant', 'beautiful',
      'helpful', 'useful', 'correct', 'right', 'yes', 'please',
    ];
    const negative = [
      'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
      'sad', 'wrong', 'incorrect', 'stupid', 'dumb', 'useless', 'broken',
      'worst', 'ugly', 'boring', 'annoying', 'frustrating', 'no',
    ];

    const tokens = this.tokenize(text);
    let score = 0;
    for (const t of tokens) {
      if (positive.includes(t)) score += 1;
      if (negative.includes(t)) score -= 1;
    }

    if (score > 0) return { sentiment: 'positive', score };
    if (score < 0) return { sentiment: 'negative', score };
    return { sentiment: 'neutral', score: 0 };
  }

  // --- Tokenizer ---

  tokenize(text) {
    return (text || '')
      .toLowerCase()
      .replace(/[^a-z0-9\s']/g, '')
      .split(/\s+/)
      .filter((w) => w.length > 1 && !this.stopWords.has(w));
  }
}

module.exports = IntelligenceLayer;
