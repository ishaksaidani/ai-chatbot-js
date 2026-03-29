const fs = require('fs');
const path = require('path');

const MEMORY_DIR = path.join(__dirname, '..', 'data');
const SHORT_TERM_FILE = path.join(MEMORY_DIR, 'short-term.json');
const LONG_TERM_FILE = path.join(MEMORY_DIR, 'long-term.json');
const USER_PREFS_FILE = path.join(MEMORY_DIR, 'user-prefs.json');
const FACTS_FILE = path.join(MEMORY_DIR, 'facts.json');

const MAX_SHORT_TERM = 50;

function ensureFile(filePath, defaultData) {
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, JSON.stringify(defaultData, null, 2));
  }
}

function readJSON(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

function writeJSON(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

class MemorySystem {
  constructor() {
    if (!fs.existsSync(MEMORY_DIR)) {
      fs.mkdirSync(MEMORY_DIR, { recursive: true });
    }
    ensureFile(SHORT_TERM_FILE, []);
    ensureFile(LONG_TERM_FILE, []);
    ensureFile(USER_PREFS_FILE, {});
    ensureFile(FACTS_FILE, []);
  }

  // --- Short-term memory (recent conversation context) ---

  getShortTermMemory() {
    return readJSON(SHORT_TERM_FILE) || [];
  }

  addToShortTerm(entry) {
    const memory = this.getShortTermMemory();
    memory.push({
      ...entry,
      timestamp: Date.now(),
    });
    // Keep only the last N entries
    while (memory.length > MAX_SHORT_TERM) {
      memory.shift();
    }
    writeJSON(SHORT_TERM_FILE, memory);
  }

  getRecentContext(count = 10) {
    const memory = this.getShortTermMemory();
    return memory.slice(-count);
  }

  // --- Long-term memory (persistent knowledge) ---

  getLongTermMemory() {
    return readJSON(LONG_TERM_FILE) || [];
  }

  addToLongTerm(entry) {
    const memory = this.getLongTermMemory();
    memory.push({
      ...entry,
      timestamp: Date.now(),
      accessCount: 0,
    });
    writeJSON(LONG_TERM_FILE, memory);
  }

  searchLongTerm(query) {
    const memory = this.getLongTermMemory();
    const queryWords = this.tokenize(query);

    const scored = memory.map((entry) => {
      const entryWords = this.tokenize(
        (entry.input || '') + ' ' + (entry.output || '') + ' ' + (entry.topic || '')
      );
      const overlap = queryWords.filter((w) => entryWords.includes(w)).length;
      const score = queryWords.length > 0 ? overlap / queryWords.length : 0;
      return { entry, score };
    });

    return scored
      .filter((s) => s.score > 0.15)
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map((s) => ({ ...s.entry, relevanceScore: s.score }));
  }

  incrementAccessCount(timestamp) {
    const memory = this.getLongTermMemory();
    const item = memory.find((m) => m.timestamp === timestamp);
    if (item) {
      item.accessCount = (item.accessCount || 0) + 1;
      writeJSON(LONG_TERM_FILE, memory);
    }
  }

  // --- User preferences ---

  getUserPrefs() {
    return readJSON(USER_PREFS_FILE) || {};
  }

  setUserPref(key, value) {
    const prefs = this.getUserPrefs();
    prefs[key] = value;
    writeJSON(USER_PREFS_FILE, prefs);
  }

  // --- Facts storage ---

  getFacts() {
    return readJSON(FACTS_FILE) || [];
  }

  addFact(fact) {
    const facts = this.getFacts();
    // Avoid duplicates
    if (!facts.some((f) => f.text === fact.text)) {
      facts.push({
        ...fact,
        timestamp: Date.now(),
      });
      writeJSON(FACTS_FILE, facts);
    }
  }

  searchFacts(query) {
    const facts = this.getFacts();
    const queryWords = this.tokenize(query);
    return facts
      .map((fact) => {
        const factWords = this.tokenize(fact.text + ' ' + (fact.topic || ''));
        const overlap = queryWords.filter((w) => factWords.includes(w)).length;
        const score = queryWords.length > 0 ? overlap / queryWords.length : 0;
        return { ...fact, score };
      })
      .filter((f) => f.score > 0.1)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);
  }

  // --- Memory stats ---

  getStats() {
    return {
      shortTermCount: this.getShortTermMemory().length,
      longTermCount: this.getLongTermMemory().length,
      factsCount: this.getFacts().length,
      userPrefs: Object.keys(this.getUserPrefs()).length,
    };
  }

  clearShortTerm() {
    writeJSON(SHORT_TERM_FILE, []);
  }

  // --- Helpers ---

  tokenize(text) {
    return (text || '')
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter((w) => w.length > 2);
  }
}

module.exports = MemorySystem;
