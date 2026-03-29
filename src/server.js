const express = require('express');
const cors = require('cors');
const path = require('path');
const ChatEngine = require('./engine');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

// Initialize chat engine
console.log('Initializing AI Chat Engine...');
const engine = new ChatEngine();
console.log('AI Chat Engine ready!');

// --- API Routes ---

/**
 * POST /api/ai/chat
 * Main chat endpoint
 * Body: { message: string }
 * Returns: { response: string, debug: object }
 */
app.post('/api/ai/chat', async (req, res) => {
  try {
    const { message } = req.body;

    if (!message || typeof message !== 'string' || message.trim().length === 0) {
      return res.status(400).json({ error: 'Message is required and must be a non-empty string.' });
    }

    const result = await engine.processMessage(message);

    res.json({
      response: result.response,
      debug: result.debug,
    });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: 'Internal server error while processing message.' });
  }
});

/**
 * GET /api/ai/memory
 * Get memory stats and contents
 */
app.get('/api/ai/memory', (req, res) => {
  try {
    const memoryDump = engine.getMemoryDump();
    res.json(memoryDump);
  } catch (err) {
    console.error('Memory error:', err);
    res.status(500).json({ error: 'Failed to retrieve memory.' });
  }
});

/**
 * GET /api/ai/memory/stats
 * Get memory statistics only
 */
app.get('/api/ai/memory/stats', (req, res) => {
  try {
    const stats = engine.getMemoryStats();
    res.json(stats);
  } catch (err) {
    console.error('Stats error:', err);
    res.status(500).json({ error: 'Failed to retrieve stats.' });
  }
});

/**
 * DELETE /api/ai/memory/conversation
 * Clear short-term conversation memory
 */
app.delete('/api/ai/memory/conversation', (req, res) => {
  try {
    const result = engine.clearConversation();
    res.json(result);
  } catch (err) {
    console.error('Clear error:', err);
    res.status(500).json({ error: 'Failed to clear conversation.' });
  }
});

/**
 * GET /api/ai/health
 * Health check endpoint
 */
app.get('/api/ai/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    memoryUsage: process.memoryUsage(),
    stats: engine.getMemoryStats(),
  });
});

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log(`\n🤖 AI Chatbot running at http://localhost:${PORT}`);
  console.log(`📡 API: http://localhost:${PORT}/api/ai/chat`);
  console.log(`🧠 Memory: http://localhost:${PORT}/api/ai/memory`);
  console.log(`❤️  Health: http://localhost:${PORT}/api/ai/health\n`);
});
