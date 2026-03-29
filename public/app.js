const API_BASE = '';

// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const charCount = document.getElementById('charCount');
const responseTime = document.getElementById('responseTime');
const statusBadge = document.getElementById('statusBadge');
const toggleDebug = document.getElementById('toggleDebug');
const debugPanel = document.getElementById('debugPanel');
const mobileDebugToggle = document.getElementById('mobileDebugToggle');
const clearChat = document.getElementById('clearChat');

// Stats elements
const statShortTerm = document.getElementById('statShortTerm');
const statLongTerm = document.getElementById('statLongTerm');
const statFacts = document.getElementById('statFacts');
const statNeural = document.getElementById('statNeural');

// Debug elements
const reasoningSteps = document.getElementById('reasoningSteps');
const memoryHits = document.getElementById('memoryHits');
const neuralInfo = document.getElementById('neuralInfo');
const sentimentInfo = document.getElementById('sentimentInfo');

let isProcessing = false;
let welcomeVisible = true;

// --- Event Listeners ---

messageInput.addEventListener('input', () => {
  charCount.textContent = `${messageInput.value.length}/2000`;
  autoResize();
});

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

toggleDebug.addEventListener('click', () => {
  debugPanel.classList.toggle('collapsed');
});

mobileDebugToggle.addEventListener('click', () => {
  debugPanel.classList.toggle('visible');
});

clearChat.addEventListener('click', clearConversation);

// Suggestion chips
document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    messageInput.value = chip.dataset.msg;
    sendMessage();
  });
});

// --- Functions ---

function autoResize() {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message || isProcessing) return;

  // Remove welcome message
  if (welcomeVisible) {
    const welcome = messagesContainer.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    welcomeVisible = false;
  }

  // Add user message
  addMessage(message, 'user');

  // Clear input
  messageInput.value = '';
  charCount.textContent = '0/2000';
  autoResize();

  // Show typing indicator
  isProcessing = true;
  sendBtn.disabled = true;
  statusBadge.textContent = 'Thinking...';
  statusBadge.classList.add('thinking');
  const typingEl = showTyping();

  try {
    const startTime = performance.now();

    const res = await fetch(`${API_BASE}/api/ai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    // Remove typing indicator
    typingEl.remove();

    if (data.error) {
      addMessage('Sorry, something went wrong. Please try again.', 'bot');
    } else {
      addMessage(data.response, 'bot');
      updateDebugPanel(data.debug);
      responseTime.textContent = `Response: ${elapsed}ms`;
    }

    // Refresh memory stats
    fetchMemoryStats();
  } catch (err) {
    typingEl.remove();
    addMessage('Connection error. Please make sure the server is running.', 'bot');
    console.error('Chat error:', err);
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
    statusBadge.textContent = 'Ready';
    statusBadge.classList.remove('thinking');
  }
}

function addMessage(text, type) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${type}`;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = type === 'user' ? 'U' : 'AI';

  const contentDiv = document.createElement('div');

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;

  const meta = document.createElement('div');
  meta.className = 'message-meta';
  const now = new Date();
  meta.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  contentDiv.appendChild(bubble);
  contentDiv.appendChild(meta);

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(contentDiv);

  messagesContainer.appendChild(msgDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showTyping() {
  const typingDiv = document.createElement('div');
  typingDiv.className = 'message bot';
  typingDiv.innerHTML = `
    <div class="message-avatar">AI</div>
    <div class="message-bubble typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  messagesContainer.appendChild(typingDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return typingDiv;
}

function updateDebugPanel(debug) {
  if (!debug) return;

  // Reasoning steps
  if (debug.steps && debug.steps.length > 0) {
    reasoningSteps.innerHTML = debug.steps
      .map(
        (step) => `
      <div class="reasoning-step">
        <span class="step-icon">&#9654;</span>
        <span>${escapeHtml(step)}</span>
      </div>
    `
      )
      .join('');
  }

  // Memory hits
  if (debug.memoryHits && debug.memoryHits.length > 0) {
    memoryHits.innerHTML = debug.memoryHits
      .map(
        (hit) => `
      <div class="memory-hit-item">
        <div class="memory-hit-score">Score: ${(hit.relevanceScore * 100).toFixed(0)}%</div>
        <div class="memory-hit-text" title="${escapeHtml(hit.input || '')}">${escapeHtml(hit.input || 'N/A')}</div>
      </div>
    `
      )
      .join('');
  } else {
    memoryHits.innerHTML = '<p class="placeholder">No memory matches for this query</p>';
  }

  // Neural prediction
  if (debug.neuralPrediction) {
    const pred = debug.neuralPrediction;
    neuralInfo.innerHTML = `
      <div class="neural-prediction">
        <span class="neural-label">${escapeHtml(pred.label)}</span>
        <span class="neural-confidence">${(pred.confidence * 100).toFixed(1)}% confidence</span>
      </div>
    `;
  }

  // Sentiment
  if (debug.sentiment) {
    const s = debug.sentiment;
    const sentimentClass =
      s.sentiment === 'positive'
        ? 'sentiment-positive'
        : s.sentiment === 'negative'
          ? 'sentiment-negative'
          : 'sentiment-neutral';
    const emoji = s.sentiment === 'positive' ? '+' : s.sentiment === 'negative' ? '-' : '~';
    sentimentInfo.innerHTML = `
      <span class="sentiment-badge ${sentimentClass}">
        ${emoji} ${s.sentiment} (score: ${s.score})
      </span>
    `;
  }
}

async function fetchMemoryStats() {
  try {
    const res = await fetch(`${API_BASE}/api/ai/memory/stats`);
    const stats = await res.json();

    statShortTerm.textContent = stats.shortTermCount || 0;
    statLongTerm.textContent = stats.longTermCount || 0;
    statFacts.textContent = stats.factsCount || 0;
    statNeural.textContent = stats.neuralVocabSize || 0;
  } catch (err) {
    // Silently fail
  }
}

async function clearConversation() {
  try {
    await fetch(`${API_BASE}/api/ai/memory/conversation`, { method: 'DELETE' });
    // Clear UI
    messagesContainer.innerHTML = '';
    welcomeVisible = true;
    const welcome = document.createElement('div');
    welcome.className = 'welcome-message';
    welcome.innerHTML = `
      <div class="welcome-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
          <circle cx="9" cy="10" r="1.5" fill="currentColor"/>
          <circle cx="15" cy="10" r="1.5" fill="currentColor"/>
          <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
        </svg>
      </div>
      <h2>Welcome to AI Chatbot!</h2>
      <p>I'm a self-improving AI that learns from our conversations. I use memory, pattern matching, and a neural network to provide intelligent responses.</p>
      <div class="suggestion-chips">
        <button class="chip" data-msg="What can you do?">What can you do?</button>
        <button class="chip" data-msg="Tell me a fun fact">Tell me a fun fact</button>
        <button class="chip" data-msg="Tell me about AI">Tell me about AI</button>
        <button class="chip" data-msg="Tell me a joke">Tell me a joke</button>
        <button class="chip" data-msg="How does the internet work?">How does the internet work?</button>
      </div>
    `;
    messagesContainer.appendChild(welcome);

    // Re-attach chip listeners
    welcome.querySelectorAll('.chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        messageInput.value = chip.dataset.msg;
        sendMessage();
      });
    });

    // Reset debug
    reasoningSteps.innerHTML = '<p class="placeholder">Send a message to see reasoning...</p>';
    memoryHits.innerHTML = '<p class="placeholder">No memory matches yet...</p>';
    neuralInfo.innerHTML = '<p class="placeholder">Waiting for prediction...</p>';
    sentimentInfo.innerHTML = '<p class="placeholder">No analysis yet...</p>';
    responseTime.textContent = '';

    fetchMemoryStats();
  } catch (err) {
    console.error('Failed to clear:', err);
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Initialize
fetchMemoryStats();
messageInput.focus();
