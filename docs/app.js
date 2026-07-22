// App Configuration
const EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2";
const LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct";

let textChunks = [];
let embeddings = [];
const EMBEDDING_DIM = 768;

// DOM Elements
const chatFeed = document.getElementById('chat-feed');
const emptyState = document.getElementById('empty-state');
const queryInput = document.getElementById('query-input');
const citationsList = document.getElementById('citations-list');
const citationsCount = document.getElementById('citations-count');
const inputTemp = document.getElementById('input-temp');
const valTemp = document.getElementById('val-temp');
const inputTopk = document.getElementById('input-topk');
const valTopk = document.getElementById('val-topk');
const btnSend = document.getElementById('btn-send-id');
const hfTokenInput = document.getElementById('hf-token-input');
const loadingProgress = document.getElementById('loading-progress');
const serverStatusText = document.getElementById('server-status-text');

// Init slider labels
inputTemp.addEventListener('input', () => valTemp.textContent = inputTemp.value);
inputTopk.addEventListener('input', () => valTopk.textContent = inputTopk.value);

// Load HF Token from LocalStorage
if (localStorage.getItem('hf_token')) {
    hfTokenInput.value = localStorage.getItem('hf_token');
}

hfTokenInput.addEventListener('input', () => {
    localStorage.setItem('hf_token', hfTokenInput.value.trim());
});

// Load Textbook Data (JSON + Binary Embeddings)
async function loadData() {
    try {
        updateProgress("Loading textbook chunks (1.3MB)...");
        const jsonRes = await fetch('text_chunks.json');
        textChunks = await jsonRes.json();

        updateProgress("Loading vector embeddings (4.9MB)...");
        const binRes = await fetch('embeddings.bin');
        const binData = await binRes.arrayBuffer();
        
        // Parse binary float32 array
        const floatArray = new Float32Array(binData);
        const numChunks = textChunks.length;
        
        embeddings = [];
        for (let i = 0; i < numChunks; i++) {
            const start = i * EMBEDDING_DIM;
            embeddings.push(floatArray.subarray(start, start + EMBEDDING_DIM));
        }

        loadingProgress.style.display = 'none';
        serverStatusText.textContent = "Serverless Engine Ready";
        serverStatusText.previousElementSibling.style.backgroundColor = "var(--success-color)";
        serverStatusText.previousElementSibling.style.boxShadow = "0 0 8px var(--success-color)";
    } catch (e) {
        console.error(e);
        updateProgress("Error loading data. Make sure files are hosted correctly.");
    }
}

function updateProgress(text) {
    loadingProgress.textContent = text;
}

// Fill sample queries
function fillQuery(text) {
    queryInput.value = text;
    queryInput.focus();
}

// Dot product similarity (since vectors are unit-normalized, this equals cosine similarity)
function dotProduct(vecA, vecB) {
    let product = 0;
    for (let i = 0; i < EMBEDDING_DIM; i++) {
        product += vecA[i] * vecB[i];
    }
    return product;
}

// RAG Pipeline
async function submitQuery(event) {
    event.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    if (textChunks.length === 0) {
        alert("Please wait for the textbook data to finish loading.");
        return;
    }

    // Clear empty state
    if (emptyState) emptyState.remove();

    appendMessage(query, 'user');
    queryInput.value = '';

    const loadingBubble = appendLoadingBubble();
    chatFeed.scrollTop = chatFeed.scrollHeight;

    try {
        const token = hfTokenInput.value.trim();
        const headers = { "Content-Type": "application/json" };
        if (token) {
            headers["Authorization"] = `Bearer ${token}`;
        }

        // 1. Get Query Embedding from Hugging Face Inference API
        const embedRes = await fetch(
            `https://api-inference.huggingface.co/models/${EMBED_MODEL}`,
            {
                method: "POST",
                headers: headers,
                body: JSON.stringify({ inputs: query })
            }
        );

        if (!embedRes.ok) {
            const err = await embedRes.json();
            throw new Error(`Embedding API failed: ${err.error || embedRes.statusText}`);
        }

        const queryEmbedding = await embedRes.json();

        // 2. Perform Cosine Similarity Search locally in JavaScript
        const scores = [];
        for (let i = 0; i < embeddings.length; i++) {
            const score = dotProduct(queryEmbedding, embeddings[i]);
            scores.push({ index: i, score: score });
        }

        // Sort by score descending
        scores.sort((a, b) => b.score - a.score);

        // Extract top-K chunks
        const topKVal = parseInt(inputTopk.value);
        const contextItems = [];
        for (let i = 0; i < topKVal; i++) {
            const match = scores[i];
            contextItems.push({
                page_number: textChunks[match.index].page,
                sentence_chunk: textChunks[match.index].text,
                score: match.score
            });
        }

        // Update citations sidebar
        updateCitations(contextItems);

        // 3. Generate Answer using Qwen2.5-7B Chat Completion API
        const contextText = contextItems.map(item => `- [Page ${item.page_number}] ${item.sentence_chunk}`).join("\n");
        
        const systemPrompt = "You are a helpful, professional nutrition assistant. Answer the user query using the provided textbook context items. State textbook page numbers when quoting facts.";
        const userPrompt = `Context items:\n${contextText}\n\nQuery: ${query}\n\nAnswer:`;

        const llmRes = await fetch(
            `https://api-inference.huggingface.co/models/${LLM_MODEL}/v1/chat/completions`,
            {
                method: "POST",
                headers: headers,
                body: JSON.stringify({
                    model: LLM_MODEL,
                    messages: [
                        { role: "system", content: systemPrompt },
                        { role: "user", content: userPrompt }
                    ],
                    temperature: parseFloat(inputTemp.value),
                    max_tokens: 512
                })
            }
        );

        loadingBubble.remove();

        if (!llmRes.ok) {
            const err = await llmRes.json();
            throw new Error(`LLM Generation failed: ${err.error || llmRes.statusText}`);
        }

        const llmData = await llmRes.json();
        const answer = llmData.choices[0].message.content.trim();

        appendMessage(answer, 'assistant');

    } catch (e) {
        loadingBubble.remove();
        let errMsg = e.message;
        if (errMsg.includes("Authorization") || errMsg.includes("401") || errMsg.includes("403")) {
            errMsg = "Hugging Face API rate limited or authentication error. Please paste a free Hugging Face User Access Token (with read permissions) in the settings panel to continue.";
        }
        appendMessage(`⚠️ Error: ${errMsg}`, 'assistant');
    }

    chatFeed.scrollTop = chatFeed.scrollHeight;
}

function appendMessage(text, sender) {
    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble', sender);
    bubble.innerHTML = text.replace(/\n/g, '<br>');
    chatFeed.appendChild(bubble);
}

function appendLoadingBubble() {
    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble', 'loading');
    bubble.innerHTML = '<span>Processing RAG pipeline</span><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>';
    chatFeed.appendChild(bubble);
    return bubble;
}

function updateCitations(items) {
    citationsList.innerHTML = '';
    citationsCount.textContent = items.length;

    if (items.length === 0) {
        citationsList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; margin-top: 2rem;">No citations found.</p>';
        return;
    }

    items.forEach((item) => {
        const card = document.createElement('div');
        card.classList.add('citation-card');
        card.onclick = () => showModal(item.page_number, item.score, item.sentence_chunk);
        
        card.innerHTML = `
            <div class="citation-meta">
                <span class="citation-page">Page ${item.page_number}</span>
                <span class="citation-score">Similarity: ${item.score.toFixed(4)}</span>
            </div>
            <div class="citation-body">
                ${item.sentence_chunk}
            </div>
        `;
        citationsList.appendChild(card);
    });
}

function showModal(page, score, text) {
    document.getElementById('modal-page').textContent = `Page ${page}`;
    document.getElementById('modal-score').textContent = `Relevance Score: ${score.toFixed(4)}`;
    document.getElementById('modal-body').textContent = text;
    document.getElementById('source-modal').classList.add('open');
}

function closeModal() {
    document.getElementById('source-modal').classList.remove('open');
}

// Start loading data
loadData();
