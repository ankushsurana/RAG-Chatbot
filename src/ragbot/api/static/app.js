"use strict";

const $ = (id) => document.getElementById(id);
const SPARK = '<svg class="pulse" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 2c1.2 7.8 6.2 12.8 14 14-7.8 1.2-12.8 6.2-14 14-1.2-7.8-6.2-12.8-14-14 7.8-1.2 12.8-6.2 14-14z"/></svg>';
const SERVER_DOWN = "Can't reach the chatbot server. Start it with: uv run ragbot serve";

const state = {
  conversationId: null,
  busy: false,
  loadToken: 0,
  sessionClientId: `session-${Math.random().toString(36).slice(2, 14)}`,
};

function icon(name) {
  return `<svg class="icon" aria-hidden="true"><use href="#${name}"/></svg>`;
}

function escapeHtml(text) {
  const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(text).replace(/[&<>"']/g, (ch) => entities[ch]);
}

function prettyName(filename) {
  return filename.replace(/\.pdf$/i, "").replace(/_SCANNED$/i, "").replace(/_/g, " ");
}

function clientId() {
  const key = "ragbot-client-id";
  try {
    let id = localStorage.getItem(key);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(key, id);
    }
    return id;
  } catch {
    return state.sessionClientId;
  }
}

function withClient(headers = {}) {
  return { ...headers, "X-Client-Id": clientId() };
}

async function getJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, { ...options, headers: withClient(options.headers) });
  } catch {
    throw new Error(SERVER_DOWN);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `The server answered with status ${response.status}.`);
  }
  return response.json();
}

/* --- Rendering ------------------------------------------------------------------------------- */

function formatInline(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, group) =>
      group.split(",").map((n) => `<button class="cite" type="button" data-source="${n.trim()}" aria-label="Open source ${n.trim()}">${n.trim()}</button>`).join(""),
    );
}

function renderText(text) {
  const parts = [];
  let list = null;
  for (const line of escapeHtml(text).split("\n")) {
    const item = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      list = list || [];
      list.push(`<li>${formatInline(item[1])}</li>`);
      continue;
    }
    if (list) { parts.push(`<ul>${list.join("")}</ul>`); list = null; }
    if (line.trim()) parts.push(`<p>${formatInline(line)}</p>`);
  }
  if (list) parts.push(`<ul>${list.join("")}</ul>`);
  return parts.join("");
}

function renderAnswer(reply, answer, rating = null) {
  const sentences = answer.verification.sentences || [];
  const unsupported = sentences.filter((sentence) => !sentence.supported).length;
  const notes = [];
  if (answer.repeated) notes.push('<p class="note">You asked this earlier in this chat, so here is the same answer.</p>');
  if (unsupported) notes.push(`<p class="note warn">${unsupported === 1 ? "One statement" : `${unsupported} statements`} here could not be matched to the source. Please check the cited page.</p>`);

  const sources = answer.citations.length
    ? `<div class="sources">
        <div class="sources-label">Sources</div>
        ${answer.citations.map((citation) => `
          <button class="source" type="button" data-source="${citation.number}">
            <span class="source-number">${citation.number}</span>
            ${icon("i-file")}
            <span class="source-name">${escapeHtml(prettyName(citation.filename))}</span>
            <span class="source-page">Page ${citation.page}</span>
          </button>`).join("")}
      </div>`
    : "";

  const elapsed = answer.timings.total_ms ? `<span class="elapsed">${(answer.timings.total_ms / 1000).toFixed(1)} s</span>` : "";
  const feedback = answer.turn_id && !answer.conversational
    ? `<div class="actions">
        <button class="icon-button feedback" type="button" data-rating="up" aria-label="Good answer" title="Good answer" aria-pressed="${rating === "up"}">${icon("i-up")}</button>
        <button class="icon-button feedback" type="button" data-rating="down" aria-label="Bad answer" title="Bad answer" aria-pressed="${rating === "down"}">${icon("i-down")}</button>
        ${elapsed}
      </div>`
    : "";

  reply.innerHTML = `<div class="answer">${renderText(answer.answer)}</div>${notes.join("")}${sources}${renderEvidence(answer)}${feedback}`;

  const byNumber = new Map(answer.citations.map((citation) => [String(citation.number), citation]));
  reply.querySelectorAll("[data-source]").forEach((button) =>
    button.addEventListener("click", () => {
      const citation = byNumber.get(button.dataset.source);
      if (citation) openViewer(citation);
    }),
  );
  reply.querySelectorAll(".feedback").forEach((button) =>
    button.addEventListener("click", () => sendFeedback(reply, answer.turn_id, button)),
  );
}

function renderEvidence(answer) {
  const chunks = answer.retrieved || [];
  if (!chunks.length) return "";
  const timings = answer.timings;
  const stages = [
    ["Search", timings.retrieval_ms],
    ["First words", timings.first_token_ms],
    ["Full answer", timings.total_ms],
  ].filter(([, value]) => value);
  return `<details class="evidence">
    <summary>Top ${chunks.length} retrieved passages</summary>
    <p class="evidence-lead">Found by meaning search and keyword search, then ranked by a reranker. Highlighted passages were given to the AI model. ${stages.map(([label, value]) => `${label}: ${(value / 1000).toFixed(1)} s`).join(" &middot; ")}</p>
    <ol class="chunks">${chunks.map((chunk) => {
      const score = chunk.rerank_score === null ? 0 : 1 / (1 + Math.exp(-chunk.rerank_score));
      const pages = chunk.page_start === chunk.page_end ? `Page ${chunk.page_start}` : `Pages ${chunk.page_start}-${chunk.page_end}`;
      return `<li class="chunk ${chunk.used_in_context ? "used" : ""}">
        <div class="chunk-head">
          <span class="chunk-name">${escapeHtml(prettyName(chunk.filename))}</span>
          <span class="chunk-page">${pages}</span>
        </div>
        <div class="chunk-score"><span>Match ${Math.round(score * 100)}%</span><div class="chunk-bar"><span style="width:${(score * 100).toFixed(0)}%"></span></div></div>
        <div class="chunk-ranks">Meaning search ${chunk.dense_rank ? `#${chunk.dense_rank}` : "no match"} &middot; Keyword search ${chunk.lexical_rank ? `#${chunk.lexical_rank}` : "no match"}${chunk.source_number ? ` &middot; Sent to AI as source ${chunk.source_number}` : ""}</div>
        <p class="chunk-text">${escapeHtml(chunk.text.slice(0, 420))}${chunk.text.length > 420 ? "..." : ""}</p>
      </li>`;
    }).join("")}</ol>
  </details>`;
}

async function sendFeedback(reply, turnId, button) {
  const buttons = [...reply.querySelectorAll(".feedback")];
  const before = buttons.map((item) => item.getAttribute("aria-pressed"));
  const wasPressed = button.getAttribute("aria-pressed") === "true";
  buttons.forEach((item) => item.setAttribute("aria-pressed", "false"));
  if (!wasPressed) button.setAttribute("aria-pressed", "true");
  try {
    await getJson(`/api/turns/${turnId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating: wasPressed ? null : button.dataset.rating }),
    });
  } catch {
    buttons.forEach((item, index) => item.setAttribute("aria-pressed", before[index]));
  }
}

function addTurn(question) {
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `<div class="bubble">${escapeHtml(question)}</div><div class="reply"></div>`;
  $("conversation").appendChild(turn);
  return turn;
}

function scrollToEnd() {
  $("scroller").scrollTop = $("scroller").scrollHeight;
}

function setChatMode(active) {
  $("main").classList.toggle("is-empty", !active);
}

/* --- Asking ---------------------------------------------------------------------------------- */

async function ask(retryQuestion) {
  const question = (retryQuestion ?? $("question").value).trim();
  if (!question || state.busy) return;
  state.busy = true;
  $("question").value = "";
  syncComposer();
  setChatMode(true);

  const conversationAtStart = state.conversationId;
  const turn = addTurn(question);
  const reply = turn.querySelector(".reply");
  reply.innerHTML = `<div class="thinking">${SPARK}<span>Searching your documents</span></div>`;
  scrollToEnd();

  const docId = $("scope").value;
  let streamed = "";
  let finished = false;
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: withClient({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        question,
        conversation_id: state.conversationId,
        doc_ids: docId ? [docId] : null,
      }),
    }).catch(() => { throw new Error(SERVER_DOWN); });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `The server answered with status ${response.status}.`);
    }

    await readEvents(response, (type, data) => {
      if (type === "retrieval") {
        reply.querySelector(".thinking span").textContent = "Reading the best passages";
      } else if (type === "token") {
        streamed += data.text;
        reply.innerHTML = `<div class="answer">${renderText(streamed)}<span class="caret"></span></div>`;
      } else if (type === "final") {
        finished = true;
        renderAnswer(reply, data);
        state.conversationId = data.conversation_id;
        if (conversationAtStart !== state.conversationId) loadHistory();
        else markHistoryTop();
      } else if (type === "error") {
        throw new Error(data.detail);
      }
      scrollToEnd();
    });
    if (!finished) throw new Error("The answer stopped before it finished.");
  } catch (error) {
    reply.innerHTML = `<div class="error"><span>${escapeHtml(error.message)}</span><button class="retry" type="button">Try again</button></div>`;
    reply.querySelector(".retry").addEventListener("click", () => {
      if (state.busy) return;
      turn.remove();
      ask(question);
    });
  } finally {
    state.busy = false;
    syncComposer();
    scrollToEnd();
    $("question").focus();
  }
}

async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const type = raw.match(/^event: (.*)$/m)?.[1];
      const data = raw.match(/^data: (.*)$/m)?.[1];
      if (type && data) onEvent(type, JSON.parse(data));
    }
  }
}

/* --- Composer -------------------------------------------------------------------------------- */

function syncComposer() {
  const field = $("question");
  field.style.height = "auto";
  field.style.height = `${Math.min(field.scrollHeight, 170)}px`;
  $("send").disabled = state.busy || !field.value.trim();
}

$("question").addEventListener("input", syncComposer);
$("ask-form").addEventListener("submit", (event) => { event.preventDefault(); ask(); });
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); ask(); }
});

/* --- Chats ----------------------------------------------------------------------------------- */

function closeMenu() {
  $("shell").classList.remove("menu-open");
  $("scrim").hidden = true;
}

$("menu-open").addEventListener("click", () => {
  $("shell").classList.add("menu-open");
  $("scrim").hidden = false;
});
$("menu-close").addEventListener("click", closeMenu);
$("scrim").addEventListener("click", closeMenu);

function startNewChat() {
  if (state.busy) return;
  state.loadToken += 1;
  state.conversationId = null;
  $("conversation").innerHTML = "";
  setChatMode(false);
  markActiveHistory();
  closeMenu();
  $("question").focus();
}
$("new-chat").addEventListener("click", startNewChat);

async function openConversation(id) {
  if (state.busy) return;
  closeMenu();
  const token = ++state.loadToken;
  try {
    const turns = await getJson(`/api/conversations/${id}`);
    if (token !== state.loadToken) return;
    state.conversationId = id;
    $("conversation").innerHTML = "";
    setChatMode(true);
    for (const item of turns) {
      const turn = addTurn(item.question);
      renderAnswer(turn.querySelector(".reply"), { ...item.answer, turn_id: item.id }, item.rating);
    }
    markActiveHistory();
    scrollToEnd();
  } catch (error) {
    setChatMode(true);
    const turn = document.createElement("div");
    turn.className = "error";
    turn.textContent = error.message;
    $("conversation").appendChild(turn);
  }
}

function markActiveHistory() {
  document.querySelectorAll(".history-row").forEach((row) => {
    if (row.dataset.id === state.conversationId) row.setAttribute("aria-current", "true");
    else row.removeAttribute("aria-current");
  });
}

function markHistoryTop() {
  const active = document.querySelector(`.history-row[data-id="${state.conversationId}"]`);
  if (active) $("history").prepend(active);
}

function resetDeleteButton(button) {
  clearTimeout(button.confirmTimer);
  button.classList.remove("confirm");
  button.setAttribute("aria-label", "Delete chat");
  button.title = "Delete chat";
  button.innerHTML = icon("i-trash");
}

// The first click asks for confirmation inside the row, the second one deletes.
async function requestDelete(button, id) {
  if (!button.classList.contains("confirm")) {
    document.querySelectorAll(".history-delete.confirm").forEach(resetDeleteButton);
    button.classList.add("confirm");
    button.setAttribute("aria-label", "Confirm: delete this chat");
    button.textContent = "Delete?";
    button.confirmTimer = setTimeout(() => resetDeleteButton(button), 4000);
    return;
  }
  clearTimeout(button.confirmTimer);
  if (state.busy && id === state.conversationId) return;
  try {
    const response = await fetch(`/api/conversations/${id}`, { method: "DELETE", headers: withClient() });
    if (!response.ok && response.status !== 404) throw new Error();
    button.closest(".history-row").remove();
    if (id === state.conversationId) startNewChat();
    if (!document.querySelector(".history-row")) renderEmptyHistory();
  } catch {
    button.textContent = "Failed";
    button.confirmTimer = setTimeout(() => resetDeleteButton(button), 2000);
  }
}

function renderEmptyHistory() {
  $("history").innerHTML = '<p class="history-empty">Your chats will appear here.</p>';
}

async function loadHistory() {
  try {
    const conversations = await getJson("/api/conversations");
    if (!conversations.length) {
      renderEmptyHistory();
      return;
    }
    $("history").innerHTML = conversations.map((chat) => `
      <div class="history-row" data-id="${chat.id}">
        <button class="history-item" type="button" title="${escapeHtml(chat.title)}">${escapeHtml(chat.title)}</button>
        <button class="history-delete" type="button" aria-label="Delete chat" title="Delete chat">${icon("i-trash")}</button>
      </div>`).join("");
    document.querySelectorAll(".history-row").forEach((row) => {
      row.querySelector(".history-item").addEventListener("click", () => openConversation(row.dataset.id));
      row.querySelector(".history-delete").addEventListener("click", (event) => requestDelete(event.currentTarget, row.dataset.id));
    });
    markActiveHistory();
  } catch {
    $("history").innerHTML = '<p class="history-empty">Chats could not be loaded.</p>';
  }
}

/* --- Documents dropdown ---------------------------------------------------------------------- */

async function loadDocuments() {
  try {
    const documents = await getJson("/api/documents");
    for (const doc of documents.filter((item) => item.status === "ready")) {
      const option = document.createElement("option");
      option.value = doc.doc_id;
      option.textContent = prettyName(doc.filename);
      $("scope").appendChild(option);
    }
  } catch {
    /* The dropdown keeps "All documents"; asking a question reports the server problem. */
  }
}

/* --- Source page viewer ---------------------------------------------------------------------- */

function openViewer(citation) {
  const query = citation.bbox ? `?highlight=${citation.bbox.map((value) => value.toFixed(1)).join(",")}` : "";
  const image = $("viewer-image");
  const status = $("viewer-status");
  $("viewer-title").textContent = prettyName(citation.filename);
  $("viewer-page").textContent = `Page ${citation.page}`;
  image.hidden = true;
  status.hidden = false;
  status.textContent = "Loading page";
  image.onload = () => { image.hidden = false; status.hidden = true; };
  image.onerror = () => { status.textContent = "This page could not be loaded."; };
  image.src = `/api/documents/${citation.doc_id}/pages/${citation.page}${query}`;
  $("viewer").showModal();
}

$("viewer-close").addEventListener("click", () => $("viewer").close());
$("viewer").addEventListener("click", (event) => {
  if (event.target === $("viewer")) $("viewer").close();
});

/* --- How it works --------------------------------------------------------------------------- */

const number = (value) => Number(value).toLocaleString("en-US");

async function openPipeline() {
  const body = $("pipeline-body");
  body.innerHTML = '<p class="viewer-status">Loading</p>';
  $("pipeline").showModal();
  closeMenu();
  try {
    const [documents, health] = await Promise.all([getJson("/api/documents"), getJson("/api/health")]);
    const ready = documents.filter((doc) => doc.status === "ready");
    const sum = (key) => ready.reduce((total, doc) => total + doc[key], 0);
    const steps = [
      ["PDFs", number(ready.length), `${number(sum("page_count"))} pages. The smallest has ${number(Math.min(...ready.map((doc) => doc.page_count)))}.`],
      ["Extract and OCR", number(sum("ocr_pages")), `scanned pages read with RapidOCR, plus text from ${number(sum("image_ocr_blocks"))} figures. The rest uses PyMuPDF.`],
      ["Clean", number(sum("removed_boilerplate_blocks")), `headers and footers removed, and ${number(sum("skipped_pages"))} contents and index pages set aside.`],
      ["Chunk", number(sum("chunk_count")), "passages of up to 500 tokens with 15% overlap, each keeping its file and page."],
      ["Embed", "768", `dimensions per passage, using the open-source ${escapeHtml(health.embedding_model)}.`],
      ["Vector DB", number(health.vectors), "vectors in ChromaDB (HNSW), plus a BM25 keyword index."],
    ];
    body.innerHTML = `<ol class="flow">${steps.map(([name, value, note]) => `
        <li class="flow-step"><span class="flow-name">${name}</span><span class="flow-value">${value}</span><span class="flow-note">${note}</span></li>`).join("")}</ol>
      <h3 class="pipeline-heading">Documents in the library</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>PDF</th><th class="num">Pages</th><th class="num">Scanned</th><th class="num">Passages</th></tr></thead>
        <tbody>${ready.map((doc) => `<tr><td>${escapeHtml(prettyName(doc.filename))}</td><td class="num">${number(doc.page_count)}</td><td class="num">${number(doc.ocr_pages)}</td><td class="num">${number(doc.chunk_count)}</td></tr>`).join("")}</tbody>
      </table></div>`;
  } catch (error) {
    body.innerHTML = `<div class="error"><span>${escapeHtml(error.message)}</span></div>`;
  }
}

$("open-pipeline").addEventListener("click", openPipeline);
$("pipeline-close").addEventListener("click", () => $("pipeline").close());
$("pipeline").addEventListener("click", (event) => {
  if (event.target === $("pipeline")) $("pipeline").close();
});

/* --- Startup --------------------------------------------------------------------------------- */

loadDocuments();
loadHistory();
syncComposer();
$("question").focus();
