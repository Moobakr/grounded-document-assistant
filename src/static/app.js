"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { documents: [], selectedId: null, selectedPage: 1, result: null, token: "", busy: false, mode: "search", view: "library", history: [], conversation: [], conversationScope: null, embeddingReady: false, groqConfigured: false };
let toastTimer;

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function icon(name) {
  const item = node("i");
  item.dataset.lucide = name;
  return item;
}

function icons() { if (window.lucide) window.lucide.createIcons({ attrs: { "aria-hidden": "true" } }); }
function displayName(name) { return name.replace(/\.pdf$/i, "").replaceAll("_", " "); }
function selectedDocument() { return state.documents.find((doc) => doc.id === state.selectedId); }

function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").classList.toggle("error", error);
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 5000);
}

async function api(path, options = {}) {
  const headers = { ...options.headers };
  if (options.method && options.method !== "GET") headers["X-Workspace-Token"] = state.token;
  let response;
  try { response = await fetch(path, { ...options, headers, cache: "no-store" }); }
  catch { throw new Error("Workspace connection lost."); }
  const body = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((value) => value.msg).join("; ") : body.detail;
    throw new Error(detail || "The operation could not be completed.");
  }
  return body;
}

function setBusy(busy, label = "") {
  state.busy = busy;
  $("#operation-banner").hidden = !busy;
  $("#operation-label").textContent = label;
  for (const selector of ["#upload-button", "#upload-zone", "#file-input", "#question", "#document-scope", "#top-k", "#mode-search", "#mode-answer", "#clear-query", "#search-document", "#ai-provider", "#learning-task", "#ai-settings", "#follow-up"]) $(selector).disabled = busy;
  $("#run-query").disabled = busy || !state.documents.length || !state.embeddingReady;
  for (const button of document.querySelectorAll(".delete-document")) button.disabled = busy;
  updateAssistantControls();
}

function setView(view) {
  state.view = view;
  $("#library-view").hidden = view !== "library";
  $("#workspace-view").hidden = view !== "workspace";
  $("#nav-library").classList.toggle("active", view === "library");
  $("#nav-workspace").classList.toggle("active", view === "workspace");
  $("#page-title").textContent = view === "library" ? "Document library" : "Learning workspace";
  $("#page-eyebrow").textContent = view === "library" ? "YOUR KNOWLEDGE, ORGANIZED" : "QUESTIONS & EVIDENCE";
  $("#breadcrumb-view").textContent = view === "library" ? "DOCUMENTS" : "WORKSPACE";
  closeDrawer();
}

function setMode(mode) {
  if (state.busy) return;
  state.mode = mode;
  for (const [id, value] of [["#mode-search", "search"], ["#mode-answer", "ask"]]) {
    $(id).classList.toggle("selected", mode === value);
    $(id).setAttribute("aria-pressed", String(mode === value));
  }
  updateAssistantControls();
}

function updateAssistantControls() {
  const cloud = $("#ai-provider").value === "groq";
  $("#assistant-controls").hidden = state.mode !== "ask";
  $("#cloud-notice").hidden = state.mode !== "ask" || !cloud;
  $("#learning-task").disabled = state.busy || !cloud;
  $("#run-query span").textContent = state.mode === "search" ? "Search" : cloud ? "Ask assistant" : "Get excerpts";
  $("#conversation-count").textContent = state.conversation.length ? `${state.conversation.length} recent turns` : "New conversation";
}

function resetConversation() {
  state.conversation = []; state.conversationScope = null;
  updateAssistantControls();
}

function closeDrawer() { $("#sidebar").classList.remove("open"); $("#drawer-scrim").hidden = true; }

async function loadDocuments(preferredId) {
  state.documents = await api("/api/documents");
  if (preferredId && state.documents.some((doc) => doc.id === preferredId)) state.selectedId = preferredId;
  if (!selectedDocument()) state.selectedId = state.documents[0]?.id || null;
  const total = state.documents.reduce((sum, doc) => sum + doc.chunks, 0);
  $("#document-total").textContent = state.documents.length;
  $("#passage-total").textContent = total;
  $("#nav-count").textContent = state.documents.length;
  $("#collection-count").textContent = state.documents.length;
  $("#footer-count").textContent = `${state.documents.length} documents / ${total} passages`;
  const scope = $("#document-scope"), previous = scope.value;
  scope.replaceChildren(new Option("All documents", ""));
  for (const doc of state.documents) scope.add(new Option(displayName(doc.source), doc.id));
  if (state.documents.some((doc) => doc.id === previous)) scope.value = previous;
  renderCollection();
  renderRows();
  selectDocument(state.selectedId);
  setBusy(state.busy);
}

function renderCollection() {
  $("#collection-list").replaceChildren();
  for (const doc of state.documents) {
    const button = node("button", "collection-item");
    button.classList.toggle("selected", doc.id === state.selectedId);
    button.title = doc.source;
    button.append(icon("file-text"), node("span", "", displayName(doc.source)));
    button.addEventListener("click", () => {
      if (state.busy) return;
      selectDocument(doc.id);
      $("#document-scope").value = doc.id;
      resetConversation();
      if (state.view === "workspace") $("#question").focus();
      closeDrawer();
    });
    $("#collection-list").append(button);
  }
  icons();
}

function renderRows() {
  const filter = $("#document-filter").value.toLowerCase().trim();
  const sort = $("#document-sort").value;
  const docs = state.documents.filter((doc) => doc.source.toLowerCase().includes(filter)).sort((a, b) => {
    if (sort === "newest") return b.created_at.localeCompare(a.created_at);
    if (sort === "passages") return b.chunks - a.chunks;
    return a.source.localeCompare(b.source, undefined, { numeric: true });
  });
  $("#document-rows").replaceChildren();
  $("#visible-count").textContent = `${docs.length} ${docs.length === 1 ? "document" : "documents"}`;
  $("#library-empty").hidden = docs.length > 0;
  $("#library-empty h3").textContent = state.documents.length ? "No matching documents" : "No documents";
  for (const doc of docs) {
    const row = node("tr");
    row.classList.toggle("selected", doc.id === state.selectedId);
    const first = node("td"), cell = node("div", "file-cell");
    if (doc.has_file) {
      const image = node("img", "file-thumb");
      image.src = `/api/documents/${doc.id}/preview/1`;
      image.alt = `${displayName(doc.source)} cover`;
      image.loading = "lazy";
      image.addEventListener("error", () => { image.replaceWith(node("span", "file-thumb file-thumb-placeholder", "PDF")); });
      cell.append(image);
    } else cell.append(node("span", "file-thumb file-thumb-placeholder", "PDF"));
    const info = node("div", "file-info"), title = node("button", "file-title", displayName(doc.source));
    title.title = doc.source;
    title.addEventListener("click", () => selectDocument(doc.id));
    const meta = node("small");
    meta.append(node("span", "indexed-dot"), document.createTextNode(doc.pages ? `PDF / ${doc.pages} pages` : "PDF / Indexed"));
    info.append(title, meta); cell.append(info); first.append(cell);
    const date = new Date(doc.created_at.replace(" ", "T") + "Z");
    const dateCell = node("td", "date-column", Number.isNaN(date.getTime()) ? "-" : date.toLocaleDateString("en-GB", { day: "2-digit", month: "short" }));
    const actions = node("td"), actionGroup = node("div", "row-actions");
    if (doc.has_file) {
      const open = node("a", "icon-button");
      open.href = `/api/documents/${doc.id}/file`;
      open.target = "_blank"; open.rel = "noopener";
      open.title = "Open PDF"; open.setAttribute("aria-label", `Open ${doc.source}`);
      open.append(icon("external-link")); actionGroup.append(open);
    }
    const remove = node("button", "icon-button danger delete-document");
    remove.title = "Delete document"; remove.setAttribute("aria-label", `Delete ${doc.source}`);
    remove.append(icon("trash-2")); remove.disabled = state.busy;
    remove.addEventListener("click", () => confirmDelete(doc));
    actionGroup.append(remove); actions.append(actionGroup);
    row.append(first, node("td", "passages-column", String(doc.chunks)), dateCell, actions);
    $("#document-rows").append(row);
  }
  icons();
}

function selectDocument(id, match = null) {
  state.selectedId = id;
  const doc = selectedDocument();
  $("#inspector-empty").hidden = Boolean(doc);
  $("#inspector-content").hidden = !doc;
  if (!doc) return;
  $("#inspector-title").textContent = match ? "Source inspection" : "Document preview";
  $("#preview-name").textContent = displayName(doc.source);
  $("#preview-metadata").textContent = `${doc.chunks} indexed passages`;
  $("#passage-title").textContent = match ? "Retrieved passage" : "Document details";
  $("#passage-score").textContent = match ? `Cosine ${match.score.toFixed(3)}` : "";
  $("#passage-text").textContent = match ? match.chunk.text : `${doc.source}\n${doc.chunks} indexed passages\nAdded ${doc.created_at} UTC`;
  $("#open-pdf").hidden = !doc.has_file;
  $("#page-total").textContent = doc.pages ? `/ ${doc.pages}` : "";
  $("#preview-page").max = Math.min(doc.pages || 300, 300);
  setPage(match ? match.chunk.page : 1);
  for (const item of document.querySelectorAll(".collection-item")) item.classList.toggle("selected", item.title === doc.source);
  for (const row of document.querySelectorAll("#document-rows tr")) row.classList.toggle("selected", row.querySelector(".file-title")?.title === doc.source);
}

function setPage(number) {
  const doc = selectedDocument();
  if (!doc) return;
  const max = Math.min(doc.pages || 300, 300);
  state.selectedPage = Math.max(1, Math.min(Number(number) || 1, max));
  $("#preview-page").value = state.selectedPage;
  $("#previous-page").disabled = state.selectedPage <= 1;
  $("#next-page").disabled = state.selectedPage >= max;
  $("#open-pdf").href = `/api/documents/${doc.id}/file#page=${state.selectedPage}`;
  const image = $("#preview-image");
  image.hidden = true;
  $("#preview-fallback").hidden = false;
  $("#preview-fallback").textContent = doc.has_file ? "Loading page..." : "Original PDF unavailable";
  if (!doc.has_file) { image.removeAttribute("src"); return; }
  const url = new URL(`/api/documents/${doc.id}/preview/${state.selectedPage}`, location.href).href;
  image.onload = () => { if (image.src === url) { image.hidden = false; $("#preview-fallback").hidden = true; } };
  image.onerror = () => { if (image.src === url) { image.hidden = true; $("#preview-fallback").hidden = false; $("#preview-fallback").textContent = "Page unavailable"; } };
  image.alt = `${displayName(doc.source)}, page ${state.selectedPage}`;
  image.src = url;
}

async function uploadFiles(files) {
  if (state.busy || !files.length) return;
  $("#upload-results").replaceChildren();
  setView("library");
  setBusy(true, "Preparing documents...");
  let preferredId;
  try {
    for (const file of Array.from(files).slice(0, 10)) {
      $("#operation-label").textContent = `Indexing ${file.name}`;
      let message, error = false;
      try {
        if (!/\.pdf$/i.test(file.name)) throw new Error("Only PDF files are accepted.");
        if (!file.size || file.size > 25 * 1024 * 1024) throw new Error("Select a nonempty PDF under 25 MB.");
        const body = new FormData(); body.append("file", file);
        const result = await api("/api/documents", { method: "POST", body });
        preferredId = result.document.id;
        message = `${file.name} / ${result.status === "duplicate" ? "Already indexed" : "Indexed"}`;
      } catch (failure) { error = true; message = `${file.name} / ${failure.message}`; }
      const line = node("div", `upload-result${error ? " error" : ""}`);
      line.append(icon(error ? "circle-alert" : "circle-check"), node("span", "", message));
      $("#upload-results").append(line); icons();
    }
    if (files.length > 10) toast("A batch accepts up to 10 PDFs. Remaining files were not uploaded.", true);
    await loadDocuments(preferredId);
    clearResults();
  } catch (failure) { toast(failure.message, true); }
  finally { $("#file-input").value = ""; setBusy(false); }
}

function confirmDelete(doc) {
  if (state.busy) return;
  const dialog = $("#delete-dialog");
  $("#delete-name").textContent = doc.source;
  dialog.returnValue = "cancel";
  dialog.onclose = async () => {
    if (dialog.returnValue !== "delete") return;
    setBusy(true, `Removing ${doc.source}`);
    try {
      await api(`/api/documents/${doc.id}`, { method: "DELETE" });
      await loadDocuments(); clearResults();
      toast("Document removed.");
    } catch (failure) { toast(failure.message, true); }
    finally { setBusy(false); }
  };
  dialog.showModal();
}

function clearResults() {
  resetConversation();
  state.result = null;
  $("#query-results").hidden = true;
  $("#workspace-empty").hidden = false;
  $("#query-error").hidden = true;
  $("#query-timing").textContent = "";
}

function inspectSource(index) {
  const match = state.result?.sources[index];
  if (!match) return;
  selectDocument(match.chunk.document_id, match);
  document.querySelectorAll(".source-item").forEach((button, i) => button.classList.toggle("selected", i === index));
}

function renderResults() {
  const result = state.result;
  $("#workspace-empty").hidden = true;
  $("#query-results").hidden = false;
  $("#query-error").hidden = !result.generation_error;
  $("#query-error").textContent = result.generation_error ? `${result.generation_error} Retrieved passages retained.` : "";
  $("#query-timing").textContent = `${result.seconds.toFixed(2)} s`;
  $("#result-count").textContent = String(result.sources.length);
  $("#source-list").replaceChildren();
  result.sources.forEach((match, index) => {
    const button = node("button", "source-item");
    const header = node("div", "source-item-header");
    header.append(node("span", "source-number", String(index + 1).padStart(2, "0")), node("span", "source-title", displayName(match.chunk.source)), icon("arrow-up-right"));
    const meta = node("div", "source-meta");
    meta.append(node("span", "", `Page ${match.chunk.page}`), node("span", "", `Cosine ${match.score.toFixed(3)}`));
    button.append(header, meta, node("p", "source-snippet", match.chunk.text));
    button.addEventListener("click", () => inspectSource(index));
    $("#source-list").append(button);
  });
  $("#answer-section").hidden = result.answer === null;
  $("#answer-title").textContent = result.provider === "local" ? "Document excerpts" : ({ explain: "Explanation", answer: "Answer", summary: "Topic summary", quiz: "Practice questions" }[result.task] || "Answer");
  $("#answer-provider").textContent = result.provider === "groq" ? "Groq / AI-generated" : "Local / Experimental";
  $("#follow-up").hidden = result.provider !== "groq" || !result.answer || result.abstained;
  $("#answer-content").replaceChildren();
  if (result.answer) {
    if (result.abstained) $("#answer-content").append(node("p", "abstention", result.answer));
    else if (result.provider === "groq") renderExplanation(result.answer);
    else for (const line of result.answer.split("\n")) {
      const parsed = line.match(/^("(?:[^"\\]|\\.)*") \[(\d+)\]$/);
      const paragraph = node("p", "answer-quote", parsed ? JSON.parse(parsed[1]) : line);
      if (parsed) {
        const sourceIndex = Number(parsed[2]) - 1;
        const match = result.sources[sourceIndex];
        if (match) {
          const citation = node("button", "citation-button");
          citation.title = "Inspect cited passage";
          citation.append(icon("file-text"), document.createTextNode(`${sourceIndex + 1} / p.${match.chunk.page}`));
          citation.addEventListener("click", () => inspectSource(sourceIndex));
          paragraph.append(citation);
        }
      }
      $("#answer-content").append(paragraph);
    }
  }
  icons();
  if (result.sources.length) inspectSource(0);
  else $("#source-list").append(node("p", "muted small", "No passages found."));
}

function appendInline(parent, text) {
  // Render only bold text and validated citation tokens. Model HTML stays literal text.
  for (const part of text.split(/(\*\*[^*]+\*\*|\[\d+\])/g)) {
    const citation = part.match(/^\[(\d+)\]$/);
    if (citation && state.result.sources[Number(citation[1]) - 1]) {
      const index = Number(citation[1]) - 1, button = node("button", "citation-button", part);
      button.title = `Inspect ${state.result.sources[index].chunk.source}, page ${state.result.sources[index].chunk.page}`;
      button.addEventListener("click", () => inspectSource(index)); parent.append(button);
    } else if (part.startsWith("**") && part.endsWith("**")) parent.append(node("strong", "", part.slice(2, -2)));
    else parent.append(document.createTextNode(part));
  }
}

function renderExplanation(answer) {
  let code = null;
  for (const line of answer.split("\n")) {
    if (line.trim().startsWith("```")) {
      if (code) code = null;
      else { code = node("pre", "answer-code"); $("#answer-content").append(code); }
      continue;
    }
    if (code) { code.textContent += line + "\n"; continue; }
    if (!line.trim()) continue;
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    const paragraph = node(heading ? "h3" : "p", heading ? "answer-heading" : "answer-quote");
    appendInline(paragraph, heading ? heading[1] : line);
    $("#answer-content").append(paragraph);
  }
}

async function runQuery() {
  if (state.busy) return;
  const question = $("#question").value.trim();
  if (!question) { $("#question").focus(); return; }
  const scope = $("#document-scope").value || null;
  if (scope !== state.conversationScope) resetConversation();
  const payload = { question, top_k: Number($("#top-k").value), document_id: scope, operation: state.mode,
    provider: $("#ai-provider").value, task: $("#learning-task").value,
    history: state.mode === "ask" && $("#ai-provider").value === "groq" ? state.conversation : [] };
  setBusy(true, state.mode === "search" ? "Searching the collection..." : "Retrieving passages and preparing your answer...");
  $("#query-error").hidden = true;
  try {
    state.result = await api("/api/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    renderResults();
    if (payload.operation === "ask" && payload.provider === "groq" && state.result.answer && !state.result.abstained) {
      state.conversation = [...state.conversation, { question, answer: state.result.answer.slice(0, 3000) }].slice(-2);
      state.conversationScope = scope;
      updateAssistantControls();
    }
    state.history = [{ question, mode: payload.operation, scope: payload.document_id || "" }, ...state.history.filter((entry) => entry.question !== question)].slice(0, 5);
    $("#history-list").replaceChildren();
    for (const entry of state.history) {
      const button = node("button", "history-item"); button.title = entry.question;
      button.append(icon("history"), node("span", "", entry.question));
      button.addEventListener("click", () => {
        if (state.busy) return;
        setView("workspace"); setMode(entry.mode); $("#question").value = entry.question;
        resetConversation();
        $("#document-scope").value = state.documents.some((doc) => doc.id === entry.scope) ? entry.scope : "";
        runQuery();
      });
      $("#history-list").append(button);
    }
    icons();
  } catch (failure) {
    clearResults();
    $("#query-error").textContent = failure.message;
    $("#query-error").hidden = false;
  } finally { setBusy(false); }
}

async function refreshModel() {
  try {
    if ($("#ai-provider").value === "groq") {
      const result = await api("/api/ai");
      state.groqConfigured = result.configured;
      $("#model-label").textContent = result.configured ? "Groq configured" : "Groq key needed";
      $("#model-dot").classList.toggle("online", result.configured);
      $("#groq-model").value = result.model;
      return;
    }
    const result = await api("/api/model");
    $("#model-label").textContent = result.online ? "Model online" : "Model offline";
    $("#model-dot").classList.toggle("online", result.online);
  } catch { $("#model-label").textContent = "Connection lost"; $("#model-dot").classList.remove("online"); }
}

$("#ai-settings").addEventListener("click", async () => {
  await refreshModel();
  const status = await api("/api/ai").catch(() => ({ configured: false }));
  $("#connection-status").textContent = status.configured ? "Key configured. Replace it to change this connection." : "Not connected";
  $("#ai-error").hidden = true; $("#groq-key").value = "";
  $("#ai-dialog").showModal();
});
$("#ai-dialog").addEventListener("close", () => { $("#groq-key").value = ""; });
$("#cancel-ai").addEventListener("click", () => $("#ai-dialog").close());
$("#ai-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#connect-ai").disabled = true;
  try {
    const body = JSON.stringify({ api_key: $("#groq-key").value, model: $("#groq-model").value });
    $("#groq-key").value = "";
    await api("/api/ai", { method: "PUT", headers: { "Content-Type": "application/json" }, body });
    $("#ai-dialog").close(); $("#ai-provider").value = "groq";
    resetConversation(); setMode("ask"); setView("workspace"); await refreshModel();
    toast("Groq configured. Ready to ask.");
  } catch (failure) { $("#ai-error").textContent = failure.message; $("#ai-error").hidden = false; }
  finally { $("#connect-ai").disabled = false; }
});
$("#disconnect-ai").addEventListener("click", async () => {
  try {
    await api("/api/ai", { method: "DELETE" }); $("#ai-dialog").close();
    resetConversation(); await refreshModel(); toast("Groq disconnected.");
  } catch (failure) { $("#ai-error").textContent = failure.message; $("#ai-error").hidden = false; }
});
$("#ai-provider").addEventListener("change", () => { resetConversation(); updateAssistantControls(); refreshModel(); });
$("#document-scope").addEventListener("change", resetConversation);
$("#follow-up").addEventListener("click", () => { $("#question").value = ""; $("#question").focus(); $("#question").scrollIntoView({ block: "center" }); });

$("#nav-library").addEventListener("click", () => setView("library"));
$("#nav-workspace").addEventListener("click", () => setView("workspace"));
$("#mode-search").addEventListener("click", () => setMode("search"));
$("#mode-answer").addEventListener("click", () => setMode("ask"));
$("#menu-button").addEventListener("click", () => { $("#sidebar").classList.toggle("open"); $("#drawer-scrim").hidden = !$("#sidebar").classList.contains("open"); });
$("#drawer-scrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
for (const id of ["#upload-button", "#upload-zone"]) $(id).addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (event) => uploadFiles(event.target.files));
$("#upload-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("#upload-zone").classList.add("drag-over"); });
$("#upload-zone").addEventListener("dragleave", () => $("#upload-zone").classList.remove("drag-over"));
$("#upload-zone").addEventListener("drop", (event) => { event.preventDefault(); $("#upload-zone").classList.remove("drag-over"); uploadFiles(event.dataTransfer.files); });
$("#document-filter").addEventListener("input", renderRows);
$("#document-sort").addEventListener("change", renderRows);
$("#query-form").addEventListener("submit", (event) => { event.preventDefault(); runQuery(); });
$("#top-k").addEventListener("input", (event) => { $("#top-k-value").textContent = event.target.value; });
$("#clear-query").addEventListener("click", () => { clearResults(); $("#question").value = ""; $("#question").focus(); selectDocument(state.selectedId); });
$("#previous-page").addEventListener("click", () => setPage(state.selectedPage - 1));
$("#next-page").addEventListener("click", () => setPage(state.selectedPage + 1));
$("#preview-page").addEventListener("change", (event) => setPage(event.target.value));
$("#search-document").addEventListener("click", () => { $("#document-scope").value = state.selectedId || ""; setView("workspace"); clearResults(); $("#question").focus(); });
$("#refresh-model").addEventListener("click", refreshModel);
for (const button of document.querySelectorAll("[data-question]")) button.addEventListener("click", () => { if (!state.busy) { $("#question").value = button.dataset.question; runQuery(); } });
$("#copy-results").addEventListener("click", async () => {
  if (!state.result) return;
  try {
    const passages = state.result.sources.map((match, i) => `[${i + 1}] ${match.chunk.source}, page ${match.chunk.page}\n${match.chunk.text}`).join("\n\n");
    await navigator.clipboard.writeText([state.result.answer, passages].filter(Boolean).join("\n\n")); toast(state.result.answer ? "Answer and passages copied." : "Passages copied.");
  } catch { toast("Clipboard access unavailable.", true); }
});
$("#download-results").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" }));
  const link = node("a"); link.href = url; link.download = "document-results.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function initialize() {
  icons();
  updateAssistantControls();
  setBusy(true, "Opening the collection...");
  try {
    const status = await api("/api/status");
    state.token = status.token; state.embeddingReady = status.embedding_ready;
    await loadDocuments();
    if (!status.embedding_ready) toast("Local embedding model is missing. Prepare it before indexing or searching.", true);
    await refreshModel();
  } catch (failure) { toast(failure.message, true); }
  finally { setBusy(false); }
}
initialize();
setInterval(refreshModel, 30000);
