"use strict";

const $ = (id) => document.getElementById(id);
const ui = { game: null, token: null, info: null, busy: false, generation: 0, controllers: new Set() };
const storageKey = "ayo-local-session";
let retryAction = null;
const playback = { frames: [], index: 0, active: false, running: false, timer: null, resolve: null, flight: null };
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

function stopPlayback(clear = false) {
  clearTimeout(playback.timer);
  playback.flight?.cancel();
  $("board").querySelector(".travel-seed")?.remove();
  playback.active = playback.running = false;
  const resolve = playback.resolve;
  playback.resolve = null;
  if (clear) { playback.frames = []; playback.index = 0; }
  resolve?.();
}

function loadFrames(game) {
  stopPlayback();
  playback.frames = game?.animation_frames ?? [];
  playback.index = Math.max(0, playback.frames.length - 1);
}

function finishPlayback() {
  playback.index = Math.max(0, playback.frames.length - 1);
  stopPlayback();
  render();
}

function scheduleFrame() {
  clearTimeout(playback.timer);
  if (!playback.active || !playback.running) return;
  playback.timer = setTimeout(() => {
    if (playback.index >= playback.frames.length - 1) { finishPlayback(); return; }
    playback.index += 1;
    render();
    scheduleFrame();
  }, 550 - Number($("speed").value));
}

async function animateMove(game, generation) {
  if (generation !== ui.generation) return;
  loadFrames(game);
  if (!playback.frames.length || reducedMotion.matches) { render(); return; }
  playback.index = 0;
  playback.active = playback.running = true;
  await new Promise((resolve) => { playback.resolve = resolve; render(); scheduleFrame(); });
}

function inspectFrame(index) {
  if (!playback.frames.length || ui.busy && !playback.active) return;
  clearTimeout(playback.timer);
  playback.active = true;
  playback.running = false;
  playback.index = Math.max(0, Math.min(playback.frames.length - 1, index));
  if (playback.index === playback.frames.length - 1) finishPlayback();
  else render();
}

function togglePlayback() {
  if (!playback.frames.length) return;
  if (!playback.active) {
    if (!ui.busy) { reducedMotion.matches ? inspectFrame(0) : animateMove(ui.game, ui.generation); }
    return;
  }
  playback.running = !playback.running;
  clearTimeout(playback.timer);
  render();
  scheduleFrame();
}

function movingSeed(frame) {
  playback.flight?.cancel();
  $("board").querySelector(".travel-seed")?.remove();
  if (!playback.active || !playback.running || reducedMotion.matches || frame?.phase !== "sow") return;
  const board = $("board"), source = board.querySelector(`[data-house="${frame.move_from}"]`), landing = board.querySelector(`[data-house="${frame.landing_house}"]`);
  if (!source || !landing) return;
  const origin = board.getBoundingClientRect();
  const center = (pit) => { const r = pit.getBoundingClientRect(); return [r.left + r.width / 2 - origin.left - board.clientLeft - 9.5, r.top + r.height / 2 - origin.top - board.clientTop - 6.5]; };
  const [x0, y0] = center(source), [x1, y1] = center(landing);
  const seed = document.createElement("span");
  seed.className = "travel-seed"; seed.setAttribute("aria-hidden", "true"); seed.style.left = "0"; seed.style.top = "0"; board.append(seed);
  playback.flight = seed.animate([{ transform: `translate(${x0}px,${y0}px)` }, { transform: `translate(${x1}px,${y1}px)` }],
    { duration: Math.min(250, (550 - Number($("speed").value)) * 0.85), easing: "linear", fill: "forwards" });
}

function frameText(frame, game) {
  if (frame?.phase === "pickup") {
    const who = frame.current_player === game.human_player ? "You" : "AlphaZero";
    return `${who} ${who === "You" ? "pick" : "picks"} up ${frame.hand_seeds} seeds from pit ${frame.pickup_house}.`;
  }
  if (frame?.phase === "complete") return `Move ${game.move_number} complete.${frame.shortened ? " Additional sowing steps skipped." : ""}`;
  return frame?.description ?? "Choose who goes first, then start a game.";
}

function saveToken(token) {
  ui.token = token;
  try { token ? sessionStorage.setItem(storageKey, token) : sessionStorage.removeItem(storageKey); } catch { /* Memory-only games still work. */ }
}

async function api(path, body, token = ui.token) {
  const controller = new AbortController();
  ui.controllers.add(controller);
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { ...(body === undefined ? {} : { "Content-Type": "application/json" }), ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw Object.assign(new Error(data.error || "The request failed."), { status: response.status });
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The agent took too long to respond. Try again, or start a new game.");
    if (error instanceof TypeError) throw new Error("Cannot reach the local agent. Check that the Python server is still running.");
    throw error;
  } finally {
    clearTimeout(timer);
    ui.controllers.delete(controller);
  }
}

function buildPits(container, human) {
  const pits = [];
  for (let index = 0; index < 6; index++) {
    const action = human ? index : 5 - index;
    const pit = document.createElement(human ? "button" : "div");
    pit.className = "pit";
    if (human) { pit.type = "button"; pit.disabled = true; pit.addEventListener("click", () => play(action)); }
    pit.innerHTML = `<span class="pit-number"></span><span class="seed-bed" aria-hidden="true"></span><span class="seed-count">4</span>`;
    pit.dataset.action = action;
    container.append(pit);
    pits.push(pit);
  }
  return pits;
}
const humanPits = buildPits($("human-pits"), true);
const opponentPits = buildPits($("opponent-pits"), false);

// Keep each seed in a stable place as the board updates. A sowing frame should
// add one seed, rather than shuffle every seed in every pit.
function renderSeeds(bed, count, house) {
  if (bed.dataset.count === String(count) && bed.dataset.house === String(house)) return;
  bed.dataset.count = count;
  bed.dataset.house = house;
  const opening = [[29, 26], [65, 31], [28, 69], [64, 72], [48, 48], [46, 16], [15, 47], [81, 48], [48, 82], [24, 15], [78, 19], [76, 82]];
  const seeds = Array.from({ length: count }, (_, index) => {
    const seed = document.createElement("span");
    seed.className = "static-seed";
    const angle = index * 2.39996 + house * .7;
    const radius = 37 * Math.sqrt((index + .5) / (count > 24 ? 48 : 24));
    const [x, y] = count <= 12 ? opening[index] : [50 + Math.cos(angle) * radius, 50 + Math.sin(angle) * radius];
    seed.style.left = `${x}%`;
    seed.style.top = `${y}%`;
    seed.style.transform = `translate(-50%, -50%) rotate(${(index * 67 + house * 31) % 110 - 55}deg)`;
    return seed;
  });
  bed.replaceChildren(...seeds);
}

function renderScore(id, count) {
  const score = $(id);
  if (Number(score.textContent) !== count && !reducedMotion.matches) {
    score.animate([{ opacity: .5, transform: "translateY(3px)" }, { opacity: 1, transform: "translateY(0)" }], { duration: 180, easing: "ease-out" });
  }
  score.textContent = count;
}

function render() {
  const game = ui.game;
  const human = game?.human_player ?? Number($("seat").value);
  const frame = playback.active ? playback.frames[playback.index] : null;
  const visible = frame ?? game;
  const locked = ui.busy || playback.active;
  for (const [pits, player] of [[humanPits, human], [opponentPits, 1 - human]]) {
    for (const pit of pits) {
      const action = Number(pit.dataset.action);
      const house = player * 6 + action;
      const count = visible?.board[house] ?? 4;
      const legal = pits === humanPits && !locked && game?.turn === "human" && game.legal_actions.includes(action);
      pit.dataset.house = house;
      pit.classList.toggle("legal", Boolean(legal));
      pit.classList.toggle("selected", frame ? frame.source_house === house : game?.last_move?.player === player && game.last_move.action === action);
      pit.classList.toggle("landing", frame?.landing_house === house && frame.phase !== "complete");
      if (pits === humanPits) pit.disabled = !legal;
      pit.setAttribute("aria-label", `${pits === humanPits ? "Your" : "AlphaZero's"} pit ${house}, ${count} ${count === 1 ? "seed" : "seeds"}${legal ? `, select or press ${action + 1} to play` : ""}`);
      pit.querySelector(".pit-number").textContent = house;
      pit.querySelector(".seed-count").textContent = count;
      const seeds = pit.querySelector(".seed-bed");
      const inFlight = playback.active && playback.running && !reducedMotion.matches && frame?.phase === "sow" && frame.landing_house === house;
      seeds.className = `seed-bed${count > 12 ? " dense" : ""}${count > 24 ? " very-dense" : ""}`;
      renderSeeds(seeds, Math.max(0, count - (inFlight ? 1 : 0)), house);
    }
  }
  const humanScore = visible?.captured[human] ?? 0, agentScore = visible?.captured[1 - human] ?? 0;
  renderScore("human-score", humanScore);
  renderScore("agent-score", agentScore);
  $("human-progress").value = Math.min(25, humanScore);
  $("agent-progress").value = Math.min(25, agentScore);
  $("human-progress").setAttribute("aria-valuetext", `${humanScore} captured seeds, 25 to win`);
  $("agent-progress").setAttribute("aria-valuetext", `${agentScore} captured seeds, 25 to win`);
  $("seat-detail").textContent = game ? `Playing ${human === 0 ? "first" : "second"}` : "Your side of the board";
  $("seat-label").textContent = game ? "Next match starts with" : "Who goes first?";
  const activePlayer = playback.active ? game?.last_move?.player : game?.turn === "human" ? human : game?.turn === "agent" ? 1 - human : null;
  $("human-player").classList.toggle("is-turn", activePlayer === human);
  $("agent-player").classList.toggle("is-turn", activePlayer === 1 - human);
  $("board").setAttribute("aria-busy", String(locked));
  $("undo").disabled = locked || !game?.can_undo;
  $("new-game").disabled = !ui.info;
  $("new-game").textContent = game?.turn === "finished" ? "Play again" : game ? "New match" : "Start match";
  $("move-count").textContent = game ? `Move ${game.move_number}` : "";
  let title = "Take a seat.", description = "Choose who goes first, then start your match.", state = "ready";
  if (playback.active) {
    const actor = game.last_move?.player === human ? "Your move" : "AlphaZero's move";
    title = `${actor} · ${playback.running ? "sowing" : "paused"}`;
    description = playback.running ? "Follow the seeds around the board." : "Resume the replay, or skip ahead to keep playing.";
    state = playback.running ? "sowing" : "paused";
  } else if (game?.turn === "finished") {
    title = { won: "You won. Well played!", lost: "AlphaZero takes this one.", draw: "An even match. It's a draw." }[game.result];
    description = `${game.captured[human]}–${game.captured[1 - human]} captured seeds. ${game.truncated ? "The move limit was reached; remaining seeds were collected by row." : "Start another game, or undo to explore a different move."}`;
    state = game.result;
  } else if (game?.turn === "agent") {
    title = ui.busy ? "AlphaZero is thinking…" : "AlphaZero's turn.";
    description = ui.busy ? "Considering the next move." : "Use Try again to continue the match.";
    state = "thinking";
  } else if (game?.turn === "human") {
    title = ui.busy ? "Playing your move…" : "Your turn.";
    description = ui.busy ? "Sowing your seeds." : "Choose a lit pit in your bottom row.";
    state = ui.busy ? "sowing" : "human";
  } else if (ui.busy) {
    title = "Setting the board…";
    description = "Your game will be ready in a moment.";
    state = "thinking";
  }
  $("turn-panel").dataset.state = state;
  if ($("turn-title").textContent !== title) $("turn-title").textContent = title;
  $("turn-description").textContent = description;
  $("frame-description").textContent = frame ? frameText(frame, game) : game?.last_move ? frameText(playback.frames.at(-1) ?? { phase: "complete" }, game) : "Four seeds in every pit. Capture 25 to win.";
  const held = frame ? 48 - frame.board.reduce((a, b) => a + b, 0) - frame.captured.reduce((a, b) => a + b, 0) : 0;
  $("hand-count").textContent = held ? `${held} in hand` : "";
  const hasFrames = playback.frames.length > 0;
  const controlsLocked = ui.busy && !playback.active;
  $("replay-play").disabled = !hasFrames || controlsLocked;
  $("replay-play").textContent = playback.active ? playback.running ? "Pause" : "Resume" : "Replay move";
  $("frame-prev").disabled = !hasFrames || controlsLocked || playback.index === 0;
  $("frame-next").disabled = !hasFrames || controlsLocked || playback.index >= playback.frames.length - 1;
  $("frame-end").disabled = !hasFrames || controlsLocked || !playback.active;
  $("timeline").disabled = !hasFrames || controlsLocked;
  $("timeline").max = Math.max(0, playback.frames.length - 1);
  $("timeline").value = playback.index;
  $("frame-status").textContent = hasFrames ? `Sowing step ${playback.index + 1} of ${playback.frames.length}` : "Play a move to review its sowing.";
  movingSeed(frame);
  const moves = game?.moves ?? [];
  $("history-empty").hidden = moves.length > 0;
  $("history-empty").querySelector("p").textContent = game ? "Your match is ready." : "Your match starts here.";
  $("history-count").textContent = moves.length ? `${Math.min(12, game.move_number)} of ${game.move_number}` : "";
  const items = [...moves].reverse().map((move) => {
    const item = document.createElement("li");
    const name = document.createElement("span"), pit = document.createElement("span"), capture = document.createElement("span");
    name.className = "move-player"; name.textContent = move.player === human ? "You" : "AlphaZero";
    pit.className = "move-pit"; pit.textContent = `Pit ${move.player * 6 + move.action}`;
    capture.className = "move-capture";
    const yours = move.captured[human], theirs = move.captured[1 - human];
    capture.textContent = yours || theirs ? `${yours ? `You +${yours}` : ""}${yours && theirs ? ", " : ""}${theirs ? `AI +${theirs}` : ""}` : "—";
    item.append(name, pit, capture);
    return item;
  });
  $("move-list").replaceChildren(...items);
}

function clearError() { $("error-panel").hidden = true; retryAction = null; }
function showError(error, retry) {
  $("error-message").textContent = error.message;
  retryAction = retry;
  $("retry-button").hidden = !retry;
  $("error-panel").hidden = false;
}

async function recover(error, generation) {
  if (generation !== ui.generation) return;
  // A response may have been lost after the server committed the move. Read the
  // authoritative revision before offering a retry; never replay the human move.
  try {
    const game = await api("/api/game");
    if (generation !== ui.generation) return;
    ui.game = game;
    loadFrames(game);
  } catch (refreshError) {
    if (generation !== ui.generation) return;
    if (refreshError.status === 404) { saveToken(null); ui.game = null; error = refreshError; }
  }
  if (generation !== ui.generation) return;
  ui.busy = false;
  render();
  showError(error, ui.game?.turn === "agent" ? resumeAgent : ui.game ? null : newGame);
}

async function agentTurn(generation) {
  if (ui.game?.turn !== "agent" || generation !== ui.generation) return;
  const game = await api("/api/agent", { revision: ui.game.revision });
  if (generation === ui.generation) { ui.game = game; await animateMove(game, generation); render(); }
}

async function resumeAgent() {
  if (ui.busy || playback.active || ui.game?.turn !== "agent") return;
  const generation = ui.generation;
  clearError(); ui.busy = true; render();
  try { await agentTurn(generation); } catch (error) { await recover(error, generation); }
  finally { if (generation === ui.generation) { ui.busy = false; render(); } }
}

async function newGame() {
  if (!ui.info) return;
  const generation = ++ui.generation;
  stopPlayback(true);
  for (const controller of ui.controllers) controller.abort();
  clearError(); ui.game = null; ui.busy = true; render();
  try {
    const data = await api("/api/new", { human_player: Number($("seat").value) }, null);
    if (generation !== ui.generation) return;
    saveToken(data.session_id); ui.game = data.game; render();
    await agentTurn(generation);
  } catch (error) {
    if (generation === ui.generation) {
      if (ui.game) await recover(error, generation);
      else { saveToken(null); showError(error, newGame); }
    }
  } finally { if (generation === ui.generation) { ui.busy = false; render(); } }
}

async function play(action) {
  if (ui.busy || playback.active || ui.game?.turn !== "human" || !ui.game.legal_actions.includes(action)) return;
  const generation = ui.generation;
  const focusedPit = document.activeElement;
  clearError(); ui.busy = true; render();
  try {
    const game = await api("/api/move", { action, revision: ui.game.revision });
    if (generation !== ui.generation) return;
    ui.game = game;
    await animateMove(game, generation);
    await agentTurn(generation);
  } catch (error) { await recover(error, generation); }
  finally {
    if (generation === ui.generation) {
      ui.busy = false; render();
      // Disabled buttons lose focus during the AI reply. Restore a legal pit for
      // keyboard users without moving focus away from controls they used meanwhile.
      if (humanPits.includes(focusedPit) && document.activeElement === document.body) {
        (humanPits.find((pit) => pit === focusedPit && !pit.disabled) || humanPits.find((pit) => !pit.disabled))?.focus();
      }
    }
  }
}

async function undo() {
  if (ui.busy || playback.active || !ui.game?.can_undo) return;
  const generation = ui.generation;
  clearError(); ui.busy = true; render();
  try {
    const game = await api("/api/undo", { revision: ui.game.revision });
    if (generation === ui.generation) { ui.game = game; loadFrames(game); }
  } catch (error) { await recover(error, generation); }
  finally { if (generation === ui.generation) { ui.busy = false; render(); } }
}

async function connect() {
  clearError();
  try {
    ui.info = await api("/api/info", undefined, null);
    $("agent-detail").textContent = `AlphaZero C${ui.info.checkpoint} ready`;
    $("connection-status").dataset.state = "online";
    $("search-detail").textContent = `AlphaZero C${ui.info.checkpoint} · ${ui.info.simulations} simulations per move`;
    try { saveToken(sessionStorage.getItem(storageKey)); } catch { /* Storage may be unavailable. */ }
    if (ui.token) {
      try {
        ui.game = await api("/api/game");
        loadFrames(ui.game);
        $("seat").value = ui.game.human_player;
      } catch (error) { saveToken(null); showError(error, newGame); }
    }
    render();
    if (ui.game?.turn === "agent") await resumeAgent();
  } catch (error) {
    $("agent-detail").textContent = "Agent is offline";
    $("connection-status").dataset.state = "offline";
    showError(error, connect);
  }
}

$("new-game").addEventListener("click", newGame);
$("undo").addEventListener("click", undo);
$("retry-button").addEventListener("click", () => retryAction?.());
$("rules-link").addEventListener("click", () => {
  $("rules").open = true;
  $("rules").querySelector("summary").focus({ preventScroll: true });
  $("rules").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "nearest" });
});
$("seat").addEventListener("change", () => { if (!ui.game) render(); });
$("replay-play").addEventListener("click", togglePlayback);
$("frame-prev").addEventListener("click", () => inspectFrame(playback.index - 1));
$("frame-next").addEventListener("click", () => inspectFrame(playback.index + 1));
$("frame-end").addEventListener("click", finishPlayback);
$("timeline").addEventListener("input", () => inspectFrame(Number($("timeline").value)));
$("speed").addEventListener("input", scheduleFrame);
reducedMotion.addEventListener("change", () => { if (reducedMotion.matches && playback.active) finishPlayback(); });
new ResizeObserver(() => render()).observe($("board"));
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.repeat || !/^[1-6]$/.test(event.key)) return;
  if (event.target.closest("input, select, textarea, [contenteditable='true']")) return;
  if (!ui.busy && !playback.active && ui.game?.turn === "human" && ui.game.legal_actions.includes(Number(event.key) - 1)) {
    event.preventDefault(); play(Number(event.key) - 1);
  }
});
render();
connect();
