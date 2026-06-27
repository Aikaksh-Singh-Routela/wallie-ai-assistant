// =====================================================================
// Wallie dashboard — Alpine component (MINIMAL WORKING VERSION)
// =====================================================================

const SECTIONS = [
  { id: "identity", label: "Identity", ico: "👤" },
  { id: "personality", label: "Personality", ico: "🎭" },
  { id: "voice", label: "Voice", ico: "🎤" },
  { id: "topics", label: "Topics", ico: "📚" },
  { id: "vision", label: "Vision", ico: "👁️" },
  { id: "play", label: "Play (MC)", ico: "🎮" },
  { id: "hearing", label: "Hearing", ico: "👂" },
  { id: "chat", label: "Chat", ico: "💬" },
  { id: "avatar", label: "Avatar", ico: "🧑" },
  { id: "engine", label: "Engine", ico: "🧠" },
  { id: "secrets", label: "API Keys", ico: "🔑" },
];

const HUMOR_OPTIONS = [
  "ironic", "deadpan", "absurd", "observational",
  "self_deprecating", "roast", "wholesome", "chaotic",
];

function formatHMS(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  seconds = Math.max(0, Math.round(seconds));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

function app() {
  return {
    sections: SECTIONS,
    humorOptions: HUMOR_OPTIONS,
    section: "identity",
    cfg: {},
    profiles: [],
    activeProfile: "default",
    running: false,
    status: {},
    logs: [],
    playLog: "",
    playBusy: false,
    drawerOpen: true,
    saveMsg: "",
    testing: false,
    testResult: "",
    secrets: [],
    wizard: { open: false, step: 1, path: "" },

    async init() {
      await this.loadProfiles();
      await this.loadConfig();
      await this.refreshStatus();
      await this.loadSecrets();
      setInterval(() => this.refreshStatus(), 3000);
    },

    chipInput(getList) {
      return {
        draft: "",
        model() { return getList() || []; },
        add() {
          const v = (this.draft || "").trim();
          if (!v) return;
          const list = getList();
          if (!list.includes(v)) list.push(v);
          this.draft = "";
        },
        remove(i) { getList().splice(i, 1); },
      };
    },

    async loadSecrets() {
      try {
        const r = await fetch("/api/secrets");
        const data = await r.json();
        this.secrets = data.secrets || [];
      } catch (e) { console.warn("loadSecrets:", e); }
    },

    async loadProfiles() {
      try {
        const r = await fetch("/api/profiles");
        const data = await r.json();
        this.profiles = data.profiles || [];
        this.activeProfile = data.active || "default";
      } catch (e) { console.warn("loadProfiles:", e); }
    },

    async loadConfig() {
      try {
        const r = await fetch("/api/config");
        this.cfg = await r.json() || {};
      } catch (e) { console.warn("loadConfig:", e); }
    },

    async save() {
      try {
        const r = await fetch("/api/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.cfg),
        });
        this.saveMsg = r.ok ? "saved" : "fail";
        setTimeout(() => (this.saveMsg = ""), 1400);
      } catch (e) { console.warn("save:", e); }
    },

    async refreshStatus() {
      try {
        const r = await fetch("/api/status");
        this.status = await r.json();
        this.running = !!this.status.running;
      } catch { this.running = false; }
    },

    async start() {
      await fetch("/api/start", { method: "POST" });
      await this.refreshStatus();
    },

    async stop() {
      await fetch("/api/stop", { method: "POST" });
      await this.refreshStatus();
    },

    async resetAudio() {
      try { await fetch("/api/audio/reset", { method: "POST" }); } catch (e) { console.warn(e); }
    },

    toggleInList(list, value) {
      const i = list.indexOf(value);
      if (i >= 0) list.splice(i, 1);
      else list.push(value);
    },

    formatHMS,
  };
}