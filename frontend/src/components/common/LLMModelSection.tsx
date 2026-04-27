"use client";

import { useEffect, useState, useCallback } from "react";
import { Cpu, Check, Plus, Loader2, X, Trash2 } from "lucide-react";
import { api, type LocalLLMStatus, type CachedModelInfo, type EnvConfigPayload } from "@/lib/api";

const DEFAULT_LOCAL_HF_ID = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF";

interface CloudModel {
    id: string;
    name: string;
    description: string;
}

const DASHSCOPE_LLM_MODELS: CloudModel[] = [
    { id: "qwen3-max", name: "Qwen3 Max", description: "Latest, strongest" },
    { id: "qwen-plus", name: "Qwen Plus", description: "Default, balanced" },
    { id: "qwen-max", name: "Qwen Max", description: "Long context" },
    { id: "qwen-turbo", name: "Qwen Turbo", description: "Fast & cheap" },
];

function formatBytes(n: number): string {
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function shortHfId(hf_id: string): string {
    // "anthfu/Qwen3.6-35B-A3B-APEX-GGUF" -> "Qwen3.6-35B-A3B-APEX-GGUF"
    const i = hf_id.indexOf("/");
    return i === -1 ? hf_id : hf_id.slice(i + 1);
}

function shortGgufFile(file?: string | null): string | null {
    if (!file) return null;
    // "model-Q4_K_M.gguf" -> "Q4_K_M"
    const m = file.match(/(Q\d+_[A-Z0-9_]+)/);
    return m ? `GGUF ${m[1]}` : "GGUF";
}

export default function LLMModelSection() {
    const [env, setEnv] = useState<EnvConfigPayload | null>(null);
    const [cached, setCached] = useState<CachedModelInfo[]>([]);
    const [status, setStatus] = useState<LocalLLMStatus | null>(null);
    const [busy, setBusy] = useState(false);
    const [addInput, setAddInput] = useState<string | null>(null);  // null = card collapsed; "" = expanded with empty input
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        // Settle each independently — a slow or failing endpoint shouldn't
        // blank the whole panel. Promise.all rejects on the first failure,
        // which previously left `cached` empty whenever any one of the
        // three calls hiccupped (e.g. /cached was slow scanning a large
        // HF cache, status check timed out, etc.).
        api.getEnvConfig().then(setEnv).catch(e => console.error("env config:", e));
        api.listCachedLLMs().then(setCached).catch(e => console.error("cached:", e));
        api.getLocalLLMStatus().then(setStatus).catch(e => console.error("status:", e));
    }, []);

    useEffect(() => {
        refresh();
        // Faster polling while a download/load is in progress (1.5s) so the
        // user sees byte progress; slower (3s) when idle.
        const fast = status?.state === "LOADING" || status?.state === "DOWNLOADING";
        const id = setInterval(refresh, fast ? 1500 : 3000);
        return () => clearInterval(id);
    }, [refresh, status?.state]);

    const provider = env?.LLM_PROVIDER ?? "dashscope";
    const dashscopeModel = (env?.DASHSCOPE_MODEL as string) ?? "qwen-plus";
    const localHfId = (env?.LOCAL_LLM_HF_ID as string) ?? "";

    const onPickCloud = async (id: string) => {
        setBusy(true);
        setErrorMsg(null);
        try {
            await api.saveEnvConfig({ LLM_PROVIDER: "dashscope", DASHSCOPE_MODEL: id } as EnvConfigPayload);
            await refresh();
        } catch (e: any) {
            setErrorMsg(`Save failed: ${e.message ?? e}`);
        } finally {
            setBusy(false);
        }
    };

    const onPickLocal = async (m: CachedModelInfo) => {
        setBusy(true);
        setErrorMsg(null);
        try {
            await api.configureLocalLLM({
                hf_id: m.hf_id,
                quant: "auto",
                idle_seconds: 3,
                gguf_file: m.active_gguf_file ?? null,
            });
            await refresh();
        } catch (e: any) {
            setErrorMsg(`Configure failed: ${e.message ?? e}`);
        } finally {
            setBusy(false);
        }
    };

    const onCancel = async (e: React.MouseEvent) => {
        e.stopPropagation();  // don't trigger card's onClick (onPickLocal)
        // No confirm — cancel is now non-destructive. Files are preserved;
        // user can resume later or explicitly delete via a separate action.
        setBusy(true);
        setErrorMsg(null);
        try {
            await api.cancelLocalLLM();
            await refresh();
        } catch (e: any) {
            setErrorMsg(`Cancel failed: ${e.response?.data?.detail ?? e.message ?? e}`);
        } finally {
            setBusy(false);
        }
    };

    const onDelete = async (e: React.MouseEvent, m: CachedModelInfo) => {
        e.stopPropagation();
        const sizeStr = formatBytes(m.size_bytes);
        if (!confirm(`Delete cached model ${shortHfId(m.hf_id)} (${sizeStr})? This frees disk but you'll have to re-download.`)) return;
        setBusy(true);
        setErrorMsg(null);
        try {
            await api.deleteCachedLLM(m.hf_id);
            await refresh();
        } catch (e: any) {
            setErrorMsg(`Delete failed: ${e.response?.data?.detail ?? e.message ?? e}`);
        } finally {
            setBusy(false);
        }
    };

    const onResume = async (e: React.MouseEvent, m: CachedModelInfo) => {
        e.stopPropagation();
        setBusy(true);
        setErrorMsg(null);
        try {
            // Re-configure with this hf_id so the manager picks it up, then
            // fire load() (don't await — hf-xet resumes from .incomplete blob).
            await api.configureLocalLLM({
                hf_id: m.hf_id,
                quant: "auto",
                idle_seconds: 3,
                gguf_file: m.active_gguf_file ?? null,
            });
            await refresh();
            setBusy(false);
            api.loadLocalLLM().catch((err: any) => {
                setErrorMsg(`${err.response?.data?.detail ?? err.message ?? err}`);
            });
        } catch (err: any) {
            setErrorMsg(`Resume failed: ${err.response?.data?.detail ?? err.message ?? err}`);
            setBusy(false);
        }
    };

    const onAddSubmit = async () => {
        const hf = (addInput ?? "").trim();
        if (!hf) return;
        setBusy(true);
        setErrorMsg(null);
        try {
            // Configure first (fast); collapse the form right away so the new
            // model card becomes the single source of truth for status.
            await api.configureLocalLLM({ hf_id: hf, quant: "auto", idle_seconds: 3, gguf_file: null });
            setAddInput(null);
            setBusy(false);
            await refresh();
            // Fire load() but don't await — UI now reflects progress via the
            // card's polled status. Errors surface in status.error.
            api.loadLocalLLM().catch((e: any) => {
                setErrorMsg(`${e.response?.data?.detail ?? e.message ?? e}`);
            });
        } catch (e: any) {
            setErrorMsg(`${e.response?.data?.detail ?? e.message ?? e}`);
            setBusy(false);
        }
    };

    const isDownloading = status?.state === "LOADING" || status?.state === "DOWNLOADING";
    const activeHfId = status?.hf_id ?? "";

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm font-bold text-white">
                <Cpu size={16} className="text-amber-400" />
                <span>LLM (Script Processing)</span>
            </div>

            {/* Cloud cards */}
            <div className="space-y-2">
                <label className="text-xs text-gray-400">Cloud (DashScope)</label>
                <div className="grid grid-cols-2 gap-2">
                    {DASHSCOPE_LLM_MODELS.map((m) => {
                        const selected = provider === "dashscope" && dashscopeModel === m.id;
                        return (
                            <button
                                key={m.id}
                                onClick={() => !busy && onPickCloud(m.id)}
                                disabled={busy}
                                className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left disabled:opacity-50 ${
                                    selected
                                        ? "border-green-500/50 bg-green-500/10"
                                        : "border-white/10 hover:border-white/20 bg-white/5"
                                }`}
                            >
                                {selected && (
                                    <div className="absolute top-2 right-2">
                                        <Check size={14} className="text-green-400" />
                                    </div>
                                )}
                                <span className="text-sm font-medium text-white">{m.name}</span>
                                <span className="text-xs text-gray-500">{m.description}</span>
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Local cards */}
            <div className="space-y-2">
                <label className="text-xs text-gray-400">Local (HuggingFace)</label>
                <div className="grid grid-cols-2 gap-2">
                    {cached.map((m) => {
                        const selected = provider === "local" && localHfId === m.hf_id;
                        const isThisError = status?.state === "ERROR" && activeHfId === m.hf_id;
                        // download_status comes from the per-card /cached endpoint
                        // (authoritative on-disk state). Falls back to "complete"
                        // for backward compat if the field is missing.
                        const dlStatus = m.download_status ?? "complete";
                        const isThisDownloading = dlStatus === "downloading";
                        const isThisPaused = dlStatus === "paused";
                        const ggufLabel = shortGgufFile(m.active_gguf_file);

                        // Progress text: show downloaded / expected if known, else just downloaded
                        const expected = m.expected_size_bytes;
                        const sizeText = expected && expected > 0
                            ? `${formatBytes(m.size_bytes)} / ${formatBytes(expected)} (${Math.floor(100 * m.size_bytes / expected)}%)`
                            : formatBytes(m.size_bytes);

                        return (
                            <button
                                key={m.hf_id}
                                onClick={() => !busy && !isThisPaused && onPickLocal(m)}
                                disabled={busy}
                                className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left disabled:opacity-50 ${
                                    isThisError
                                        ? "border-red-500/50 bg-red-500/10"
                                        : isThisDownloading
                                            ? "border-blue-500/50 bg-blue-500/10"
                                            : isThisPaused
                                                ? "border-yellow-500/50 bg-yellow-500/10"
                                                : selected
                                                    ? "border-purple-500/50 bg-purple-500/10"
                                                    : "border-white/10 hover:border-white/20 bg-white/5"
                                }`}
                            >
                                <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                                    <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
                                        Local
                                    </span>
                                    {ggufLabel && (
                                        <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300">
                                            {ggufLabel}
                                        </span>
                                    )}
                                    {isThisDownloading && (
                                        <span className="flex items-center gap-1 text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300">
                                            <Loader2 size={10} className="animate-spin" />
                                            Downloading
                                        </span>
                                    )}
                                    {isThisPaused && (
                                        <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-300">
                                            Paused
                                        </span>
                                    )}
                                    {isThisError && (
                                        <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-red-500/20 text-red-300">
                                            ERROR
                                        </span>
                                    )}
                                    {selected && !isThisDownloading && !isThisError && !isThisPaused && (
                                        <span className="flex items-center gap-1 text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
                                            <Check size={10} /> Active
                                        </span>
                                    )}
                                </div>
                                <div className="absolute top-2 right-2 flex items-center gap-1">
                                    {isThisPaused && (
                                        <div
                                            role="button"
                                            tabIndex={0}
                                            onClick={(e) => onResume(e, m)}
                                            onKeyDown={(e) => { if (e.key === "Enter") onResume(e as any, m); }}
                                            title="Resume download"
                                            className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-yellow-500/30 hover:bg-yellow-500/50 text-yellow-100 cursor-pointer"
                                        >
                                            ▶ Resume
                                        </div>
                                    )}
                                    {(isThisDownloading || isThisError) && (
                                        <div
                                            role="button"
                                            tabIndex={0}
                                            onClick={onCancel}
                                            onKeyDown={(e) => { if (e.key === "Enter") onCancel(e as any); }}
                                            title={isThisError
                                                ? "Dismiss error (keeps cached files on disk)"
                                                : "Stop download (keeps partial files for resume)"}
                                            className="p-0.5 rounded hover:bg-white/10 text-gray-300 hover:text-white cursor-pointer"
                                        >
                                            <X size={14} />
                                        </div>
                                    )}
                                    <div
                                        role="button"
                                        tabIndex={0}
                                        onClick={(e) => onDelete(e, m)}
                                        onKeyDown={(e) => { if (e.key === "Enter") onDelete(e as any, m); }}
                                        title={`Delete ${formatBytes(m.size_bytes)} from disk (asks confirmation)`}
                                        className="p-0.5 rounded hover:bg-red-500/30 text-red-400/60 hover:text-red-200 cursor-pointer"
                                    >
                                        <Trash2 size={14} />
                                    </div>
                                </div>
                                <span className="text-sm font-medium text-white" title={m.hf_id}>
                                    {shortHfId(m.hf_id)}
                                </span>
                                <span className={`text-xs ${
                                    isThisDownloading ? "text-blue-300" :
                                    isThisPaused ? "text-yellow-300" :
                                    "text-gray-500"
                                }`}>
                                    {isThisDownloading ? `↓ ${sizeText}` :
                                     isThisPaused ? `⏸ ${sizeText}` :
                                     sizeText}
                                </span>
                            </button>
                        );
                    })}

                    {/* Add Local Model card */}
                    {addInput === null ? (
                        <button
                            onClick={() => setAddInput(DEFAULT_LOCAL_HF_ID)}
                            disabled={busy}
                            className="flex flex-col items-center justify-center p-3 rounded-lg border border-dashed border-white/20 hover:border-white/40 bg-white/5 transition-all text-gray-400 hover:text-white disabled:opacity-50"
                        >
                            <Plus size={20} className="mb-1" />
                            <span className="text-sm">Add Local Model</span>
                        </button>
                    ) : (
                        <div className="col-span-2 p-3 rounded-lg border border-amber-500/30 bg-amber-500/5 space-y-2">
                            <div className="text-xs text-gray-400">HuggingFace Model ID</div>
                            <input
                                type="text"
                                value={addInput}
                                onChange={(e) => setAddInput(e.target.value)}
                                placeholder="org/model-name"
                                disabled={busy}
                                className="w-full rounded bg-black/40 border border-white/10 px-3 py-2 text-sm text-white"
                            />
                            <div className="flex justify-end gap-2 pt-1">
                                <button
                                    onClick={() => { setAddInput(null); setErrorMsg(null); }}
                                    disabled={busy}
                                    className="px-3 py-1.5 text-xs text-gray-400 hover:text-white"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={onAddSubmit}
                                    disabled={busy || !addInput.trim()}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-amber-600 hover:bg-amber-500 text-white rounded disabled:opacity-50"
                                >
                                    {busy && isDownloading ? (
                                        <>
                                            <Loader2 size={12} className="animate-spin" />
                                            Downloading...
                                        </>
                                    ) : busy ? (
                                        <>
                                            <Loader2 size={12} className="animate-spin" />
                                            Working...
                                        </>
                                    ) : (
                                        "Download & Use"
                                    )}
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {errorMsg && (
                <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded px-3 py-2">
                    {errorMsg}
                </div>
            )}

            {/* Manager-level error banner with dismiss button */}
            {status?.state === "ERROR" && status.error && (
                <div className="flex items-start gap-2 text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded px-3 py-2">
                    <div className="flex-1">
                        <div className="font-bold mb-1">Model load failed</div>
                        <div className="font-mono text-[10px] break-all opacity-80">{status.error}</div>
                        {activeHfId && (
                            <div className="text-[10px] opacity-60 mt-1">hf_id: {activeHfId}</div>
                        )}
                    </div>
                    <button
                        onClick={onCancel as any}
                        disabled={busy}
                        title="Clear the error state. Cached model files are kept on disk — use the trash icon on the card to delete."
                        className="text-[10px] font-bold uppercase px-2 py-1 rounded bg-red-500/20 hover:bg-red-500/40 text-red-100 disabled:opacity-50"
                    >
                        Dismiss
                    </button>
                </div>
            )}
        </div>
    );
}
