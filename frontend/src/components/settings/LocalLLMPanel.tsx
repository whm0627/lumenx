"use client";

import { useEffect, useState, useCallback } from "react";
import {
    api,
    type LocalLLMStatus,
    type LocalLLMQuant,
    type CachedModelInfo,
} from "@/lib/api";

const stateBadgeColors: Record<string, string> = {
    UNLOADED: "bg-gray-600",
    DOWNLOADING: "bg-blue-600 animate-pulse",
    LOADING: "bg-blue-600 animate-pulse",
    READY: "bg-green-600",
    ERROR: "bg-red-600",
};

function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function formatRelativeTime(ts: number): string {
    if (!ts) return "never";
    const diff = Date.now() / 1000 - ts;
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
}

export default function LocalLLMPanel() {
    const [hfId, setHfId] = useState("");
    const [quant, setQuant] = useState<LocalLLMQuant>("auto");
    const [idleSec, setIdleSec] = useState(3);
    const [status, setStatus] = useState<LocalLLMStatus | null>(null);
    const [cached, setCached] = useState<CachedModelInfo[]>([]);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [hydrated, setHydrated] = useState(false);

    const refreshStatus = useCallback(async () => {
        try {
            const s = await api.getLocalLLMStatus();
            setStatus(s);
            // Hydrate form once from server
            if (!hydrated) {
                if (s.hf_id) setHfId(s.hf_id);
                if (s.idle_seconds) setIdleSec(s.idle_seconds);
                setHydrated(true);
            }
        } catch (e) {
            console.error("status fetch failed", e);
        }
    }, [hydrated]);

    const refreshCached = useCallback(async () => {
        try {
            setCached(await api.listCachedLLMs());
        } catch (e) {
            console.error("cached fetch failed", e);
        }
    }, []);

    useEffect(() => {
        refreshStatus();
        refreshCached();
        const id = setInterval(refreshStatus, 2000);
        return () => clearInterval(id);
    }, [refreshStatus, refreshCached]);

    const onSave = async () => {
        setBusy(true);
        setMessage(null);
        try {
            await api.configureLocalLLM({ hf_id: hfId, quant, idle_seconds: idleSec });
            setMessage("✓ Configuration saved");
            await refreshStatus();
        } catch (e: any) {
            setMessage(`✗ ${e.message || e}`);
        } finally {
            setBusy(false);
        }
    };

    const onLoad = async () => {
        setBusy(true);
        setMessage(null);
        try {
            await api.loadLocalLLM();
            setMessage("✓ Model loaded");
            await refreshStatus();
            await refreshCached();
        } catch (e: any) {
            setMessage(`✗ ${e.response?.data?.detail || e.message || e}`);
        } finally {
            setBusy(false);
        }
    };

    const onUnload = async () => {
        setBusy(true);
        try {
            await api.unloadLocalLLM();
            await refreshStatus();
        } finally {
            setBusy(false);
        }
    };

    const onTest = async () => {
        setBusy(true);
        setMessage(null);
        try {
            const res = await api.testLocalLLM();
            setMessage(`✓ Test passed in ${res.elapsed_sec}s — "${res.response.slice(0, 80)}"`);
            await refreshStatus();
        } catch (e: any) {
            setMessage(`✗ ${e.response?.data?.detail || e.message || e}`);
        } finally {
            setBusy(false);
        }
    };

    const onDeleteCached = async (id: string) => {
        if (!confirm(`Delete cached model ${id}? This frees disk space.`)) return;
        try {
            await api.deleteCachedLLM(id);
            await refreshCached();
        } catch (e: any) {
            alert(`Delete failed: ${e.response?.data?.detail || e.message || e}`);
        }
    };

    return (
        <div className="space-y-6 rounded-lg bg-gray-900 p-4">
            <h3 className="text-sm font-bold text-white">Local LLM (HuggingFace)</h3>

            {/* Configuration form */}
            <div className="space-y-3">
                <div>
                    <label className="block text-xs text-gray-400 mb-1">HF Model ID</label>
                    <input
                        type="text"
                        value={hfId}
                        onChange={(e) => setHfId(e.target.value)}
                        placeholder="Qwen/Qwen3-8B-Instruct"
                        className="w-full rounded bg-gray-800 px-3 py-2 text-sm text-white"
                    />
                </div>
                <div className="grid grid-cols-2 gap-3">
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Quantization</label>
                        <select
                            value={quant}
                            onChange={(e) => setQuant(e.target.value as LocalLLMQuant)}
                            className="w-full rounded bg-gray-800 px-3 py-2 text-sm text-white"
                        >
                            <option value="auto">Auto</option>
                            <option value="fp16">FP16</option>
                            <option value="bf16">BF16</option>
                            <option value="8bit">8-bit</option>
                            <option value="4bit">4-bit</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Idle unload (s)</label>
                        <input
                            type="number"
                            min={1}
                            max={3600}
                            value={idleSec}
                            onChange={(e) => setIdleSec(parseInt(e.target.value, 10) || 3)}
                            className="w-full rounded bg-gray-800 px-3 py-2 text-sm text-white"
                        />
                    </div>
                </div>
                <button
                    type="button"
                    disabled={busy || !hfId}
                    onClick={onSave}
                    className="rounded bg-blue-600 px-4 py-2 text-sm text-white disabled:opacity-50"
                >
                    Save Configuration
                </button>
            </div>

            <hr className="border-gray-700" />

            {/* Status block */}
            {status && (
                <div className="space-y-2 text-sm text-gray-300">
                    <div className="flex items-center gap-2">
                        <span className={`inline-block h-2 w-2 rounded-full ${stateBadgeColors[status.state] ?? "bg-gray-600"}`} />
                        <span className="font-mono text-white">{status.state}</span>
                        {status.error && <span className="text-red-400 text-xs">— {status.error}</span>}
                    </div>
                    <div>Model: <span className="font-mono">{status.hf_id || "—"}</span></div>
                    <div>
                        VRAM: {(status.vram_used_mb / 1024).toFixed(1)} / {(status.vram_total_mb / 1024).toFixed(1)} GB
                    </div>
                    <div>Last used: {formatRelativeTime(status.last_used_ts)}</div>
                </div>
            )}

            <div className="flex gap-2">
                <button type="button" onClick={onLoad} disabled={busy || !hfId} className="rounded bg-green-700 px-3 py-1.5 text-xs text-white disabled:opacity-50">
                    Load Now
                </button>
                <button type="button" onClick={onUnload} disabled={busy} className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50">
                    Unload
                </button>
                <button type="button" onClick={onTest} disabled={busy || !hfId} className="rounded bg-purple-700 px-3 py-1.5 text-xs text-white disabled:opacity-50">
                    Test
                </button>
            </div>

            {message && (
                <div className="rounded bg-gray-800 px-3 py-2 text-xs text-gray-200">{message}</div>
            )}

            <hr className="border-gray-700" />

            {/* Cached models */}
            <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <h4 className="text-xs font-bold text-white">Cached Models (output/models/LLM/)</h4>
                    <button onClick={refreshCached} className="text-xs text-blue-400 hover:underline">Refresh</button>
                </div>
                {cached.length === 0 ? (
                    <div className="text-xs text-gray-500 italic">No models downloaded yet.</div>
                ) : (
                    <ul className="space-y-1">
                        {cached.map((m) => (
                            <li key={m.hf_id} className="flex items-center justify-between rounded bg-gray-800 px-3 py-1.5 text-xs">
                                <span className="font-mono text-gray-200">{m.hf_id}</span>
                                <div className="flex items-center gap-3">
                                    <span className="text-gray-400">{formatBytes(m.size_bytes)}</span>
                                    <button
                                        onClick={() => onDeleteCached(m.hf_id)}
                                        className="text-red-400 hover:text-red-300"
                                        title="Delete cached model"
                                    >
                                        🗑
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}
