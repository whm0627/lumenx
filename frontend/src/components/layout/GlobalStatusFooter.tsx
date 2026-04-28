"use client";

import { useCallback, useEffect, useState } from "react";
import { Brain, Image as ImageIcon, Cpu, Loader2, AlertTriangle, Mic } from "lucide-react";
import { api, type LocalLLMStatus, type LocalImageStatus, type LocalAudioStatus, type GpuStats, type EnvConfigPayload } from "@/lib/api";

const POLL_INTERVAL_MS = 2000;

function shortHfId(hf_id: string | undefined | null): string {
    if (!hf_id) return "—";
    const i = hf_id.indexOf("/");
    return i === -1 ? hf_id : hf_id.slice(i + 1);
}

function fmtMB(mb: number): string {
    if (mb < 1024) return `${mb} MB`;
    return `${(mb / 1024).toFixed(1)} GB`;
}

type StateColor = "gray" | "amber" | "green" | "red";

function stateColor(state: string): StateColor {
    switch (state) {
        case "READY": return "green";
        case "LOADING":
        case "DOWNLOADING":
        case "SYNTHESIZING":
        case "GENERATING": return "amber";
        case "ERROR": return "red";
        default: return "gray";
    }
}

const COLOR_CLASSES: Record<StateColor, string> = {
    gray: "bg-gray-500/15 text-gray-400 border-gray-500/30",
    amber: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    green: "bg-green-500/15 text-green-300 border-green-500/30",
    red: "bg-red-500/15 text-red-300 border-red-500/30",
};

interface RowProps {
    icon: React.ComponentType<{ size?: number; className?: string }>;
    label: string;
    hfId?: string | null;
    state: string;
    detail?: string;
    error?: string | null;
}

function StatusRow({ icon: Icon, label, hfId, state, detail, error }: RowProps) {
    const c = stateColor(state);
    const isBusy = state === "LOADING" || state === "DOWNLOADING" || state === "SYNTHESIZING" || state === "GENERATING";
    return (
        <div className="flex items-center gap-2 text-xs">
            <Icon size={14} className="text-gray-400 flex-shrink-0" />
            <span className="text-gray-500 font-medium w-7">{label}</span>
            <span className="text-gray-300 truncate max-w-[180px]" title={hfId || ""}>
                {shortHfId(hfId)}
            </span>
            <span
                className={`px-1.5 py-0.5 rounded text-[10px] border ${COLOR_CLASSES[c]} flex items-center gap-1`}
                title={error || ""}
            >
                {isBusy && <Loader2 size={10} className="animate-spin" />}
                {state === "ERROR" && <AlertTriangle size={10} />}
                {state}
            </span>
            {detail && <span className="text-gray-500">{detail}</span>}
        </div>
    );
}

export default function GlobalStatusFooter() {
    const [env, setEnv] = useState<EnvConfigPayload | null>(null);
    const [llm, setLlm] = useState<LocalLLMStatus | null>(null);
    const [img, setImg] = useState<LocalImageStatus | null>(null);
    const [tts, setTts] = useState<LocalAudioStatus | null>(null);
    const [gpu, setGpu] = useState<GpuStats | null>(null);

    // Read env once on mount to decide whether to render at all.
    useEffect(() => {
        api.getEnvConfig().then(setEnv).catch(() => setEnv(null));
    }, []);

    const llmActive = env?.LLM_PROVIDER === "local";
    const imgActive = env?.IMAGE_PROVIDER === "local";
    const ttsActive = env?.TTS_PROVIDER === "local";
    const anyLocal = llmActive || imgActive || ttsActive;

    const refresh = useCallback(async () => {
        const promises: Promise<void>[] = [];
        if (llmActive) {
            promises.push(api.getLocalLLMStatus().then(setLlm).catch(() => { }));
        }
        if (imgActive) {
            promises.push(api.getLocalImageStatus().then(setImg).catch(() => { }));
        }
        if (ttsActive) {
            promises.push(api.getLocalAudioStatus().then(setTts).catch(() => { }));
        }
        // GPU is cheap; poll regardless of which providers are local.
        promises.push(api.getGpuStats().then(setGpu).catch(() => { }));
        await Promise.all(promises);
    }, [llmActive, imgActive, ttsActive]);

    useEffect(() => {
        if (!anyLocal) return;
        refresh();
        const id = setInterval(refresh, POLL_INTERVAL_MS);
        return () => clearInterval(id);
    }, [anyLocal, refresh]);

    if (!anyLocal) return null;

    const gpuPct = gpu && gpu.total_mb > 0 ? Math.min(100, Math.round((gpu.used_mb / gpu.total_mb) * 100)) : 0;

    return (
        // Fixed to the viewport so flex / overflow shenanigans in nested
        // page layouts can't ever push us out of view. Aligned with the
        // right column (left-56 = w-56 sidebar) so we don't overlap the
        // sidebar's own footer.
        <footer className="fixed bottom-0 left-56 right-0 z-40 border-t border-glass-border bg-black/75 backdrop-blur-md px-5 py-2.5 flex items-center gap-6 overflow-x-auto">
            {llmActive && llm && (
                <StatusRow
                    icon={Brain}
                    label="LLM"
                    hfId={llm.hf_id}
                    state={llm.state}
                    detail={llm.state === "READY" && llm.vram_used_mb > 0 ? fmtMB(llm.vram_used_mb) : undefined}
                    error={llm.error}
                />
            )}
            {imgActive && img && (img.pipes && img.pipes.length > 0
                ? img.pipes.map((p) => (
                    <StatusRow
                        key={p.role}
                        icon={ImageIcon}
                        label={p.role === "edit" ? "I2I" : "T2I"}
                        hfId={p.hf_id}
                        state={
                            p.state === "LOADING" && img.phase === "downloading"
                                ? "DOWNLOADING"
                                : p.state
                        }
                        detail={
                            p.progress > 0 && p.progress < 1
                                ? `${Math.round(p.progress * 100)}%`
                                : undefined
                        }
                        error={img.error}
                    />
                ))
                : (
                    // Backwards-compat: older backend that doesn't yet
                    // expose `pipes` — collapse to one IMG row.
                    <StatusRow
                        icon={ImageIcon}
                        label="IMG"
                        hfId={img.hf_id}
                        state={
                            img.state === "LOADING" && img.phase === "downloading"
                                ? "DOWNLOADING"
                                : img.state
                        }
                        detail={
                            img.progress > 0 && img.progress < 1
                                ? `${Math.round(img.progress * 100)}%`
                                : img.phase || undefined
                        }
                        error={img.error}
                    />
                ))}
            {ttsActive && tts && (
                <StatusRow
                    icon={Mic}
                    label="TTS"
                    hfId={tts.hf_id}
                    state={tts.state}
                    detail={
                        tts.progress > 0 && tts.progress < 1
                            ? `${Math.round(tts.progress * 100)}%`
                            : tts.phase_label || tts.phase || undefined
                    }
                    error={tts.error}
                />
            )}
            {gpu && gpu.total_mb > 0 && (
                <div className="flex items-center gap-2 text-xs ml-auto">
                    <Cpu size={14} className="text-gray-400" />
                    <span className="text-gray-500 font-medium">GPU</span>
                    <span className="text-gray-300 tabular-nums">
                        {fmtMB(gpu.used_mb)} / {fmtMB(gpu.total_mb)}
                    </span>
                    <div className="w-24 h-1.5 bg-gray-800 rounded overflow-hidden">
                        <div
                            className={`h-full transition-all ${gpuPct > 90 ? "bg-red-500" : gpuPct > 70 ? "bg-amber-500" : "bg-green-500"}`}
                            style={{ width: `${gpuPct}%` }}
                        />
                    </div>
                    <span className="text-gray-500 tabular-nums w-9 text-right">{gpuPct}%</span>
                </div>
            )}
        </footer>
    );
}
