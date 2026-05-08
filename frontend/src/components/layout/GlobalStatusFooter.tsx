"use client";

import { useEffect, useRef, useState } from "react";
import { Brain, Image as ImageIcon, Cpu, Loader2, AlertTriangle, Mic, Film } from "lucide-react";
import { api, type LocalLLMStatus, type LocalImageStatus, type LocalAudioStatus, type LocalVideoStatus, type GpuStats } from "@/lib/api";

const POLL_INTERVAL_MS = 2000;
// Per-endpoint timeout. Generous because the FastAPI event loop competes
// with a sync model-load + torchao quantize_ thread for the GIL during
// the first ~60s of a generation. async status endpoints still respond,
// just with much higher tail latency. A short timeout would discard
// every call as "failed" and leave the footer perpetually empty.
const PER_CALL_TIMEOUT_MS = 30000;

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

// Race a promise against a timeout. Resolves to null on timeout/error so
// callers can simply `keep previous state`.
function withTimeout<T>(p: Promise<T>, ms: number): Promise<T | null> {
    return new Promise((resolve) => {
        const timer = setTimeout(() => resolve(null), ms);
        p.then(
            (v) => { clearTimeout(timer); resolve(v); },
            () => { clearTimeout(timer); resolve(null); },
        );
    });
}

export default function GlobalStatusFooter() {
    const [llm, setLlm] = useState<LocalLLMStatus | null>(null);
    const [img, setImg] = useState<LocalImageStatus | null>(null);
    const [tts, setTts] = useState<LocalAudioStatus | null>(null);
    const [video, setVideo] = useState<LocalVideoStatus | null>(null);
    const [gpu, setGpu] = useState<GpuStats | null>(null);

    // useRef rather than useState to avoid recreating the interval whenever
    // a poll completes. The interval body always reads the latest setters.
    const pollingRef = useRef(false);

    useEffect(() => {
        const tick = async () => {
            if (pollingRef.current) return;  // skip if previous tick still in flight
            pollingRef.current = true;
            try {
                // Independent + timed-out — one stalled endpoint can't block
                // the others, and on stall we keep the previous render.
                const [l, i, t, v, g] = await Promise.all([
                    withTimeout(api.getLocalLLMStatus(), PER_CALL_TIMEOUT_MS),
                    withTimeout(api.getLocalImageStatus(), PER_CALL_TIMEOUT_MS),
                    withTimeout(api.getLocalAudioStatus(), PER_CALL_TIMEOUT_MS),
                    withTimeout(api.getLocalVideoStatus(), PER_CALL_TIMEOUT_MS),
                    withTimeout(api.getGpuStats(), PER_CALL_TIMEOUT_MS),
                ]);
                if (l) setLlm(l);
                if (i) setImg(i);
                if (t) setTts(t);
                if (v) setVideo(v);
                if (g) setGpu(g);
            } finally {
                pollingRef.current = false;
            }
        };
        tick();
        const id = setInterval(tick, POLL_INTERVAL_MS);
        return () => clearInterval(id);
    }, []);

    // Show a row only when the manager reports the provider is configured —
    // that's what hf_id non-empty means. State alone is unreliable
    // (UNLOADED could mean "not configured" or "configured but unloaded").
    const hasLLM = !!llm?.hf_id;
    const hasImg = !!img?.hf_id || !!(img?.pipes && img.pipes.some((p) => p.hf_id));
    const hasTts = !!tts?.hf_id;
    const hasVideo = !!video?.hf_id;
    const hasGpu = !!gpu && gpu.total_mb > 0;
    const hasAnyData = hasLLM || hasImg || hasTts || hasVideo || hasGpu;

    // Footer is ALWAYS visible. Even when every API is failing or no local
    // provider is configured, we render a baseline "● Backend ..." indicator
    // so the user can always see the footer is mounted and functioning. This
    // prevents the recurring "footer disappeared!" panic during heavy
    // generation when status endpoints can briefly stall.
    const gpuPct = hasGpu ? Math.min(100, Math.round((gpu!.used_mb / gpu!.total_mb) * 100)) : 0;

    return (
        <footer className="fixed bottom-0 left-0 right-0 z-40 border-t-2 border-primary/20 bg-black/90 backdrop-blur-md px-5 py-2.5 flex items-center gap-6 overflow-x-auto min-h-[44px]">
            {hasLLM && llm && (
                <StatusRow
                    icon={Brain}
                    label="LLM"
                    hfId={llm.hf_id}
                    state={llm.state}
                    detail={llm.state === "READY" && llm.vram_used_mb > 0 ? fmtMB(llm.vram_used_mb) : undefined}
                    error={llm.error}
                />
            )}
            {hasImg && img && (img.pipes && img.pipes.length > 0
                ? img.pipes.filter((p) => p.hf_id).map((p) => (
                    <StatusRow
                        key={p.role}
                        icon={ImageIcon}
                        label={p.role === "edit" ? "I2I" : "T2I"}
                        hfId={p.hf_id}
                        state={p.state === "LOADING" && img.phase === "downloading" ? "DOWNLOADING" : p.state}
                        detail={p.progress > 0 && p.progress < 1 ? `${Math.round(p.progress * 100)}%` : undefined}
                        error={img.error}
                    />
                ))
                : (
                    <StatusRow
                        icon={ImageIcon}
                        label="IMG"
                        hfId={img.hf_id}
                        state={img.state === "LOADING" && img.phase === "downloading" ? "DOWNLOADING" : img.state}
                        detail={img.progress > 0 && img.progress < 1 ? `${Math.round(img.progress * 100)}%` : img.phase || undefined}
                        error={img.error}
                    />
                ))}
            {hasTts && tts && (
                <StatusRow
                    icon={Mic}
                    label="TTS"
                    hfId={tts.hf_id}
                    state={tts.state}
                    detail={tts.progress > 0 && tts.progress < 1 ? `${Math.round(tts.progress * 100)}%` : tts.phase_label || tts.phase || undefined}
                    error={tts.error}
                />
            )}
            {hasVideo && video && (
                <StatusRow
                    icon={Film}
                    label="VID"
                    hfId={`${video.hf_id} (${video.quant})`}
                    state={video.state}
                    detail={video.progress > 0 && video.progress < 1 ? `${Math.round(video.progress * 100)}%` : video.phase_label || video.phase || undefined}
                    error={video.error}
                />
            )}
            {/* Baseline indicator — always visible. Stays so the user can
                always confirm the footer is alive even when every status
                endpoint is stalled by a heavy generation. */}
            {!hasAnyData && (
                <div className="flex items-center gap-2 text-xs">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                    <span className="text-gray-400">Backend connecting…</span>
                </div>
            )}

            {/* GPU — always shown if data is present. ml-auto pushes it to
                the right edge. */}
            <div className="flex items-center gap-2 text-xs ml-auto">
                <Cpu size={14} className="text-gray-400" />
                <span className="text-gray-500 font-medium">GPU</span>
                {hasGpu ? (
                    <>
                        <span className="text-gray-300 tabular-nums">
                            {fmtMB(gpu!.used_mb)} / {fmtMB(gpu!.total_mb)}
                        </span>
                        <div className="w-24 h-1.5 bg-gray-800 rounded overflow-hidden">
                            <div
                                className={`h-full transition-all ${gpuPct > 90 ? "bg-red-500" : gpuPct > 70 ? "bg-amber-500" : "bg-green-500"}`}
                                style={{ width: `${gpuPct}%` }}
                            />
                        </div>
                        <span className="text-gray-500 tabular-nums w-9 text-right">{gpuPct}%</span>
                    </>
                ) : (
                    <span className="text-gray-600 tabular-nums">— / —</span>
                )}
            </div>
        </footer>
    );
}
