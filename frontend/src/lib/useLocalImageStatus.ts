"use client";

import { useEffect, useRef, useState } from "react";
import { api, type LocalImageStatus } from "./api";

/**
 * Subscribe to /img/local/status while `enabled` is true. Returns the
 * latest status (null until the first response). Polls only while
 * enabled — silent / no network when not generating.
 *
 * Used by every overlay that wants to show generation progress: the
 * AssetCard spinner, the VariantSelector spinner inside the workbench,
 * and the GlobalStatusFooter (which has its own polling, but could
 * migrate here).
 */
export function useLocalImageStatus(
    enabled: boolean,
    intervalMs: number = 1000,
): LocalImageStatus | null {
    const [status, setStatus] = useState<LocalImageStatus | null>(null);
    const aliveRef = useRef(true);

    useEffect(() => {
        aliveRef.current = true;
        return () => {
            aliveRef.current = false;
        };
    }, []);

    useEffect(() => {
        if (!enabled) return;
        let cancelled = false;
        const tick = async () => {
            if (cancelled || !aliveRef.current) return;
            try {
                const s = await api.getLocalImageStatus();
                if (!cancelled && aliveRef.current) setStatus(s);
            } catch {
                // Endpoint failure is non-fatal — overlay just shows the
                // last-known state (or stays as static "Generating...").
            }
        };
        tick(); // immediate first read so the user doesn't wait `intervalMs`
        const id = setInterval(tick, intervalMs);
        return () => {
            cancelled = true;
            clearInterval(id);
        };
    }, [enabled, intervalMs]);

    return status;
}

/**
 * Compact human-friendly label for a LocalImageStatus state. Used in
 * spinner overlays that have ~one line of vertical space.
 */
export function formatLocalImageStatus(s: LocalImageStatus | null): string {
    if (!s) return "Generating...";
    const pct = s.progress > 0 && s.progress < 1
        ? ` ${Math.round(s.progress * 100)}%`
        : "";
    switch (s.state) {
        case "DOWNLOADING": return `Downloading model${pct}`;
        case "LOADING": return `Loading to GPU${pct}`;
        case "GENERATING": return `Generating${pct}`;
        case "READY": return "Ready";
        case "ERROR": return `Error: ${s.error ?? "unknown"}`;
        default: return "Generating...";
    }
}
