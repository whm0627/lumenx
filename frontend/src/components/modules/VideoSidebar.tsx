"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings2, List, RefreshCw, ChevronDown, ChevronUp, Mic, Music, VolumeX, Wand2 } from "lucide-react";
import VideoQueue from "./VideoQueue";
import { VideoTask, api } from "@/lib/api";
import { I2V_MODELS, DurationConfig, ModelParamSupport, VideoParams, GRID_COLS_CLASS } from "@/store/projectStore";

interface VideoSidebarProps {
    tasks: VideoTask[];
    onRemix: (task: VideoTask) => void;
    params: VideoParams;
    setParams: (params: VideoParams) => void;
}

export default function VideoSidebar({ tasks, onRemix, params, setParams }: VideoSidebarProps) {
    const [activeTab, setActiveTab] = useState<"settings" | "queue">("settings");
    const [isUploadingAudio, setIsUploadingAudio] = useState(false);
    const audioInputRef = useRef<HTMLInputElement>(null);
    const [showNegative, setShowNegative] = useState(false);

    const currentModelConfig = I2V_MODELS.find(m => m.id === params.model);
    const modelParams: ModelParamSupport = currentModelConfig?.params ?? {};

    const updateParam = (key: string, value: any) => {
        const newParams = { ...params, [key]: value };
        // When model changes, clamp duration and reset model-specific params
        if (key === "model") {
            const newModelConfig = I2V_MODELS.find(m => m.id === value);
            if (newModelConfig?.duration) {
                const dc = newModelConfig.duration;
                if (dc.type === 'fixed') {
                    newParams.duration = dc.value;
                } else if (dc.type === 'slider') {
                    if (newParams.duration < dc.min || newParams.duration > dc.max) {
                        newParams.duration = dc.default;
                    }
                } else if (dc.type === 'buttons') {
                    if (!dc.options.includes(newParams.duration)) {
                        newParams.duration = dc.default;
                    }
                }
            }
            // Reset model-specific params to defaults
            const np = newModelConfig?.params ?? {};
            newParams.resolution = np.resolution?.default ?? "720p";
            newParams.promptExtend = !!np.promptExtend;
            newParams.negativePrompt = "";
            newParams.shotType = "single";
            newParams.generateAudio = false;
            newParams.audioUrl = "";
            // Kling defaults
            newParams.mode = np.mode?.default ?? "std";
            newParams.sound = false;
            newParams.cfgScale = np.cfgScale?.default ?? 0.5;
            // Vidu defaults
            newParams.viduAudio = true;
            newParams.movementAmplitude = np.movementAmplitude?.default ?? "auto";
        }
        setParams(newParams);
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploadingAudio(true);
        try {
            const res = await api.uploadFile(file);
            updateParam("audioUrl", res.url);
            setAudioMode("custom");
        } catch (error) {
            console.error("Audio upload failed:", error);
        } finally {
            setIsUploadingAudio(false);
            // Reset input
            if (audioInputRef.current) audioInputRef.current.value = "";
        }
    };

    // Audio Mode Logic
    const audioMode = params.audioUrl ? "custom" : params.generateAudio ? "ai" : "mute";
    const setAudioMode = (mode: "mute" | "ai" | "custom") => {
        if (mode === "mute") {
            setParams({ ...params, generateAudio: false, audioUrl: "" });
        } else if (mode === "ai") {
            setParams({ ...params, generateAudio: true, audioUrl: "" });
        } else {
            // Custom / Sound Driven
            setParams({ ...params, generateAudio: false });
            // Trigger upload if no URL exists
            if (!params.audioUrl && audioInputRef.current) {
                audioInputRef.current.click();
            }
        }
    };

    return (
        <div className="h-full flex flex-col bg-black/40 backdrop-blur-sm border-l border-white/5">
            <input
                type="file"
                ref={audioInputRef}
                className="hidden"
                accept="audio/*"
                onChange={handleFileUpload}
            />
            {/* Tab Navigation */}
            <div className="flex border-b border-white/5">
                <button
                    onClick={() => setActiveTab("settings")}
                    className={`flex-1 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${activeTab === "settings"
                        ? "text-white border-b-2 border-primary bg-white/5"
                        : "text-gray-500 hover:text-gray-300 hover:bg-white/5"
                        }`}
                >
                    <Settings2 size={16} />
                    Motion Params
                </button>
                <button
                    onClick={() => setActiveTab("queue")}
                    className={`flex-1 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${activeTab === "queue"
                        ? "text-white border-b-2 border-primary bg-white/5"
                        : "text-gray-500 hover:text-gray-300 hover:bg-white/5"
                        }`}
                >
                    <List size={16} />
                    Queue
                    {tasks.filter(t => t.status === "pending" || t.status === "processing").length > 0 && (
                        <span className="bg-primary text-white text-[10px] px-1.5 rounded-full">
                            {tasks.filter(t => t.status === "pending" || t.status === "processing").length}
                        </span>
                    )}
                </button>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden relative">
                <AnimatePresence mode="wait">
                    {activeTab === "settings" ? (
                        <motion.div
                            key="settings"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            className="absolute inset-0 overflow-y-auto custom-scrollbar p-6 space-y-8"
                        >
                            {/* 1. Basic Settings */}
                            <section className="space-y-4">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                                    <div className="w-1 h-3 bg-primary rounded-full" />
                                    Basic Settings
                                </h3>

                                {/* Model Selection - R2V mode: only Wan 2.6 is selectable */}
                                <div>
                                    <label className="block text-xs text-gray-400 mb-2">
                                        Model (模型)
                                        {params.generationMode === "r2v" && (
                                            <span className="text-purple-400 ml-2">(R2V仅支持 Wan 2.6)</span>
                                        )}
                                    </label>
                                    <div className="space-y-2">
                                        {I2V_MODELS.map((model) => {
                                            const isR2VMode = params.generationMode === "r2v";
                                            const isWan26 = model.id === "wan2.6-i2v";
                                            const isDisabled = isR2VMode && !isWan26;
                                            const isSelected = isR2VMode ? isWan26 : params.model === model.id;

                                            return (
                                                <button
                                                    key={model.id}
                                                    onClick={() => {
                                                        if (isDisabled) return;
                                                        updateParam("model", model.id);
                                                        // Flip VIDEO_PROVIDER + (re)trigger local load when
                                                        // user picks the local card; flip back to cloud
                                                        // for any other selection. Mirrors how the image
                                                        // local card flips IMAGE_PROVIDER in ModelSettingsModal.
                                                        const isLocal = model.id === "wan2.2-s2v-14b-local" || model.id === "wan2.2-ti2v-local";
                                                        api.saveEnvConfig({
                                                            VIDEO_PROVIDER: isLocal ? "local" : "wanx",
                                                        }).catch(() => { });
                                                        if (isLocal) {
                                                            api.loadLocalVideo().catch(() => { });
                                                        }
                                                    }}
                                                    disabled={isDisabled}
                                                    className={`w-full flex items-center justify-between p-2.5 rounded-lg border transition-all text-left ${isSelected
                                                        ? 'border-primary/50 bg-primary/10'
                                                        : 'border-white/10 hover:border-white/20 bg-white/5'
                                                        } ${isDisabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                                                >
                                                    <div>
                                                        <span className="text-xs font-medium text-white">{model.name}</span>
                                                        <p className="text-[10px] text-gray-500">{model.description}</p>
                                                    </div>
                                                    {isSelected && (
                                                        <div className="w-2 h-2 bg-primary rounded-full" />
                                                    )}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Duration - Dynamic per model */}
                                {(() => {
                                    const durationConfig: DurationConfig = currentModelConfig?.duration ?? { type: 'buttons', options: [5, 10], default: 5 };

                                    if (durationConfig.type === 'fixed') {
                                        return (
                                            <div>
                                                <label className="block text-xs text-gray-400 mb-2">Duration (生成时长)</label>
                                                <div className="py-1.5 text-xs text-gray-500 bg-white/5 rounded-lg text-center border border-transparent">
                                                    {durationConfig.value}s (固定)
                                                </div>
                                            </div>
                                        );
                                    }

                                    if (durationConfig.type === 'slider') {
                                        return (
                                            <div>
                                                <label className="block text-xs text-gray-400 mb-2">
                                                    Duration (生成时长) <span className="text-primary font-medium">{params.duration}s</span>
                                                </label>
                                                <input
                                                    type="range"
                                                    min={durationConfig.min}
                                                    max={durationConfig.max}
                                                    step={durationConfig.step}
                                                    value={params.duration}
                                                    onChange={(e) => updateParam("duration", parseInt(e.target.value))}
                                                    className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer accent-primary [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:shadow-lg"
                                                />
                                                <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                                                    <span>{durationConfig.min}s</span>
                                                    <span>{durationConfig.max}s</span>
                                                </div>
                                            </div>
                                        );
                                    }

                                    // buttons
                                    return (
                                        <div>
                                            <label className="block text-xs text-gray-400 mb-2">Duration (生成时长)</label>
                                            <div className={`grid ${GRID_COLS_CLASS[durationConfig.options.length] ?? 'grid-cols-3'} gap-2`}>
                                                {durationConfig.options.map(dur => (
                                                    <button
                                                        key={dur}
                                                        onClick={() => updateParam("duration", dur)}
                                                        className={`py-1.5 text-xs rounded-lg border transition-all ${params.duration === dur
                                                            ? "bg-primary/20 border-primary text-primary"
                                                            : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                            }`}
                                                    >
                                                        {dur}s
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    );
                                })()}

                                {/* Shot Type - Only when model supports it and promptExtend is enabled */}
                                {modelParams.shotType && (
                                    <div>
                                        <label className="block text-xs text-gray-400 mb-2">
                                            Shot Type (镜头类型)
                                            {!params.promptExtend && (
                                                <span className="text-yellow-500 ml-2">(需开启智能扩写)</span>
                                            )}
                                        </label>
                                        <div className="grid grid-cols-2 gap-2">
                                            <button
                                                onClick={() => updateParam("shotType", "single")}
                                                disabled={!params.promptExtend}
                                                className={`py-2 text-xs rounded-lg border transition-all flex flex-col items-center gap-1 ${params.shotType === "single"
                                                    ? "bg-primary/20 border-primary text-primary"
                                                    : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                    } ${!params.promptExtend ? 'opacity-50 cursor-not-allowed' : ''}`}
                                            >
                                                <span className="font-medium">Single</span>
                                                <span className="text-[10px] text-gray-500">单镜头</span>
                                            </button>
                                            <button
                                                onClick={() => updateParam("shotType", "multi")}
                                                disabled={!params.promptExtend}
                                                className={`py-2 text-xs rounded-lg border transition-all flex flex-col items-center gap-1 ${params.shotType === "multi"
                                                    ? "bg-primary/20 border-primary text-primary"
                                                    : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                    } ${!params.promptExtend ? 'opacity-50 cursor-not-allowed' : ''}`}
                                            >
                                                <span className="font-medium">Multi</span>
                                                <span className="text-[10px] text-gray-500">多镜头叙事</span>
                                            </button>
                                        </div>
                                        <p className="text-[10px] text-gray-600 mt-1.5">
                                            多镜头模式会生成包含多个切换镜头的叙事视频
                                        </p>
                                    </div>
                                )}

                                {/* Kling: Mode (std/pro) */}
                                {modelParams.mode && (
                                    <div>
                                        <label className="block text-xs text-gray-400 mb-2">Mode (生成模式)</label>
                                        <div className="grid grid-cols-2 gap-2">
                                            {modelParams.mode.options.map(opt => (
                                                <button
                                                    key={opt}
                                                    onClick={() => updateParam("mode", opt)}
                                                    className={`py-1.5 text-xs rounded-lg border transition-all ${params.mode === opt
                                                        ? "bg-primary/20 border-primary text-primary"
                                                        : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                        }`}
                                                >
                                                    {opt.toUpperCase()}
                                                </button>
                                            ))}
                                        </div>
                                        <p className="text-[10px] text-gray-600 mt-1.5">
                                            Pro 模式质量更高但耗时更长
                                        </p>
                                    </div>
                                )}
                            </section>

                            <div className="w-full h-px bg-white/5" />

                            {/* 2. Quality & Specs */}
                            <section className="space-y-4">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                                    <div className="w-1 h-3 bg-blue-500 rounded-full" />
                                    Quality & Specs
                                </h3>

                                {/* Resolution - only when model supports it */}
                                {modelParams.resolution && (
                                    <div>
                                        <label className="block text-xs text-gray-400 mb-2">Resolution (画质)</label>
                                        <div className={`grid ${GRID_COLS_CLASS[modelParams.resolution.options.length] ?? 'grid-cols-3'} gap-2`}>
                                            {modelParams.resolution.options.map(res => (
                                                <button
                                                    key={res}
                                                    onClick={() => updateParam("resolution", res)}
                                                    className={`py-1.5 text-xs rounded-lg border transition-all ${params.resolution === res
                                                        ? "bg-blue-500/20 border-blue-500 text-blue-500"
                                                        : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                        }`}
                                                >
                                                    {res}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Batch Size */}
                                <div>
                                    <label className="block text-xs text-gray-400 mb-2">Batch Size (生成数量)</label>
                                    <div className="grid grid-cols-3 gap-2">
                                        {[1, 2, 4].map(size => (
                                            <button
                                                key={size}
                                                onClick={() => updateParam("batchSize", size)}
                                                className={`py-1.5 text-xs rounded-lg border transition-all ${params.batchSize === size
                                                    ? "bg-blue-500/20 border-blue-500 text-blue-500"
                                                    : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                    }`}
                                            >
                                                {size}x
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Kling: CFG Scale */}
                                {modelParams.cfgScale && (
                                    <div>
                                        <label className="block text-xs text-gray-400 mb-2">
                                            CFG Scale (创意度) <span className="text-blue-500 font-medium">{params.cfgScale.toFixed(1)}</span>
                                        </label>
                                        <input
                                            type="range"
                                            min={modelParams.cfgScale.min}
                                            max={modelParams.cfgScale.max}
                                            step={modelParams.cfgScale.step}
                                            value={params.cfgScale}
                                            onChange={(e) => updateParam("cfgScale", parseFloat(e.target.value))}
                                            className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer accent-blue-500 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:shadow-lg"
                                        />
                                        <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                                            <span>{modelParams.cfgScale.min} (自由)</span>
                                            <span>{modelParams.cfgScale.max} (严格)</span>
                                        </div>
                                    </div>
                                )}

                                {/* Vidu: Movement Amplitude */}
                                {modelParams.movementAmplitude && (
                                    <div>
                                        <label className="block text-xs text-gray-400 mb-2">Movement Amplitude (运动幅度)</label>
                                        <div className={`grid ${GRID_COLS_CLASS[modelParams.movementAmplitude.options.length] ?? 'grid-cols-4'} gap-2`}>
                                            {modelParams.movementAmplitude.options.map(opt => (
                                                <button
                                                    key={opt}
                                                    onClick={() => updateParam("movementAmplitude", opt)}
                                                    className={`py-1.5 text-xs rounded-lg border transition-all capitalize ${params.movementAmplitude === opt
                                                        ? "bg-blue-500/20 border-blue-500 text-blue-500"
                                                        : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                        }`}
                                                >
                                                    {opt === 'auto' ? 'Auto' : opt === 'small' ? 'S' : opt === 'medium' ? 'M' : 'L'}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </section>

                            <div className="w-full h-px bg-white/5" />

                            {/* 3. Creative & Audio */}
                            <section className="space-y-4">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                                    <div className="w-1 h-3 bg-purple-500 rounded-full" />
                                    Creative & Audio
                                </h3>

                                {/* Prompt Enhancer - only when model supports it */}
                                {modelParams.promptExtend && (
                                    <div className="flex items-center justify-between">
                                        <label className="text-xs text-gray-400 flex items-center gap-2">
                                            <Wand2 size={12} />
                                            Prompt Enhancer (智能扩写)
                                        </label>
                                        <button
                                            onClick={() => updateParam("promptExtend", !params.promptExtend)}
                                            className={`w-10 h-5 rounded-full relative transition-colors ${params.promptExtend ? "bg-purple-500" : "bg-white/10"}`}
                                        >
                                            <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${params.promptExtend ? "left-6" : "left-1"}`} />
                                        </button>
                                    </div>
                                )}

                                {/* Wan Audio Settings (三模式) - only when model supports it */}
                                {modelParams.audio && (
                                    <div>
                                        <label className="block text-xs text-gray-400 mb-2">
                                            Audio Settings (音频)
                                        </label>
                                        <div className="grid grid-cols-3 gap-2 mb-2">
                                            <button
                                                onClick={() => setAudioMode("mute")}
                                                className={`py-1.5 text-xs rounded-lg border flex items-center justify-center gap-1 transition-all ${audioMode === "mute"
                                                    ? "bg-purple-500/20 border-purple-500 text-purple-500"
                                                    : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                    }`}
                                            >
                                                <VolumeX size={12} /> Mute
                                            </button>
                                            <button
                                                onClick={() => setAudioMode("ai")}
                                                className={`py-1.5 text-xs rounded-lg border flex items-center justify-center gap-1 transition-all ${audioMode === "ai"
                                                    ? "bg-purple-500/20 border-purple-500 text-purple-500"
                                                    : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                    }`}
                                            >
                                                <Mic size={12} /> AI Sound
                                            </button>
                                            <button
                                                onClick={() => setAudioMode("custom")}
                                                className={`py-1.5 text-xs rounded-lg border flex items-center justify-center gap-1 transition-all ${audioMode === "custom"
                                                    ? "bg-purple-500/20 border-purple-500 text-purple-500"
                                                    : "bg-white/5 border-transparent text-gray-400 hover:bg-white/10"
                                                    }`}
                                            >
                                                <Music size={12} /> Sound Driven
                                            </button>
                                        </div>
                                        {audioMode === "custom" && (
                                            <div className="relative">
                                                <input
                                                    type="text"
                                                    value={params.audioUrl || ""}
                                                    readOnly
                                                    placeholder={isUploadingAudio ? "Uploading..." : "Click to upload audio"}
                                                    onClick={() => audioInputRef.current?.click()}
                                                    className="w-full bg-white/5 border border-white/10 rounded-lg py-1.5 px-2 text-xs text-white focus:border-purple-500 focus:outline-none cursor-pointer"
                                                />
                                                {params.audioUrl && (
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            updateParam("audioUrl", "");
                                                            setAudioMode("mute");
                                                        }}
                                                        className="absolute right-2 top-1.5 text-gray-500 hover:text-white"
                                                    >
                                                        <VolumeX size={12} />
                                                    </button>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Kling: Sound on/off */}
                                {modelParams.sound && (
                                    <div className="flex items-center justify-between">
                                        <label className="text-xs text-gray-400 flex items-center gap-2">
                                            <Mic size={12} />
                                            Sound (AI音效)
                                        </label>
                                        <button
                                            onClick={() => updateParam("sound", !params.sound)}
                                            className={`w-10 h-5 rounded-full relative transition-colors ${params.sound ? "bg-purple-500" : "bg-white/10"}`}
                                        >
                                            <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${params.sound ? "left-6" : "left-1"}`} />
                                        </button>
                                    </div>
                                )}

                                {/* Vidu: Audio on/off */}
                                {modelParams.viduAudio && (
                                    <div className="flex items-center justify-between">
                                        <label className="text-xs text-gray-400 flex items-center gap-2">
                                            <Mic size={12} />
                                            Audio Output (音视频直出)
                                        </label>
                                        <button
                                            onClick={() => updateParam("viduAudio", !params.viduAudio)}
                                            className={`w-10 h-5 rounded-full relative transition-colors ${params.viduAudio ? "bg-purple-500" : "bg-white/10"}`}
                                        >
                                            <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${params.viduAudio ? "left-6" : "left-1"}`} />
                                        </button>
                                    </div>
                                )}

                                {/* Negative Prompt - only when model supports it */}
                                {modelParams.negativePrompt && (
                                    <div>
                                        <button
                                            onClick={() => setShowNegative(!showNegative)}
                                            className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-2"
                                        >
                                            {showNegative ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
                                            Negative Prompt (负向提示词)
                                        </button>
                                        <AnimatePresence>
                                            {showNegative && (
                                                <motion.div
                                                    initial={{ height: 0, opacity: 0 }}
                                                    animate={{ height: "auto", opacity: 1 }}
                                                    exit={{ height: 0, opacity: 0 }}
                                                    className="overflow-hidden"
                                                >
                                                    <textarea
                                                        value={params.negativePrompt || ""}
                                                        onChange={(e) => updateParam("negativePrompt", e.target.value)}
                                                        placeholder="Low quality, blurry, distorted..."
                                                        className="w-full h-20 bg-white/5 border border-white/10 rounded-lg p-2 text-xs text-white focus:border-purple-500 focus:outline-none resize-none"
                                                    />
                                                </motion.div>
                                            )}
                                        </AnimatePresence>
                                    </div>
                                )}
                            </section>

                            <div className="w-full h-px bg-white/5" />

                            {/* 4. Advanced / Effects */}
                            <section className="space-y-4">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                                    <div className="w-1 h-3 bg-orange-500 rounded-full" />
                                    Advanced
                                </h3>

                                {/* Seed - only when model supports it */}
                                {modelParams.seed && (
                                    <div>
                                        <label className="block text-xs text-gray-400 mb-2">Seed (随机种子)</label>
                                        <div className="relative">
                                            <input
                                                type="number"
                                                value={params.seed ?? ""}
                                                onChange={(e) => updateParam("seed", e.target.value ? parseInt(e.target.value) : undefined)}
                                                placeholder="Random (-1)"
                                                className="w-full bg-white/5 border border-white/10 rounded-lg py-1.5 pl-2 pr-8 text-xs text-white focus:border-orange-500 focus:outline-none [&::-webkit-inner-spin-button]:appearance-none"
                                            />
                                            <button
                                                onClick={() => updateParam("seed", Math.floor(Math.random() * 2147483647))}
                                                className="absolute right-2 top-1.5 text-gray-500 hover:text-white"
                                                title="Randomize"
                                            >
                                                <RefreshCw size={12} />
                                            </button>
                                        </div>
                                    </div>
                                )}


                            </section>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="queue"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                            className="absolute inset-0"
                        >
                            <VideoQueue tasks={tasks} onRemix={onRemix} />
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
