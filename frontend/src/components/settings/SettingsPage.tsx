"use client";

import { useState, useEffect } from "react";
import { Save, Loader2, Key, ChevronDown, ChevronRight, Settings, MessageSquareCode } from "lucide-react";
import { api, type EnvConfigPayload, type ProviderMode } from "@/lib/api";
import { T2I_MODELS, I2I_MODELS, I2V_MODELS, ASPECT_RATIOS } from "@/store/projectStore";
import { Image, Video, Layout, Check, User, Building, Box } from "lucide-react";
import LocalLLMPanel from "./LocalLLMPanel";

type LLMProvider = "dashscope" | "openai" | "local";

type EnvConfig = EnvConfigPayload & {
  DASHSCOPE_API_KEY: string;
  ALIBABA_CLOUD_ACCESS_KEY_ID: string;
  ALIBABA_CLOUD_ACCESS_KEY_SECRET: string;
  OSS_BUCKET_NAME: string;
  OSS_ENDPOINT: string;
  OSS_BASE_PATH: string;
  KLING_PROVIDER_MODE: ProviderMode;
  VIDU_PROVIDER_MODE: ProviderMode;
  PIXVERSE_PROVIDER_MODE: ProviderMode;
  KLING_ACCESS_KEY: string;
  KLING_SECRET_KEY: string;
  VIDU_API_KEY: string;
  endpoint_overrides: Record<string, string>;
};

const ENDPOINT_PROVIDERS = [
  { key: "DASHSCOPE_BASE_URL", label: "DashScope", placeholder: "https://dashscope.aliyuncs.com" },
  { key: "KLING_BASE_URL", label: "Kling", placeholder: "https://api-beijing.klingai.com/v1" },
  { key: "VIDU_BASE_URL", label: "Vidu", placeholder: "https://api.vidu.cn/ent/v2" },
];

const DEFAULT_CONFIG: EnvConfig = {
  DASHSCOPE_API_KEY: "",
  ALIBABA_CLOUD_ACCESS_KEY_ID: "",
  ALIBABA_CLOUD_ACCESS_KEY_SECRET: "",
  OSS_BUCKET_NAME: "",
  OSS_ENDPOINT: "",
  OSS_BASE_PATH: "",
  KLING_PROVIDER_MODE: "dashscope",
  VIDU_PROVIDER_MODE: "dashscope",
  PIXVERSE_PROVIDER_MODE: "dashscope",
  KLING_ACCESS_KEY: "",
  KLING_SECRET_KEY: "",
  VIDU_API_KEY: "",
  endpoint_overrides: {},
};

const normalizeProviderMode = (mode?: string): ProviderMode => (mode === "vendor" ? "vendor" : "dashscope");

const normalizeEnvConfig = (existing: EnvConfig, data?: EnvConfigPayload): EnvConfig => ({
  ...existing,
  ...data,
  KLING_PROVIDER_MODE: normalizeProviderMode(data?.KLING_PROVIDER_MODE ?? existing.KLING_PROVIDER_MODE),
  VIDU_PROVIDER_MODE: normalizeProviderMode(data?.VIDU_PROVIDER_MODE ?? existing.VIDU_PROVIDER_MODE),
  PIXVERSE_PROVIDER_MODE: normalizeProviderMode(data?.PIXVERSE_PROVIDER_MODE ?? existing.PIXVERSE_PROVIDER_MODE),
  endpoint_overrides: data?.endpoint_overrides ?? existing.endpoint_overrides ?? {},
});

const getValidationErrors = (env: EnvConfig): string[] => {
  const errors: string[] = [];

  if (!env.DASHSCOPE_API_KEY?.trim()) {
    errors.push("DashScope API Key");
  }
  if (env.KLING_PROVIDER_MODE === "vendor") {
    if (!env.KLING_ACCESS_KEY?.trim()) {
      errors.push("Kling Access Key (vendor mode)");
    }
    if (!env.KLING_SECRET_KEY?.trim()) {
      errors.push("Kling Secret Key (vendor mode)");
    }
  }
  if (env.VIDU_PROVIDER_MODE === "vendor" && !env.VIDU_API_KEY?.trim()) {
    errors.push("Vidu API Key (vendor mode)");
  }

  return errors;
};

const LS_KEY_MODEL = "lumenx_default_model_settings";
const LS_KEY_PROMPT = "lumenx_default_prompt_config";

interface DefaultModelSettings {
  t2i_model: string;
  i2i_model: string;
  i2v_model: string;
  character_aspect_ratio: string;
  scene_aspect_ratio: string;
  prop_aspect_ratio: string;
  storyboard_aspect_ratio: string;
}

interface DefaultPromptConfig {
  storyboard_polish: string;
  video_polish: string;
  r2v_polish: string;
}

function loadFromLS<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export default function SettingsPage() {
  // ── API Config ──
  const [config, setConfig] = useState<EnvConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [endpointsOpen, setEndpointsOpen] = useState(false);

  // ── Default Model Settings ──
  const [modelSettings, setModelSettings] = useState<DefaultModelSettings>(() =>
    loadFromLS(LS_KEY_MODEL, {
      t2i_model: "wan2.5-t2i-preview",
      i2i_model: "wan2.5-i2i-preview",
      i2v_model: "wan2.5-i2v-preview",
      character_aspect_ratio: "9:16",
      scene_aspect_ratio: "16:9",
      prop_aspect_ratio: "1:1",
      storyboard_aspect_ratio: "16:9",
    })
  );

  // ── Default Prompt Config ──
  const [promptConfig, setPromptConfig] = useState<DefaultPromptConfig>(() =>
    loadFromLS(LS_KEY_PROMPT, { storyboard_polish: "", video_polish: "", r2v_polish: "" })
  );

  // ── LLM Provider selector ──
  const [llmProvider, setLlmProvider] = useState<LLMProvider>("dashscope");

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const data = await api.getEnvConfig();
      setConfig((prev) => normalizeEnvConfig(prev, data));
      const provider = (data as any).LLM_PROVIDER;
      if (provider === "openai" || provider === "local") {
        setLlmProvider(provider);
      } else {
        setLlmProvider("dashscope");
      }
    } catch {
      setLoadError("Failed to load configuration. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveApiConfig = async () => {
    const errors = getValidationErrors(config);
    if (errors.length > 0) {
      alert(`Please fill in required fields:\n- ${errors.join("\n- ")}`);
      return;
    }

    setSaving(true);
    try {
      await api.saveEnvConfig({ ...config, LLM_PROVIDER: llmProvider } as EnvConfigPayload);
      alert("Configuration saved successfully!");
    } catch {
      alert("Failed to save configuration.");
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (key: keyof EnvConfig, value: string) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleEndpointChange = (envKey: string, value: string) => {
    setConfig((prev) => ({
      ...prev,
      endpoint_overrides: { ...prev.endpoint_overrides, [envKey]: value },
    }));
  };

  const handleSaveModelDefaults = () => {
    localStorage.setItem(LS_KEY_MODEL, JSON.stringify(modelSettings));
    alert("Default model settings saved!");
  };

  const handleSavePromptDefaults = () => {
    localStorage.setItem(LS_KEY_PROMPT, JSON.stringify(promptConfig));
    alert("Default prompt configuration saved!");
  };

  const inputClass =
    "w-full bg-black/30 border border-white/10 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-primary/50 transition-colors";
  const modeButtonClass = (active: boolean) =>
    `px-3 py-1.5 text-xs rounded-md border transition-colors ${active ? "border-amber-500/60 bg-amber-500/15 text-amber-200" : "border-white/10 bg-white/5 text-gray-400 hover:text-gray-200"}`;

  return (
    <div className="container mx-auto px-6 py-8 max-w-4xl space-y-8">
      <h1 className="text-2xl font-display font-bold text-white">设置</h1>

      {/* ── Section 1: API Configuration ── */}
      <section className="glass-panel rounded-xl p-6 space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-lg">
            <Key size={20} className="text-amber-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">API 配置</h2>
            <p className="text-xs text-gray-500">DashScope-first setup with optional OSS mirror and provider-direct routing</p>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={24} className="animate-spin text-amber-400" />
            <span className="ml-2 text-gray-400">Loading configuration...</span>
          </div>
        ) : loadError ? (
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-sm text-red-300">
            {loadError}
          </div>
        ) : (
          <>
            <div>
              <label className="flex items-center justify-between text-sm font-medium text-gray-300 mb-2">
                <span>DashScope API Key <span className="text-red-500">*</span></span>
                <span className="text-gray-600 font-normal text-xs">e.g. sk-xxx</span>
              </label>
              <input type="password" value={config.DASHSCOPE_API_KEY} onChange={(e) => handleChange("DASHSCOPE_API_KEY", e.target.value)} placeholder="Required for DashScope-first model routing" className={inputClass} />
            </div>

            <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-4">
              <p className="text-xs text-gray-400">Storage is local-first by default. These credentials are only needed when enabling OSS mirror.</p>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Alibaba Cloud Access Key ID</label>
                <input type="password" value={config.ALIBABA_CLOUD_ACCESS_KEY_ID} onChange={(e) => handleChange("ALIBABA_CLOUD_ACCESS_KEY_ID", e.target.value)} placeholder="Optional, for OSS mirror" className={inputClass} />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Alibaba Cloud Access Key Secret</label>
                <input type="password" value={config.ALIBABA_CLOUD_ACCESS_KEY_SECRET} onChange={(e) => handleChange("ALIBABA_CLOUD_ACCESS_KEY_SECRET", e.target.value)} placeholder="Optional, for OSS mirror" className={inputClass} />
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <h3 className="text-sm font-bold text-white mb-2">OSS Mirror (Optional)</h3>
              <p className="text-[10px] text-gray-500 mb-4">Generated assets always save locally first. Configure OSS to keep an optional cloud mirror.</p>
              <div className="space-y-4">
                <div>
                  <label className="flex items-center justify-between text-sm font-medium text-gray-300 mb-2">
                    <span>OSS Bucket Name</span>
                  </label>
                  <input type="text" value={config.OSS_BUCKET_NAME} onChange={(e) => handleChange("OSS_BUCKET_NAME", e.target.value)} placeholder="your_bucket_name (optional)" className={inputClass} />
                </div>
                <div>
                  <label className="flex items-center justify-between text-sm font-medium text-gray-300 mb-2">
                    <span>OSS Endpoint</span>
                  </label>
                  <input type="text" value={config.OSS_ENDPOINT} onChange={(e) => handleChange("OSS_ENDPOINT", e.target.value)} placeholder="oss-cn-beijing.aliyuncs.com (optional)" className={inputClass} />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">OSS Base Path</label>
                  <input type="text" value={config.OSS_BASE_PATH} onChange={(e) => handleChange("OSS_BASE_PATH", e.target.value)} placeholder="lumenx" className={inputClass} />
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <h3 className="text-sm font-bold text-white mb-4">LLM Provider (text generation)</h3>
              <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-4">
                <div className="flex flex-wrap gap-2">
                  <button type="button" onClick={() => setLlmProvider("dashscope")} className={modeButtonClass(llmProvider === "dashscope")}>
                    DashScope
                  </button>
                  <button type="button" onClick={() => setLlmProvider("openai")} className={modeButtonClass(llmProvider === "openai")}>
                    OpenAI-Compatible
                  </button>
                  <button type="button" onClick={() => setLlmProvider("local")} className={modeButtonClass(llmProvider === "local")}>
                    Local (HuggingFace)
                  </button>
                </div>
                {llmProvider === "dashscope" && (
                  <p className="text-xs text-gray-500">
                    Uses DASHSCOPE_API_KEY (default). Configure your key in the API Keys section above.
                  </p>
                )}
                {llmProvider === "openai" && (
                  <p className="text-xs text-gray-500">
                    Configure OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL in .env, then save and restart.
                  </p>
                )}
                {llmProvider === "local" && <LocalLLMPanel />}
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <h3 className="text-sm font-bold text-white mb-4">Kling Provider</h3>
              <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-4">
                <div className="flex flex-wrap gap-2">
                  <button type="button" onClick={() => handleChange("KLING_PROVIDER_MODE", "dashscope")} className={modeButtonClass(config.KLING_PROVIDER_MODE === "dashscope")}>
                    DashScope
                  </button>
                  <button type="button" onClick={() => handleChange("KLING_PROVIDER_MODE", "vendor")} className={modeButtonClass(config.KLING_PROVIDER_MODE === "vendor")}>
                    Vendor Direct
                  </button>
                </div>
                <p className="text-xs text-gray-500">
                  DashScope mode uses your DashScope API key. Vendor-direct mode requires Kling Access Key and Secret Key.
                </p>
                {config.KLING_PROVIDER_MODE === "vendor" && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Kling Access Key <span className="text-red-500">*</span></label>
                      <input type="password" value={config.KLING_ACCESS_KEY} onChange={(e) => handleChange("KLING_ACCESS_KEY", e.target.value)} placeholder="Kling API Access Key" className={inputClass} />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Kling Secret Key <span className="text-red-500">*</span></label>
                      <input type="password" value={config.KLING_SECRET_KEY} onChange={(e) => handleChange("KLING_SECRET_KEY", e.target.value)} placeholder="Kling API Secret Key" className={inputClass} />
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <h3 className="text-sm font-bold text-white mb-4">Vidu Provider</h3>
              <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-4">
                <div className="flex flex-wrap gap-2">
                  <button type="button" onClick={() => handleChange("VIDU_PROVIDER_MODE", "dashscope")} className={modeButtonClass(config.VIDU_PROVIDER_MODE === "dashscope")}>
                    DashScope
                  </button>
                  <button type="button" onClick={() => handleChange("VIDU_PROVIDER_MODE", "vendor")} className={modeButtonClass(config.VIDU_PROVIDER_MODE === "vendor")}>
                    Vendor Direct
                  </button>
                </div>
                <p className="text-xs text-gray-500">
                  DashScope mode uses your DashScope API key. Vendor-direct mode requires a Vidu API key.
                </p>
                {config.VIDU_PROVIDER_MODE === "vendor" && (
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Vidu API Key <span className="text-red-500">*</span></label>
                    <input type="password" value={config.VIDU_API_KEY} onChange={(e) => handleChange("VIDU_API_KEY", e.target.value)} placeholder="Vidu API Key" className={inputClass} />
                  </div>
                )}
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <button type="button" onClick={() => setEndpointsOpen(!endpointsOpen)} className="flex items-center gap-2 text-sm font-medium text-gray-400 hover:text-gray-200 transition-colors">
                {endpointsOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                Advanced: API Endpoints
              </button>
              {endpointsOpen && (
                <div className="mt-4 space-y-4">
                  <p className="text-xs text-gray-500">Custom API endpoint URLs. Leave empty to use defaults. Overrides are preserved regardless of provider mode.</p>
                  {ENDPOINT_PROVIDERS.map(({ key, label, placeholder }) => (
                    <div key={key}>
                      <label className="flex items-center justify-between text-sm font-medium text-gray-300 mb-2">
                        <span>{label} Base URL</span>
                        <span className="text-gray-600 font-normal text-xs">{placeholder}</span>
                      </label>
                      <input type="text" value={config.endpoint_overrides[key] || ""} onChange={(e) => handleEndpointChange(key, e.target.value)} placeholder={placeholder} className={inputClass + " text-sm"} />
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex justify-end">
              <button
                onClick={handleSaveApiConfig}
                disabled={saving || loading}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white text-sm font-medium rounded-lg transition-all disabled:opacity-50"
              >
                {saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                {saving ? "Saving..." : "Save Configuration"}
              </button>
            </div>
          </>
        )}
      </section>

      {/* ── Section 2: Default Model Settings ── */}
      <section className="glass-panel rounded-xl p-6 space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg">
            <Settings size={20} className="text-blue-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">默认模型设置</h2>
            <p className="text-xs text-gray-500">Default models and aspect ratios for new projects</p>
          </div>
        </div>

        <div className="space-y-5">
          <div className="flex items-center gap-2 text-sm font-bold text-white">
            <Image size={16} className="text-green-400" />
            <span>Text-to-Image Model</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {T2I_MODELS.map((model) => (
              <button
                key={model.id}
                onClick={() => setModelSettings((s) => ({ ...s, t2i_model: model.id }))}
                className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left ${modelSettings.t2i_model === model.id ? "border-green-500/50 bg-green-500/10" : "border-white/10 hover:border-white/20 bg-white/5"}`}
              >
                {modelSettings.t2i_model === model.id && <div className="absolute top-2 right-2"><Check size={14} className="text-green-400" /></div>}
                <span className="text-sm font-medium text-white">{model.name}</span>
                <span className="text-xs text-gray-500">{model.description}</span>
              </button>
            ))}
          </div>

          <div className="grid grid-cols-3 gap-4">
            {(
              [
                { key: "character_aspect_ratio" as const, label: "Character", icon: User },
                { key: "scene_aspect_ratio" as const, label: "Scene", icon: Building },
                { key: "prop_aspect_ratio" as const, label: "Prop", icon: Box },
              ] as const
            ).map(({ key, label, icon: Icon }) => (
              <div key={key} className="space-y-2">
                <div className="flex items-center gap-1 text-xs text-gray-400"><Icon size={12} /><label>{label}</label></div>
                <div className="space-y-1">
                  {ASPECT_RATIOS.map((ratio) => (
                    <button key={ratio.id} onClick={() => setModelSettings((s) => ({ ...s, [key]: ratio.id }))} className={`w-full flex flex-col items-center py-2 px-2 rounded border transition-all ${modelSettings[key] === ratio.id ? "border-green-500/50 bg-green-500/10" : "border-white/10 hover:border-white/20 bg-white/5"}`}>
                      <span className="text-xs font-medium text-white">{ratio.name}</span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-white/10 pt-4">
            <div className="flex items-center gap-2 text-sm font-bold text-white">
              <Layout size={16} className="text-blue-400" />
              <span>Storyboard (Image-to-Image)</span>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {I2I_MODELS.map((model) => (
                <button key={model.id} onClick={() => setModelSettings((s) => ({ ...s, i2i_model: model.id }))} className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left ${modelSettings.i2i_model === model.id ? "border-blue-500/50 bg-blue-500/10" : "border-white/10 hover:border-white/20 bg-white/5"}`}>
                  {modelSettings.i2i_model === model.id && <div className="absolute top-2 right-2"><Check size={14} className="text-blue-400" /></div>}
                  <span className="text-sm font-medium text-white">{model.name}</span>
                  <span className="text-xs text-gray-500">{model.description}</span>
                </button>
              ))}
            </div>
            <div className="mt-3 space-y-2">
              <label className="text-xs text-gray-400">Storyboard Aspect Ratio</label>
              <div className="grid grid-cols-3 gap-2">
                {ASPECT_RATIOS.map((ratio) => (
                  <button key={ratio.id} onClick={() => setModelSettings((s) => ({ ...s, storyboard_aspect_ratio: ratio.id }))} className={`flex flex-col items-center p-3 rounded-lg border transition-all ${modelSettings.storyboard_aspect_ratio === ratio.id ? "border-blue-500/50 bg-blue-500/10" : "border-white/10 hover:border-white/20 bg-white/5"}`}>
                    <span className="text-sm font-medium text-white">{ratio.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="border-t border-white/10 pt-4">
            <div className="flex items-center gap-2 text-sm font-bold text-white">
              <Video size={16} className="text-purple-400" />
              <span>Motion (Image-to-Video)</span>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {I2V_MODELS.map((model) => (
                <button key={model.id} onClick={() => setModelSettings((s) => ({ ...s, i2v_model: model.id }))} className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left ${modelSettings.i2v_model === model.id ? "border-purple-500/50 bg-purple-500/10" : "border-white/10 hover:border-white/20 bg-white/5"}`}>
                  {modelSettings.i2v_model === model.id && <div className="absolute top-2 right-2"><Check size={14} className="text-purple-400" /></div>}
                  <span className="text-sm font-medium text-white">{model.name}</span>
                  <span className="text-xs text-gray-500">{model.description}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <button onClick={handleSaveModelDefaults} className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white text-sm font-medium rounded-lg transition-all">
            <Save size={16} />
            Save Defaults
          </button>
        </div>
      </section>

      {/* ── Section 3: Default Prompt Config ── */}
      <section className="glass-panel rounded-xl p-6 space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <MessageSquareCode size={20} className="text-purple-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">默认提示词配置</h2>
            <p className="text-xs text-gray-500">Default system prompts for new projects (leave empty for built-in defaults)</p>
          </div>
        </div>

        {(
          [
            { key: "storyboard_polish" as const, label: "Storyboard Polish", desc: "System prompt for storyboard/image prompt polishing" },
            { key: "video_polish" as const, label: "Video I2V Polish", desc: "System prompt for Image-to-Video prompt polishing" },
            { key: "r2v_polish" as const, label: "Video R2V Polish", desc: "System prompt for Reference-to-Video prompt polishing" },
          ] as const
        ).map((section) => (
          <div key={section.key} className="space-y-2">
            <h3 className="text-sm font-bold text-white">{section.label}</h3>
            <p className="text-[10px] text-gray-500">{section.desc}</p>
            <textarea
              value={promptConfig[section.key]}
              onChange={(e) => setPromptConfig((prev) => ({ ...prev, [section.key]: e.target.value }))}
              placeholder="Leave empty to use system default..."
              className="w-full h-32 bg-black/30 border border-white/10 rounded-lg p-3 text-xs text-gray-300 resize-y focus:outline-none focus:border-purple-500/50 font-mono placeholder-gray-600"
            />
          </div>
        ))}

        <div className="flex justify-end">
          <button onClick={handleSavePromptDefaults} className="px-6 py-2 text-sm font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors flex items-center gap-2">
            <Save size={16} />
            Save Defaults
          </button>
        </div>
      </section>

      <div className="pb-8" />
    </div>
  );
}
