"use client";

import { useRef, useState } from "react";
import ColorScale from "~/components/ColorScale";
import FeatureMap from "~/components/FeatureMap";
import Waveform from "~/components/Waveform";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Progress } from "~/components/ui/progress";

interface Prediction {
  class: string;
  confidence: number;
}
interface LayerData {
  shape: number[];
  values: number[][];
}
interface VisualizationData {
  [layerName: string]: LayerData;
}
interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}
interface ApiResponse {
  predictions: Prediction[];
  visualization: VisualizationData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

const ESC50_EMOJI_MAP: Record<string, string> = {
  dog: "🐕",
  rain: "🌧️",
  crying_baby: "👶",
  door_wood_knock: "🚪",
  helicopter: "🚁",
  rooster: "🐓",
  sea_waves: "🌊",
  sneezing: "🤧",
  mouse_click: "🖱️",
  chainsaw: "🪚",
  pig: "🐷",
  crackling_fire: "🔥",
  clapping: "👏",
  keyboard_typing: "⌨️",
  siren: "🚨",
  cow: "🐄",
  crickets: "🦗",
  breathing: "💨",
  door_wood_creaks: "🚪",
  car_horn: "📯",
  frog: "🐸",
  chirping_birds: "🐦",
  coughing: "😷",
  can_opening: "🥫",
  engine: "🚗",
  cat: "🐱",
  water_drops: "💧",
  footsteps: "👣",
  washing_machine: "🧺",
  train: "🚂",
  hen: "🐔",
  wind: "💨",
  laughing: "😂",
  vacuum_cleaner: "🧹",
  church_bells: "🔔",
  insects: "🦟",
  pouring_water: "🚰",
  brushing_teeth: "🪥",
  clock_alarm: "⏰",
  airplane: "✈️",
  sheep: "🐑",
  toilet_flush: "🚽",
  snoring: "😴",
  clock_tick: "⏱️",
  fireworks: "🎆",
  crow: "🐦‍⬛",
  thunderstorm: "⛈️",
  drinking_sipping: "🥤",
  glass_breaking: "🔨",
  hand_saw: "🪚",
};
const getEmojiForClass = (c: string) => ESC50_EMOJI_MAP[c] || "🔈";

function splitLayers(visualization: VisualizationData) {
  const main: [string, LayerData][] = [];
  const internals: Record<string, [string, LayerData][]> = {};
  for (const [name, data] of Object.entries(visualization)) {
    if (!name.includes(".")) main.push([name, data]);
    else {
      const [parent] = name.split(".");
      if (!parent) continue;
      (internals[parent] ||= []).push([name, data]);
    }
  }
  return { main, internals };
}

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFile = async (file: File | null) => {
    if (!file) return;
    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);
    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (s, b) => s + String.fromCharCode(b),
            "",
          ),
        );
        const response = await fetch(
          "https://shahbhavya7--audio-cnn-inference-audioclassifier-inference.modal.run/",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64String }),
          },
        );
        if (!response.ok) throw new Error(`API error ${response.status}`);
        const data: ApiResponse = await response.json();
        setVizData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsLoading(false);
      }
    };
    reader.onerror = () => {
      setError("Failed to read file");
      setIsLoading(false);
    };
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    handleFile(e.target.files?.[0] ?? null);
  const triggerFile = () => fileInputRef.current?.click();

  const { main, internals } = vizData
    ? splitLayers(vizData.visualization)
    : { main: [], internals: {} };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800 p-8 text-slate-100">
      <style jsx global>{`
        .visualizer-container,
        .waveform-wrapper,
        .featuremap-inner {
          position: relative;
          overflow: hidden;
        }
        .waveform-wrapper canvas,
        .waveform-wrapper svg {
          height: 180px;
          width: 100%;
          display: block;
          object-fit: contain;
        }
        .featuremap-scroll {
          max-height: 220px;
          overflow: auto;
          -webkit-overflow-scrolling: touch;
          padding-right: 6px;
        }
        .featuremap-card {
          display: flex;
          flex-direction: column;
          gap: 8px;
          overflow: hidden;
        }
        .feature-preview-chip {
          min-width: 0;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .spectrogram-reserved {
          position: relative;
          overflow: hidden;
          min-height: 260px;
        }
        @media (max-width: 768px) {
          .waveform-wrapper canvas,
          .waveform-wrapper svg {
            height: 120px;
          }
          .featuremap-scroll {
            max-height: 160px;
          }
        }
      `}</style>

      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Header */}
        <div className="mb-2 flex items-center justify-between lg:col-span-12">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-cyan-400 to-indigo-600 font-bold text-slate-900">
              AI
            </div>
            <div>
              <h1 className="text-2xl font-semibold">Resona</h1>
              <p className="text-sm text-slate-300">
                CNN Audio Visualizer → Inspect predictions, spectrogram & maps
              </p>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <aside className="space-y-6 lg:col-span-3">
          <Card className="border border-slate-700 bg-slate-800/80">
            <CardHeader>
              <CardTitle>Upload</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border-2 border-dashed border-slate-700 bg-slate-900/40 p-4 text-center">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav"
                  onChange={onFileChange}
                  className="hidden"
                />
                <div className="mb-3">
                  <Button onClick={triggerFile} className="px-4 py-2">
                    Choose file
                  </Button>
                </div>
                <div className="text-sm text-slate-300">
                  {isLoading ? "Analysing..." : fileName || "No file chosen"}
                </div>
                {error && (
                  <div className="mt-2 text-xs text-red-400">{error}</div>
                )}
              </div>
              <div className="mt-3 text-xs text-slate-400">
                Supported: <strong>WAV</strong> · Max ~10MB
              </div>
            </CardContent>
          </Card>

          <Card className="border border-slate-700 bg-slate-800/80">
            <CardHeader>
              <CardTitle>Top Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              {!vizData && (
                <div className="space-y-2">
                  <div className="h-3 w-3/4 animate-pulse rounded bg-slate-700" />
                  <div className="h-3 w-1/2 animate-pulse rounded bg-slate-700" />
                  <div className="h-3 w-2/3 animate-pulse rounded bg-slate-700" />
                </div>
              )}

              {vizData &&
                vizData.predictions.slice(0, 5).map((p, i) => (
                  <div key={p.class} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium text-slate-200">
                        {getEmojiForClass(p.class)}{" "}
                        <span className="ml-2">
                          {p.class.replaceAll("_", " ")}
                        </span>
                      </div>
                      <Badge className="bg-slate-700/60 text-slate-100">
                        {(p.confidence * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <Progress
                      value={p.confidence * 100}
                      className="h-2 rounded"
                    />
                  </div>
                ))}
            </CardContent>
          </Card>

          <Card className="border border-slate-700 bg-slate-800/80">
            <CardHeader>
              <CardTitle>Details</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-slate-300">
              <div>
                <strong>Model:</strong> AudioCNN v1
              </div>
              <div>
                <strong>Input:</strong> Mel-Spectrogram
              </div>
            </CardContent>
          </Card>
        </aside>

        {/* Main content */}
        <section className="space-y-6 lg:col-span-9">
          <Card className="border border-slate-700 bg-slate-800/70 p-4">
            <div className="flex items-start justify-between">
              <h2 className="text-lg font-medium">Input Spectrogram</h2>
              <div className="text-sm text-slate-400">
                {vizData
                  ? `${vizData.input_spectrogram.shape.join(" x ")}`
                  : ""}
              </div>
            </div>

            {vizData ? (
              <FeatureMap data={vizData.input_spectrogram.values} title={`${vizData.input_spectrogram.shape.join(" x ")}`} spectrogram />
            ) : (
              <div className="text-slate-400">No data yet — upload a file</div>
            )}

            <div className="mt-3 flex justify-end">
              <ColorScale width={220} height={16} min={-1} max={1} />
            </div>
          </Card>

          <Card className="border border-slate-700 bg-slate-800/70 p-4">
            <h3 className="mb-2 font-medium">Waveform</h3>
            <div className="waveform-wrapper rounded-md border border-slate-700 bg-black/20">
              {vizData ? (
                <Waveform
                  data={vizData.waveform.values}
                  title={`${vizData.waveform.duration.toFixed(2)}s · ${vizData.waveform.sample_rate}Hz`}
                />
              ) : (
                <div className="flex h-28 items-center justify-center text-slate-400">
                  Waveform will appear here
                </div>
              )}
            </div>
          </Card>

          <Card className="featuremap-area border border-slate-700 bg-slate-800/70 p-4">
            <h3 className="mb-3 font-medium">Convolutional Layer Outputs</h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {main.length === 0 && (
                <div className="col-span-full text-slate-400">
                  Upload a file to populate feature maps
                </div>
              )}
              {main.map(([name, data]) => (
                <div
                  key={name}
                  className="featuremap-card rounded border border-slate-700 bg-slate-900 p-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-semibold">{name}</div>
                    <div className="text-xs text-slate-400">
                      {data.shape.join("×")}
                    </div>
                  </div>
                  <div className="mt-2 flex h-32 items-center justify-center rounded bg-black/20">
                    <FeatureMap data={data.values} title={name} />
                  </div>
                  {internals[name] && (
                    <div className="featuremap-scroll mt-2">
                      <div className="grid grid-cols-2 gap-2">
                        {internals[name].map(([lname, ldata]) => (
                          <div
                            key={lname}
                            className="rounded border border-slate-700 bg-slate-800/60 p-2"
                          >
                            <FeatureMap
                              data={ldata.values}
                              title={lname.replace(`${name}.`, "")}
                              internal
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </section>
      </div>
    </main>
  );
  
}
