import { useState, useEffect } from 'react';
import { useSensors } from './sensors/useSensors';
import { useInferenceLoop } from './sensors/useInferenceLoop';
import { useNavigation } from './navigation/useNavigation';
import { NavigationMap } from './components/NavigationMap';
import { NavigationStateMode } from './navigation/gnss_quality';
import { 
  Navigation, Map as MapIcon, Settings, Activity, 
  Cpu, WifiOff, Server, Play, Satellite, AlertTriangle, Compass, RotateCcw
} from 'lucide-react';
import type { SensorFrame } from './types/sensor';

export default function App() {
  const [simulateGnssLoss, setSimulateGnssLoss] = useState(false);
  const [replayMode, setReplayMode] = useState(false);
  const [replayData, setReplayData] = useState<SensorFrame[] | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  const { requestPermissions, resetAlignment, diagnostics: sensorDiag, latestFrame } = useSensors({ 
    simulateGnssLoss,
    replayData
  });
  
  const { diagnostics: inferenceDiag } = useInferenceLoop(latestFrame);
  const aiOutput = inferenceDiag.latestOutput;
  const { navState, history } = useNavigation(latestFrame, aiOutput, sensorDiag.stationary.isStationary);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (replayMode) {
      const mockReplay: SensorFrame[] = [];
      const startLat = navState ? navState.latitude : 26.1445;
      const startLon = navState ? navState.longitude : 91.7362;
      
      for(let i=0; i<300; i++) {
        // Simulate a car accelerating and turning
        const simulatedSpeed = Math.min(15, i * 0.2); // Accelerate up to 15 m/s
        
        mockReplay.push({
          timestamp: Date.now() + i * 100,
          ax: i < 50 ? 2.0 : 0, // Accelerating for first 5 seconds
          ay: 0.5, // Slight lateral turn force
          az: 9.8, 
          gx: 0, gy: 0, gz: 0.1, // Turning
          gnss: { 
            timestamp: Date.now()+i*100, 
            latitude: startLat + (i * 0.00005), 
            longitude: startLon + (i * 0.00005), 
            speed: simulatedSpeed, 
            heading: 45 + i * 0.5, 
            accuracy: 2 
          }
        });
      }
      setReplayData(mockReplay);
    } else {
      setReplayData(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayMode]);

  if (sensorDiag.permissionGranted === null) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white p-6">
        <div className="flex flex-col items-center bg-slate-900 p-10 rounded-3xl border border-slate-800 shadow-2xl max-w-md w-full text-center">
          <Navigation size={56} className="text-blue-500 mb-6" />
          <h1 className="text-3xl font-bold mb-3 text-white">IDR Navigation</h1>
          <p className="text-slate-400 mb-8 text-sm leading-relaxed">
            Intelligent Dead Reckoning requires access to your device's motion sensors and location to provide seamless navigation during GPS outages.
          </p>
          <button 
            onClick={requestPermissions}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-xl font-bold shadow-lg transition-colors"
          >
            Grant Permissions & Start
          </button>
        </div>
      </div>
    );
  }

  const isDeadReckoning = navState?.mode === NavigationStateMode.DEAD_RECKONING;
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  const formatDate = (date: Date) => {
    return date.toLocaleDateString([], { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' });
  };

  return (
    <div className="h-screen w-screen bg-[#0B0F19] flex flex-col font-sans text-slate-300 overflow-hidden">
      {/* Top Navigation Bar */}
      <header className="h-auto md:h-16 border-b border-slate-800/60 bg-[#0B0F19] flex flex-wrap items-center justify-between px-4 md:px-6 py-2 md:py-0 shrink-0 z-20 gap-2">
        <div className="flex items-center gap-3 md:gap-4 max-w-full">
          <div className="bg-blue-600 p-2 rounded-lg shrink-0">
            <Navigation size={20} className="text-white" />
          </div>
          <div className="flex flex-col min-w-0">
            <h1 className="text-white font-bold text-base md:text-lg leading-tight truncate">IDR Navigation</h1>
            <div className="text-slate-400 text-[10px] md:text-xs truncate">Intelligent Dead Reckoning</div>
          </div>
          <div className="hidden sm:flex px-3 py-1 bg-emerald-600 border border-emerald-700 rounded-full items-center gap-2 shrink-0">
            <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
            <span className="text-white text-xs font-bold tracking-wider">SYSTEM READY</span>
          </div>
        </div>
        <div className="flex items-center shrink-0">
          <div className="text-right">
            <div className="text-white font-medium text-sm md:text-base">{formatTime(currentTime)}</div>
            <div className="text-slate-400 text-[10px] md:text-xs">{formatDate(currentTime)}</div>
          </div>
        </div>
      </header>

      {/* Main Content Dashboard */}
      <div className="flex flex-col lg:flex-row flex-1 overflow-y-auto lg:overflow-hidden p-4 gap-4">
        
        {/* Left Column: Map */}
        <div className="min-h-[50vh] lg:min-h-0 flex-1 relative rounded-2xl overflow-hidden border border-slate-700 shadow-md flex flex-col bg-slate-800">
          <div className="absolute inset-0 z-0">
            <NavigationMap navState={navState} history={history} />
          </div>

        </div>

        {/* Right Column: Sidebar Dashboard */}
        <div className="w-full lg:w-[420px] flex flex-col gap-4 overflow-y-auto no-scrollbar shrink-0">

          {/* GNSS Lost Alert */}
          {isDeadReckoning && (
            <div className="bg-red-600 border border-red-700 rounded-2xl p-4 flex items-center gap-4">
              <AlertTriangle className="text-white shrink-0" size={28} />
              <div>
                <div className="text-white font-bold tracking-wider text-sm">GNSS SIGNAL LOST</div>
                <div className="text-red-100 text-xs mt-0.5">Continuing with AI Dead Reckoning</div>
              </div>
            </div>
          )}

          {/* Speedometer Widget */}
          <div className="bg-slate-800 border border-slate-700 p-5 rounded-2xl">
            <div className="flex items-center gap-2 mb-4">
              <Compass className="text-blue-400" size={18} />
              <span className="font-semibold text-slate-200">
                {isDeadReckoning ? 'Dead Reckoning' : 'GNSS Aided'}
              </span>
            </div>
            <div className="flex items-end gap-6 mb-4">
              <div>
                <div className="text-6xl font-bold text-white tracking-tighter leading-none">
                  {navState ? Math.round(navState.speed) : '0'}
                </div>
                <div className="text-slate-400 font-medium mt-1">km/h</div>
              </div>
              <div className="mb-1">
                <div className="text-slate-400 text-xs mb-1">Heading</div>
                <div className="text-white font-medium flex items-center gap-1">
                  <Navigation size={14} className="text-slate-300" />
                  {navState?.heading.toFixed(0)}°
                </div>
              </div>
            </div>
            <div className="flex justify-between items-center text-xs pt-3 border-t border-slate-800">
              <span className="text-slate-400">AI-assisted</span>
              <span className="text-blue-400 font-medium">{((navState?.aiConfidence || 0) * 100).toFixed(0)}% confidence</span>
            </div>
          </div>
          
          {/* System Status Panel */}
          <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5 flex flex-col gap-4">
            <h2 className="text-white font-semibold flex items-center gap-2 mb-1">
              <Activity className="text-blue-500" size={18} /> System Status
            </h2>
            <div className="grid grid-cols-[auto_1fr_auto] gap-x-4 gap-y-3 items-center text-sm">
              <Satellite size={16} className="text-slate-500" />
              <span className="text-slate-300">GNSS</span>
              {sensorDiag.gnssAvailable ? (
                <span className="text-emerald-400 font-medium flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-emerald-400"></div> ACTIVE</span>
              ) : (
                <span className="text-red-400 font-medium flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-red-400"></div> LOST</span>
              )}

              <Cpu size={16} className="text-slate-500" />
              <span className="text-slate-300">IMU (Phone)</span>
              <span className="text-emerald-400 font-medium">{sensorDiag.rawImuRateHz} Hz</span>

              <Server size={16} className="text-slate-500" />
              <span className="text-slate-300">AI Model (ONNX)</span>
              <span 
                className={`font-medium ${inferenceDiag.modelLoaded ? 'text-emerald-400' : (inferenceDiag.modelError ? 'text-red-400' : 'text-yellow-400')}`}
                title={inferenceDiag.modelError || undefined}
              >
                {inferenceDiag.modelLoaded ? 'READY' : (inferenceDiag.modelError ? 'ERROR' : 'LOADING')}
              </span>

              <MapIcon size={16} className="text-slate-500" />
              <span className="text-slate-300">Navigation Mode</span>
              <span className={`font-medium text-right ${isDeadReckoning ? 'text-red-400' : 'text-blue-400'}`}>
                {navState?.mode?.replace('_', ' ') || 'INIT'}
              </span>
            </div>
          </div>

          {/* AI Inference Panel */}
          <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5 flex flex-col gap-4">
            <h2 className="text-white font-semibold flex items-center gap-2 mb-1">
              <Cpu className="text-blue-500" size={18} /> AI Inference (Tiny TCN)
            </h2>
            <div className="grid grid-cols-3 gap-4 border-b border-slate-800 pb-4">
              <div>
                <div className="text-xs text-slate-400 mb-1">Predicted Speed</div>
                <div className="text-xl text-white font-semibold">{aiOutput ? aiOutput.speed_mps.toFixed(1) : '0.0'} <span className="text-sm font-normal text-slate-500">m/s</span></div>
              </div>
              <div>
                <div className="text-xs text-slate-400 mb-1">Uncertainty (σ)</div>
                <div className="text-xl text-white font-semibold">{aiOutput ? Math.sqrt(aiOutput.variance).toFixed(1) : '0.0'} <span className="text-sm font-normal text-slate-500">m/s</span></div>
              </div>
              <div>
                <div className="text-xs text-slate-400 mb-1">Latency</div>
                <div className="text-xl text-white font-semibold flex items-center gap-1">
                  <Activity size={14} className="text-blue-500" /> {inferenceDiag.lastLatencyMs.toFixed(1)} <span className="text-sm font-normal text-slate-500">ms</span>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between pt-1">
              <span className="text-sm text-slate-400">Motion State</span>
              <span className="bg-blue-500/10 text-blue-400 px-3 py-1 rounded-full text-xs font-bold border border-blue-500/20">
                {aiOutput ? ['STATIONARY','ACCELERATING','CRUISING','BRAKING','TURNING'][aiOutput.motionClassId] : 'UNKNOWN'}
              </span>
            </div>
          </div>

          {/* Sensor Diagnostics Panel */}
          <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5 flex flex-col gap-4">
            <h2 className="text-white font-semibold flex items-center gap-2 mb-1">
              <Activity className="text-blue-500" size={18} /> Sensor Diagnostics
            </h2>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                <span className="text-slate-400">Accelerometer (Vehicle Frame)</span>
                <span className="text-slate-300 font-mono text-[10px] text-right">
                  x {latestFrame?.ax?.toFixed(2) || '0.00'} y {latestFrame?.ay?.toFixed(2) || '0.00'} z {latestFrame?.az?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                <span className="text-slate-400">Alignment Status</span>
                <span className={`font-mono text-[10px] text-right ${sensorDiag.alignment.status === 'ALIGNED' ? 'text-emerald-400' : 'text-yellow-400'}`}>
                  {sensorDiag.alignment.status} ({(sensorDiag.alignment.confidence * 100).toFixed(0)}%)
                </span>
              </div>
              <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                <span className="text-slate-400">Stationary State</span>
                <span className={`font-mono text-[10px] text-right ${sensorDiag.stationary.isStationary ? 'text-red-400' : 'text-emerald-400'}`}>
                  {sensorDiag.stationary.status} ({(sensorDiag.stationary.confidence * 100).toFixed(0)}%)
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Sample Rate</span>
                <span className="text-emerald-400 font-medium">{sensorDiag.resampledRateHz} Hz</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Window Buffer</span>
                <span className="text-emerald-400 font-medium">{sensorDiag.bufferSize} / 20</span>
              </div>
            </div>
          </div>

          {/* Controls Panel */}
          <div className="bg-[#111827] border border-slate-800/80 rounded-2xl p-5 flex flex-col gap-4 mt-auto">
            <h2 className="text-white font-semibold flex items-center gap-2 mb-1">
              <Settings className="text-blue-500" size={18} /> Controls
            </h2>
            <div className="grid grid-cols-2 gap-3">
              <button 
                onClick={() => setSimulateGnssLoss(!simulateGnssLoss)}
                className={`py-3 px-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors border ${
                  simulateGnssLoss 
                    ? 'bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20' 
                    : 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700'
                }`}
              >
                {simulateGnssLoss ? <WifiOff size={16} /> : <Satellite size={16} />}
                {simulateGnssLoss ? 'Restore GNSS' : 'Simulate Loss'}
              </button>
              
              <button 
                onClick={() => setReplayMode(!replayMode)}
                className={`py-3 px-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors border ${
                  replayMode
                    ? 'bg-blue-500/10 border-blue-500/30 text-blue-400 hover:bg-blue-500/20'
                    : 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700'
                }`}
              >
                <Play size={16} /> {replayMode ? 'Stop Replay' : 'Replay Demo'}
              </button>
              
              <button 
                onClick={resetAlignment}
                className="col-span-2 py-3 px-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors border bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700"
              >
                <RotateCcw size={16} /> Recalibrate Alignment
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
