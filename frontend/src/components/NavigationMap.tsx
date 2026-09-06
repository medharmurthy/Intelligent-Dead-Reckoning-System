import { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap, Polyline } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { NavigationState } from '../navigation/NavigationEngine';
import { NavigationStateMode } from '../navigation/gnss_quality';

// Fix Leaflet's default icon path issues with Vite
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

interface NavigationMapProps {
  navState: NavigationState | null;
  history: NavigationState[];
}

// Custom hook to re-center the map on the vehicle's position
function MapUpdater({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, map.getZoom(), { animate: true });
  }, [center, map]);
  return null;
}

export const NavigationMap: React.FC<NavigationMapProps> = ({ navState, history }) => {
  const defaultCenter: [number, number] = [26.1445, 91.7362]; // Fallback to Guwahati
  const currentPos: [number, number] = navState ? [navState.latitude, navState.longitude] : defaultCenter;

  // Split history into segments based on mode to color-code the trajectory
  const segments: { positions: [number, number][], color: string }[] = [];
  let currentSegment: { positions: [number, number][], color: string } | null = null;

  const getColorForMode = (mode: string) => {
    switch (mode) {
      case NavigationStateMode.GNSS_AIDED: return '#3b82f6'; // blue
      case NavigationStateMode.DEGRADED: return '#eab308'; // yellow
      case NavigationStateMode.DEAD_RECKONING: return '#ef4444'; // red
      case NavigationStateMode.REACQUISITION: return '#f97316'; // orange
      default: return '#3b82f6';
    }
  };

  history.forEach((state) => {
    const pos: [number, number] = [state.latitude, state.longitude];
    const color = getColorForMode(state.mode);
    
    if (!currentSegment || currentSegment.color !== color) {
      currentSegment = { positions: [pos], color };
      segments.push(currentSegment);
    } else {
      currentSegment.positions.push(pos);
    }
  });

  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      <MapContainer 
        center={currentPos} 
        zoom={18} 
        style={{ height: '100%', width: '100%', zIndex: 1 }}
        zoomControl={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {segments.map((seg, idx) => (
          <Polyline key={idx} positions={seg.positions} color={seg.color} weight={5} opacity={0.8} />
        ))}

        {navState && (
          <Marker position={currentPos}>
            <Popup>
              Mode: {navState.mode}<br/>
              Speed: {navState.speed} km/h
            </Popup>
          </Marker>
        )}
        
        {navState && <MapUpdater center={currentPos} />}
      </MapContainer>
    </div>
  );
};
