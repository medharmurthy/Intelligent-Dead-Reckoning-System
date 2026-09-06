import os
import csv
import math

def parse_single_trip(s_path, v_path):
    if not (os.path.exists(s_path) and os.path.exists(v_path)):
        return None, None
    if os.path.getsize(s_path) < 1000 or os.path.getsize(v_path) < 1000:
        return None, None
        
    phone_data = []
    with open(s_path, mode='r', encoding='latin1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                ax_keys = [v for k, v in cleaned_row.items() if k.startswith('ACCELEROMETER X')]
                ay_keys = [v for k, v in cleaned_row.items() if k.startswith('ACCELEROMETER Y')]
                az_keys = [v for k, v in cleaned_row.items() if k.startswith('ACCELEROMETER Z')]
                
                if not (ax_keys and ay_keys and az_keys):
                    continue
                    
                ax = float(ax_keys[0])
                ay = float(ay_keys[0])
                az = float(az_keys[0])
                
                gx = float(cleaned_row.get('GYROSCOPE Pitch (rad/s)', 0))
                gy = float(cleaned_row.get('GYROSCOPE Roll (rad/s)', 0))
                gz = float(cleaned_row.get('GYROSCOPE Yaw (rad/s)', 0))
                
                time_ms = float(cleaned_row.get('TIME SINCE START (ms)', 0))
                accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
                
                phone_data.append({
                    'time_ms': time_ms,
                    'ax': ax, 'ay': ay, 'az': az,
                    'gx': gx, 'gy': gy, 'gz': gz,
                    'accel_mag': round(accel_mag, 4)
                })
            except (ValueError, IndexError):
                continue
                
    vehicle_data = []
    with open(v_path, mode='r', encoding='latin1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                time_s = float(cleaned_row.get('Time Since Start of Day (seconds)', 0))
                speed_kmh = float(cleaned_row.get('Velocity (km/hr)', 0))
                speed_mps = round(speed_kmh / 3.6, 4)
                heading = float(cleaned_row.get('Heading (degrees)', 0))
                
                vehicle_data.append({
                    'time_s': time_s,
                    'speed_kmh': speed_kmh,
                    'speed_mps': speed_mps,
                    'heading': heading
                })
            except (ValueError, IndexError):
                continue
                
    return phone_data, vehicle_data

def load_trip_s1(base_dir="data/raw/IO-VNBD/IO-VNBD-master"):
    s_path = os.path.join(base_dir, "Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv")
    v_path = os.path.join(base_dir, "Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv")
    return parse_single_trip(s_path, v_path)

def load_all_featured_trips(base_dir="data/raw/IO-VNBD/IO-VNBD-master"):
    all_trips = []
    cat_dir = os.path.join(base_dir, "Synchronised V abd S datasets/Categorised IOVNB Dataset")
    if not os.path.exists(cat_dir):
        cat_dir = base_dir
        
    for root, dirs, files in os.walk(cat_dir):
        s_files = [f for f in files if f.startswith('S-') and f.endswith('.csv')]
        v_files = [f for f in files if f.startswith('V-') and f.endswith('.csv')]
        if s_files and v_files:
            s_path = os.path.join(root, s_files[0])
            v_path = os.path.join(root, v_files[0])
            p_data, v_data = parse_single_trip(s_path, v_path)
            if p_data and v_data and len(p_data) > 100:
                trip_name = os.path.basename(root)
                print(f"Loaded Trip [{trip_name}]: {len(p_data)} IMU frames, {len(v_data)} GT frames")
                all_trips.append((p_data, v_data))
                
    print(f"\nTotal Multi-Driver Trips Successfully Parsed: {len(all_trips)}")
    return all_trips

if __name__ == "__main__":
    load_all_featured_trips()
