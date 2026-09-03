import os
import csv
import math

def load_trip_s1(base_dir="data/raw/IO-VNBD/IO-VNBD-master"):
    s_path = os.path.join(base_dir, "Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv")
    v_path = os.path.join(base_dir, "Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv")
    
    if not os.path.exists(s_path) or os.path.getsize(s_path) < 1000:
        raise FileNotFoundError(f"Real LFS data file not found at {s_path}. Run download_trip.py first!")
        
    print(f"Loading Smartphone Data: {s_path}")
    print(f"Loading Vehicle GT Data:  {v_path}")
    
    # Read Smartphone Data
    phone_data = []
    with open(s_path, mode='r', encoding='latin1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                # Find accel keys by checking prefix
                ax = float([v for k, v in cleaned_row.items() if k.startswith('ACCELEROMETER X')][0])
                ay = float([v for k, v in cleaned_row.items() if k.startswith('ACCELEROMETER Y')][0])
                az = float([v for k, v in cleaned_row.items() if k.startswith('ACCELEROMETER Z')][0])
                
                gx = float(cleaned_row.get('GYROSCOPE Pitch (rad/s)', 0))
                gy = float(cleaned_row.get('GYROSCOPE Roll (rad/s)', 0))
                gz = float(cleaned_row.get('GYROSCOPE Yaw (rad/s)', 0))
                
                time_ms = float(cleaned_row.get('TIME SINCE START (ms)', 0))
                
                # Compute total acceleration magnitude: |a| = sqrt(ax^2 + ay^2 + az^2)
                accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
                
                phone_data.append({
                    'time_ms': time_ms,
                    'ax': ax, 'ay': ay, 'az': az,
                    'gx': gx, 'gy': gy, 'gz': gz,
                    'accel_mag': round(accel_mag, 4)
                })
            except (ValueError, IndexError):
                continue
                
    # Read Vehicle Ground Truth Data
    vehicle_data = []
    with open(v_path, mode='r', encoding='latin1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                time_s = float(cleaned_row.get('Time Since Start of Day (seconds)', 0))
                speed_kmh = float(cleaned_row.get('Velocity (km/hr)', 0))
                speed_mps = round(speed_kmh / 3.6, 4)  # convert km/h to m/s
                
                heading = float(cleaned_row.get('Heading (degrees)', 0))
                
                vehicle_data.append({
                    'time_s': time_s,
                    'speed_kmh': speed_kmh,
                    'speed_mps': speed_mps,
                    'heading': heading
                })
            except (ValueError, IndexError):
                continue
                
    print(f"\nSuccessfully parsed {len(phone_data)} Smartphone IMU frames (~{len(phone_data)/10/60:.1f} minutes of recording at 10Hz)")
    print(f"Successfully parsed {len(vehicle_data)} Vehicle Ground-Truth frames")
    
    print("\n--- First 3 Phone IMU Samples ---")
    for sample in phone_data[:3]:
        print(sample)
        
    print("\n--- First 3 Vehicle GT Samples ---")
    for sample in vehicle_data[:3]:
        print(sample)
        
    return phone_data, vehicle_data

if __name__ == "__main__":
    load_trip_s1()
