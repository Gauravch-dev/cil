"""Build a labelled, deterministic format example. Never household/model evidence."""
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path

folder = Path(__file__).resolve().parents[1] / "frontend" / "public" / "templates"
folder.mkdir(parents=True, exist_ok=True)
target = folder / "SYNTHETIC-electricity-readings-8-days.csv"
columns = ["Datetime", "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
           "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Data_source"]
with target.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(columns)
    for minute in range(8*24*60):
        stamp = datetime(2026, 1, 1) + timedelta(minutes=minute)
        hour = stamp.hour + stamp.minute/60
        cycle = math.sin(2*math.pi*hour/24)
        power = .55 + .3*cycle**2 + (.8 if 18 <= hour < 21 else 0) + (.35 if 7 <= hour < 9 else 0)
        voltage = 230 + 1.5*cycle
        writer.writerow([stamp.strftime("%Y-%m-%d %H:%M:%S"), f"{power:.4f}", ".1800", f"{voltage:.3f}",
                         f"{power*1000/voltage:.4f}", "1.0", "2.0", "3.0", "synthetic_sample"])
print(f"Built synthetic format sample: {target.name} (11,520 rows, UTC)")
