"""
MapReduce Log Analyzer Simulation in Python
-------------------------------------------
This script simulates how a distributed MapReduce job would process
a system log file to find which user has been logged in for the longest time.

Steps:
1. Create a sample log file 'logs.txt'
2. Mapper: emits (user, (event, timestamp))
3. Shuffle: groups all events by user
4. Reducer: computes total login duration per user
5. Output: prints intermediate and final results
"""

from datetime import datetime
from collections import defaultdict
import pprint

# === STEP 1: Create a sample log file ===
def create_log_file(filename="logs.txt"):
    log_data = """\
2025-11-01T08:00:00 alice LOGIN 
2025-11-01T08:45:00 alice LOGOUT
2025-11-01T09:00:00 bob LOGIN
2025-11-01T12:30:00 bob LOGOUT
2025-11-01T10:00:00 carol LOGIN
2025-11-01T17:00:00 carol LOGOUT
2025-11-01T14:00:00 alice LOGIN
2025-11-01T18:00:00 alice LOGOUT
2025-11-02T08:30:00 bob LOGIN
2025-11-02T11:30:00 bob LOGOUT
"""
    with open(filename, "w") as f:
        f.write(log_data)
    print(f"Created sample log file: {filename}\n")

# === STEP 2: Mapper function ===
def mapper(filename="logs.txt"):
    mapped = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            ts, user, event = parts
            mapped.append((user, (event, ts)))
    return mapped

# === STEP 3: Shuffle function (group by user) ===
def shuffle(mapped_data):
    grouped = defaultdict(list)
    for user, value in mapped_data:
        grouped[user].append(value)
    # Sort each user's events by timestamp
    for user in grouped:
        grouped[user].sort(key=lambda x: x[1])
    return grouped

# === STEP 4: Reducer function ===
def reducer(user, events):
    total_time = 0
    login_time = None
    sessions = []
    for event, ts_str in events:
        ts = datetime.fromisoformat(ts_str)
        if event == "LOGIN":
            login_time = ts
        elif event == "LOGOUT" and login_time:
            duration = (ts - login_time).total_seconds()
            total_time += duration
            sessions.append((login_time, ts, duration))
            login_time = None
    return {
        "user": user,
        "total_hours": total_time / 3600,
        "sessions": sessions
    }

# === STEP 5: Driver / Coordinator ===
def main():
    # Create a sample log file
    create_log_file()

    # Mapper phase
    mapped = mapper()
    print("=== MAPPER OUTPUT ===")
    pprint.pprint(mapped)
    print()

    # Shuffle phase
    shuffled = shuffle(mapped)
    print("=== SHUFFLE OUTPUT ===")
    for user, events in shuffled.items():
        print(user, "->", events)
    print()

    # Reducer phase
    results = []
    for user, events in shuffled.items():
        results.append(reducer(user, events))

    print("=== REDUCER OUTPUT ===")
    for r in results:
        print(f"{r['user']}: {r['total_hours']:.2f} hours total")
        for s in r['sessions']:
            start, end, dur = s
            print(f"  {start} -> {end} ({dur/3600:.2f} hrs)")
        print()

    # Find the user(s) with max total time
    max_time = max(r['total_hours'] for r in results)
    winners = [r for r in results if r['total_hours'] == max_time]

    print("=== FINAL RESULT ===")
    for w in winners:
        print(f"User with maximum login duration: {w['user']} ({w['total_hours']:.2f} hours)")
    print()

# === RUN ===
if __name__ == "__main__":
    main()
