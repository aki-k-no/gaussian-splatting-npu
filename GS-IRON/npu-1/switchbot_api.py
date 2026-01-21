import requests
import argparse
import time
import subprocess

# ====== コマンドライン引数 ======
parser = argparse.ArgumentParser(description="SwitchBot Plug Mini 消費電力差分測定（平常時時間自動調整）")
parser.add_argument("--token", required=True, help="SwitchBot APIトークン")
parser.add_argument("--interval", type=int, default=1, help="取得間隔（秒）")
args = parser.parse_args()

API_TOKEN = args.token
INTERVAL = args.interval

# ====== デバイスID固定 ======
DEVICE_ID = "48CA43C1BAEE"
API_URL = f"https://api.switch-bot.com/v1.0/devices/{DEVICE_ID}/status"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json; charset=utf8"
}

def read_total_cpu_energy():
    paths = [
        "/sys/class/powercap/intel-rapl:0/energy_uj",
    ]
    return sum(int(open(p).read()) for p in paths)


# ====== 測定関数 ======
def measure_energy(duration=None, description="測定", run_process=False):
    total_energy = 0.0
    samples = 0
    cpu_start = read_total_cpu_energy()

    proc = None
    if run_process:
        proc = subprocess.Popen(["make", "run"])
        print(f"{description}測定開始（プログラム実行中）...")
    else:
        print(f"{description}測定開始（平常時）...")

    start_time = time.time()
    try:
        while True:
            if run_process and proc.poll() is not None:
                break
            elif not run_process and duration is not None and time.time() - start_time >= duration:
                break

            try:
                response = requests.get(API_URL, headers=HEADERS, timeout=5)
                response.raise_for_status()
                data = response.json()

                if data["statusCode"] == 100:
                    power = data["body"].get("weight")  # W
                    if power is not None:
                        # print(f"[{description}] 現在の消費電力: {power:.2f} W")
                        total_energy += power * INTERVAL  # J
                        samples += 1
                    else:
                        print("電力情報が取得できません")
                else:
                    print(f"APIエラー: {data}")

            except requests.exceptions.RequestException as e:
                print(f"通信エラー: {e}")

            time.sleep(INTERVAL)
    finally:
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        elapsed_minutes = elapsed_seconds / 60
        avg_power = total_energy / elapsed_seconds if elapsed_seconds > 0 else 0.0  # W
        if proc:
            proc.terminate()
            
    cpu_end = read_total_cpu_energy()

    return {
        "total_energy": total_energy,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_minutes,
        "avg_power": avg_power,
        "samples": samples,
        "cpu_energy": (cpu_end - cpu_start) / 1e6  # J
    }

# ====== プログラム実行時測定 ======
active_result = measure_energy(None, "プログラム実行時", run_process=True)
program_duration = active_result["elapsed_seconds"]  # 秒単位

# ====== 平常時測定（プログラム時間と同じ） ======
idle_result = measure_energy(duration=program_duration, description="平常時", run_process=False)

# ====== 差分計算 ======
diff_power = active_result["avg_power"] - idle_result["avg_power"]

# ====== 結果表示 ======
print("\n===== 測定結果 =====")
print("プログラム実行時:")
print(f"  測定サンプル数: {active_result['samples']}")
print(f"  総消費エネルギー: {active_result['total_energy']:.2f} J")
print(f"  CPU総消費エネルギー: {active_result['cpu_energy']:.2f} J")
print(f"  経過時間: {active_result['elapsed_seconds']:.2f} 秒")
print(f"  毎秒平均消費電力: {active_result['avg_power']:.2f} W")

print("平常時（同じ時間）:")
print(f"  測定サンプル数: {idle_result['samples']}")
print(f"  総消費エネルギー: {idle_result['total_energy']:.2f} J")
print(f"  CPU総消費エネルギー: {idle_result['cpu_energy']:.2f} J")
print(f"  経過時間: {idle_result['elapsed_seconds']:.2f} 秒")
print(f"  毎秒平均消費電力: {idle_result['avg_power']:.2f} W")

print(f"\nプログラム負荷分の差分消費電力: {diff_power:.2f} W")
print(f"プログラム全体の差分電力消費: {active_result['total_energy'] - idle_result['total_energy']:.2f} J")
print(f"CPU負荷の差分エネルギー消費: {active_result['cpu_energy'] - idle_result['cpu_energy']:.2f} J")