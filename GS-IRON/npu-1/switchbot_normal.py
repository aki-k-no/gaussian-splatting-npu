import requests
import argparse
import time

# ====== コマンドライン引数 ======
parser = argparse.ArgumentParser(description="SwitchBot Plug Mini 平常時消費電力測定（ジュール）")
parser.add_argument("--token", required=True, help="SwitchBot APIトークン")
parser.add_argument("--duration", type=int, default=60, help="測定時間（秒）")
args = parser.parse_args()

API_TOKEN = args.token
DURATION = args.duration

# ====== デバイスID固定 ======
DEVICE_ID = "48CA43C1BAEE"
API_URL = f"https://api.switch-bot.com/v1.0/devices/{DEVICE_ID}/status"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json; charset=utf8"
}

interval = 1  # 秒
total_energy_joules = 0.0
sample_count = 0

print(f"平常時消費電力測定開始（{DURATION}秒）...")

start_time = time.time()
while time.time() - start_time < DURATION:
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data["statusCode"] == 100:
            power = data["body"].get("weight")  # W
            if power is not None:
                print(f"現在の消費電力: {power:.2f} W")
                total_energy_joules += power * interval  # J
                sample_count += 1
            else:
                print("電力情報が取得できません")
        else:
            print(f"APIエラー: {data}")

    except requests.exceptions.RequestException as e:
        print(f"通信エラー: {e}")

    time.sleep(interval)

elapsed_seconds = time.time() - start_time
elapsed_minutes = elapsed_seconds / 60
avg_power = total_energy_joules / elapsed_seconds if elapsed_seconds > 0 else 0.0

print("\n===== 結果（平常時）=====")
print(f"測定サンプル数: {sample_count}")
print(f"総消費エネルギー: {total_energy_joules:.2f} J")
print(f"経過時間: {elapsed_minutes:.2f} 分")
print(f"毎秒平均消費電力: {avg_power:.2f} W")
