import subprocess
import time
from datetime import datetime


INTERVAL = 300  # 5 dakika


def run_git(*args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True
    )

    return result


def has_changes():
    result = run_git("status", "--porcelain")
    return bool(result.stdout.strip())


def sync():
    print(f"[{datetime.now():%H:%M:%S}] Kontrol ediliyor...")

    if not has_changes():
        print("  Değişiklik yok.")
        return

    print("  Değişiklik bulundu.")

    result = run_git("add", ".")

    if result.returncode != 0:
        print("  git add başarısız!")
        print(result.stderr)
        return

    message = f"auto: {datetime.now():%Y-%m-%d %H:%M}"

    result = run_git("commit", "-m", message)

    if result.returncode != 0:
        print("  git commit başarısız!")
        print(result.stderr)
        return

    print("  Commit oluşturuldu.")

    result = run_git("push")

    if result.returncode != 0:
        print("  git push başarısız!")
        print(result.stderr)
        return

    print("  ✓ GitHub güncellendi.")


print("GitHub Auto Sync başladı.")
print(f"Kontrol aralığı: {INTERVAL} saniye")
print("Durdurmak için Ctrl+C.\n")

while True:
    try:
        sync()
        time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nAuto Sync kapatıldı.")
        break