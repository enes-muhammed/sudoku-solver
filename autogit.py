import subprocess
import time
from datetime import datetime
import platform


system = platform.system()
INTERVAL = 300  # 5 dakika


def run_git(*args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, end="")

    return result


def get_status():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def get_sync_status():
    result = subprocess.run(
        [
            "git",
            "rev-list",
            "--left-right",
            "--count",
            "HEAD...origin/main"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return None

    ahead, behind = map(int, result.stdout.strip().split())

    return ahead, behind


def sync_once():
    print("\n" + "=" * 60)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        "SYNC CHECK"
    )
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. GitHub'daki son durumu öğren
    # ---------------------------------------------------------

    print("\n[1] GitHub kontrol ediliyor...")

    fetch = run_git("fetch", "origin")

    if fetch.returncode != 0:
        print("❌ GitHub kontrol edilemedi.")
        return

    sync = get_sync_status()

    if sync is None:
        print("❌ Git branch durumu okunamadı.")
        return

    ahead, behind = sync

    print(f"Local : {ahead} commit önde")
    print(f"Remote: {behind} commit önde")

    # ---------------------------------------------------------
    # 2. GitHub'da yeni commit varsa DUR
    # ---------------------------------------------------------

    if behind > 0:

        print("\n⚠ GitHub'da daha yeni commit var.")
        print("Autogit otomatik push yapmayacak.")
        print("Önce 'python dev.py' ile senkronize ol.")

        return

    # ---------------------------------------------------------
    # 3. Local değişiklikleri kontrol et
    # ---------------------------------------------------------

    status = get_status()

    if status is None:
        print("❌ Git status alınamadı.")
        return

    if not status:
        print("\n✓ Yeni değişiklik yok.")
        return

    print("\nDeğişiklik bulundu:")
    print(status)

    # ---------------------------------------------------------
    # 4. Commit
    # ---------------------------------------------------------

    print("\n[2] Değişiklikler stage ediliyor...")

    add = run_git("add", ".")

    if add.returncode != 0:
        print("❌ git add başarısız.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n[3] Commit oluşturuluyor...")

    commit = run_git(
        "commit",
        "-m",
        f"auto [{system}]: {timestamp}"
    )

    if commit.returncode != 0:
        print("❌ Commit başarısız.")
        return

    # ---------------------------------------------------------
    # 5. Push
    # ---------------------------------------------------------

    print("\n[4] GitHub'a gönderiliyor...")

    push = run_git("push", "origin", "main")

    if push.returncode != 0:
        print("\n❌ Push başarısız.")
        print("Commit localde kaldı.")
        return

    print("\n✓ Değişiklikler GitHub'a gönderildi.")


def main():
    print("=" * 60)
    print("           SUDOKU SOLVER - AUTOGIT")
    print("=" * 60)

    print(f"\nOtomatik kontrol aralığı: {INTERVAL // 60} dakika")
    print("Çıkmak için CTRL+C\n")

    while True:
        try:
            sync_once()

            print(
                f"\nSonraki kontrol "
                f"{INTERVAL // 60} dakika sonra..."
            )

            time.sleep(INTERVAL)

        except KeyboardInterrupt:
            print("\n\nAutogit durduruldu.")
            break

        except Exception as e:
            print(f"\n❌ Beklenmeyen hata: {e}")
            print("Program çalışmaya devam edecek.")

            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()