import subprocess
import platform
from datetime import datetime


MACHINE_FILE = ".machine_name"


def get_machine_name():
    try:
        with open(MACHINE_FILE, "r", encoding="utf-8") as file:
            name = file.read().strip()

        if name:
            return name

    except FileNotFoundError:
        pass

    return platform.system().lower()


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
    result = run_git("status", "--porcelain")

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def get_sync_status():
    result = run_git(
        "rev-list",
        "--left-right",
        "--count",
        "HEAD...origin/main"
    )

    if result.returncode != 0:
        return None

    ahead, behind = map(
        int,
        result.stdout.strip().split()
    )

    return ahead, behind


def sync_once():
    machine_name = get_machine_name()

    print("=" * 60)
    print("              GIT PUSH")
    print("=" * 60)
    print(
        f"Machine : {machine_name}"
    )
    print(
        f"Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ---------------------------------------------------------
    # 1. GitHub'daki son durumu öğren
    # ---------------------------------------------------------

    print("\n[1] GitHub kontrol ediliyor...")

    fetch = run_git("fetch", "origin")

    if fetch.returncode != 0:
        print("\n❌ GitHub kontrol edilemedi.")
        return False

    sync = get_sync_status()

    if sync is None:
        print("\n❌ Git branch durumu okunamadı.")
        return False

    ahead, behind = sync

    print(f"\nLocal : {ahead} commit önde")
    print(f"Remote: {behind} commit önde")

    # ---------------------------------------------------------
    # 2. GitHub'da daha yeni commit varsa DUR
    # ---------------------------------------------------------

    if behind > 0:

        print("\n⚠ GitHub'da daha yeni commit var.")
        print("Push yapılmayacak.")
        print("Önce 'python dev.py' çalıştır.")

        return False

    # ---------------------------------------------------------
    # 3. Local değişiklikleri kontrol et
    # ---------------------------------------------------------

    status = get_status()

    if status is None:
        print("\n❌ Git status alınamadı.")
        return False

    if not status:
        print("\n✓ Gönderilecek yeni değişiklik yok.")
        return True

    print("\nDeğişiklik bulundu:")
    print(status)

    # ---------------------------------------------------------
    # 4. Stage
    # ---------------------------------------------------------

    print("\n[2] Değişiklikler stage ediliyor...")

    add = run_git("add", ".")

    if add.returncode != 0:
        print("\n❌ git add başarısız.")
        return False

    # ---------------------------------------------------------
    # 5. Commit
    # ---------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print("\n[3] Commit oluşturuluyor...")

    commit = run_git(
        "commit",
        "-m",
        f"auto [{machine_name}]: {timestamp}"
    )

    if commit.returncode != 0:
        print("\n❌ Commit başarısız.")
        return False

    # ---------------------------------------------------------
    # 6. Push
    # ---------------------------------------------------------

    print("\n[4] GitHub'a gönderiliyor...")

    push = run_git(
        "push",
        "origin",
        "main"
    )

    if push.returncode != 0:
        print("\n❌ Push başarısız.")
        print("Commit localde kaldı.")

        return False

    print("\n✓ Değişiklikler GitHub'a gönderildi.")

    return True


def main():

    try:
        success = sync_once()

        print("\n" + "=" * 60)

        if success:
            print("                 DONE")
        else:
            print("                 STOPPED")

        print("=" * 60)

    except Exception as e:

        print("\n❌ Beklenmeyen hata:")
        print(e)


if __name__ == "__main__":
    main()
