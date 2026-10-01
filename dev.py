import platform
import shutil
import subprocess
import sys


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


def detect_environment():
    system = platform.system()

    print("\n[ENVIRONMENT]")
    print(f"Operating system : {system}")
    print(f"Platform         : {platform.platform()}")
    print(f"Python           : {platform.python_version()}")

    if system == "Windows":
        print("Environment      : Windows PC")

    elif system == "Linux":
        print("Environment      : Linux PC")
        
        # Arch Linux kontrolü
        if shutil.which("pacman"):
            print("Distribution     : Arch-based Linux")
        else:
            print("Distribution     : Unknown Linux")

    else:
        print(f"Environment      : Unsupported ({system})")
        sys.exit(1)

    print()


def check_git():
    print("[CHECK] Git kontrol ediliyor...")

    if shutil.which("git") is None:
        print("Git bulunamadı.")
        print("Önce Git kurulmalı.")
        sys.exit(1)

    result = run_git("--version")

    if result.returncode != 0:
        print("Git çalıştırılamadı.")
        sys.exit(1)

    print()


def get_git_status():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Git repository durumu okunamadı.")
        sys.exit(1)

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
        print("Branch durumu okunamadı.")
        sys.exit(1)

    ahead, behind = map(int, result.stdout.strip().split())

    return ahead, behind


def main():
    print("=" * 60)
    print("       SUDOKU SOLVER - DEVELOPMENT START")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. ENVIRONMENT
    # ---------------------------------------------------------

    detect_environment()

    # ---------------------------------------------------------
    # 2. GIT
    # ---------------------------------------------------------

    check_git()

    print("[1/4] Git çalışma durumu kontrol ediliyor...\n")

    status = get_git_status()

    if status:
        print("Yerel değişiklikler var:")
        print(status)
    else:
        print("Yerel çalışma alanı temiz.")

    # ---------------------------------------------------------
    # 3. GITHUB
    # ---------------------------------------------------------

    print("\n[2/4] GitHub kontrol ediliyor...\n")

    fetch = run_git("fetch", "origin")

    if fetch.returncode != 0:
        print("\nGitHub'a ulaşılamadı.")
        print("İnternet bağlantısını veya remote ayarlarını kontrol et.")
        sys.exit(1)

    ahead, behind = get_sync_status()

    print(f"\nLocal yeni commitler : {ahead}")
    print(f"GitHub yeni commitler: {behind}")

    # ---------------------------------------------------------
    # 4. SYNC
    # ---------------------------------------------------------

    if behind > 0 and ahead == 0:

        print("\nGitHub'da daha yeni bir sürüm var.")

        if status:
            print("\n⚠ Yerel olarak kaydedilmemiş değişikliklerin var.")
            print("Otomatik pull yapılmayacak.")
            print("Önce değişikliklerini commit veya stash et.")
            sys.exit(1)

        print("Son sürüm çekiliyor...\n")

        pull = run_git("pull", "--ff-only")

        if pull.returncode != 0:
            print("\nPull başarısız oldu.")
            print("Hiçbir merge işlemi otomatik yapılmadı.")
            sys.exit(1)

        print("\nGitHub sürümü başarıyla alındı.")

    elif behind > 0 and ahead > 0:

        print("\n⚠ LOCAL VE GITHUB AYRILMIŞ DURUMDA")
        print()
        print("Local tarafında yeni commitler:")
        print(f"    {ahead}")
        print("GitHub tarafında yeni commitler:")
        print(f"    {behind}")
        print()
        print("Otomatik işlem yapılmadı.")
        print("Bu durumu manuel çözmemiz gerekiyor.")
        sys.exit(1)

    elif ahead > 0:

        print("\nLocal'de GitHub'a gönderilmemiş commitler var.")
        print(f"Bekleyen commit: {ahead}")
        print("Şimdilik otomatik push yapılmadı.")

    else:

        print("\nGitHub ile tamamen güncelsin.")

    print("\n[4/4] Development ortamı hazır.")
    print("=" * 60)


if __name__ == "__main__":
    main()