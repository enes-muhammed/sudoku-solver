import platform
import shutil
import subprocess
import sys


MACHINE_FILE = ".machine_name"


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, end="")

    return result


def run_git(*args):
    return run_command(["git", *args])


def get_machine_name():
    try:
        with open(MACHINE_FILE, "r", encoding="utf-8") as file:
            name = file.read().strip()

        if name:
            return name

    except FileNotFoundError:
        pass

    print("[MACHINE] Bu bilgisayar için bir isim bulunamadı.")
    print("Örnek: home-pc, school-pc, laptop")
    
    while True:
        name = input("Makine adı: ").strip()

        if name:
            break

        print("Makine adı boş bırakılamaz.")

    with open(MACHINE_FILE, "w", encoding="utf-8") as file:
        file.write(name)

    print(f"✓ Makine adı kaydedildi: {name}\n")

    return name


def detect_environment(machine_name):
    system = platform.system()

    print("[ENVIRONMENT]")
    print(f"Operating system : {system}")
    print(f"Platform         : {platform.platform()}")
    print(f"Python           : {platform.python_version()}")
    print(f"Machine          : {machine_name}")

    if system == "Windows":
        print("Environment      : Windows")

    elif system == "Linux":
        print("Environment      : Linux")

        if shutil.which("pacman"):
            print("Distribution     : Arch-based Linux")
        else:
            print("Distribution     : Unknown Linux")

    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)

    print()


def check_git():
    print("[CHECK] Git kontrol ediliyor...")

    if shutil.which("git") is None:
        print("Git bulunamadı.")
        sys.exit(1)

    result = run_git("--version")

    if result.returncode != 0:
        print("Git çalıştırılamadı.")
        sys.exit(1)

    print()


def get_git_status():
    result = run_git("status", "--porcelain")

    if result.returncode != 0:
        print("Git status alınamadı.")
        sys.exit(1)

    return result.stdout.strip()


def get_sync_status():
    result = run_git(
        "rev-list",
        "--left-right",
        "--count",
        "HEAD...origin/main"
    )

    if result.returncode != 0:
        print("Branch durumu okunamadı.")
        sys.exit(1)

    ahead, behind = map(int, result.stdout.strip().split())

    return ahead, behind


def sync_git(status):
    print("[GIT] GitHub kontrol ediliyor...\n")

    fetch = run_git("fetch", "origin")

    if fetch.returncode != 0:
        print("GitHub'a ulaşılamadı.")
        sys.exit(1)

    ahead, behind = get_sync_status()

    print(f"Local yeni commitler : {ahead}")
    print(f"GitHub yeni commitler: {behind}")

    if behind > 0 and ahead == 0:

        if status:
            print("\n⚠ Yerel değişiklikler var.")
            print("Otomatik pull yapılmayacak.")
            print("Önce değişikliklerini commit veya stash et.")
            sys.exit(1)

        print("\nGitHub'daki yeni sürüm çekiliyor...\n")

        pull = run_git("pull", "--ff-only")

        if pull.returncode != 0:
            print("Pull başarısız oldu.")
            sys.exit(1)

        print("✓ GitHub sürümü alındı.")

    elif behind > 0 and ahead > 0:

        print("\n⚠ Local ve GitHub birbirinden ayrılmış.")
        print("Otomatik işlem yapılmadı.")
        sys.exit(1)

    elif ahead > 0:

        print("\nLocal'de GitHub'a gönderilmemiş commitler var.")
        print("Push işlemini autogit.py halledebilir.")

    else:

        print("\n✓ GitHub ile güncelsin.")

    print()


def install_requirements():
    print("[PYTHON] Bağımlılıklar kontrol ediliyor...\n")

    try:
        with open(
            "requirements.txt",
            "r",
            encoding="utf-8"
        ) as file:

            requirements = [
                line.strip()
                for line in file
                if line.strip() and not line.startswith("#")
            ]

    except FileNotFoundError:

        print("requirements.txt bulunamadı.")
        print("Bağımlılık kurulumu atlandı.")
        return

    if not requirements:

        print("requirements.txt boş.")
        return

    print("Gerekli paketler:")

    for package in requirements:
        print(f"  - {package}")

    print("\nPip çalıştırılıyor...\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt"
        ]
    )

    if result.returncode != 0:
        print("\n❌ Paket kurulumu başarısız.")
        sys.exit(1)

    print("\n✓ Python bağımlılıkları hazır.")


def main():

    print("=" * 60)
    print("       SUDOKU SOLVER - DEVELOPMENT START")
    print("=" * 60)
    print()

    # 1
    machine_name = get_machine_name()

    # 2
    detect_environment(machine_name)

    # 3
    check_git()

    # 4
    print("[1/3] Git çalışma durumu kontrol ediliyor...\n")

    status = get_git_status()

    if status:
        print("Yerel değişiklikler:")
        print(status)

    else:
        print("Çalışma alanı temiz.")

    print()

    # 5
    print("[2/3] Git senkronizasyonu...\n")

    sync_git(status)

    # 6
    print("[3/3] Python bağımlılıkları...\n")

    install_requirements()

    print("\n" + "=" * 60)
    print("       DEVELOPMENT ENVIRONMENT READY")
    print("=" * 60)


if __name__ == "__main__":
    main()