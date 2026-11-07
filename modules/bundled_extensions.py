"""Installs the extensions listed in bundled-extensions.txt, and guards the webui's core packages
while extensions install their own.

Runs from launch.py (prepare_environment), before the extensions' install.py scripts:

- install_missing() puts every listed extension that is not in the extensions folder yet in
  place. When the user data lives outside the code (--data-dir, as in the Docker image) and the
  code's own extensions folder already has a copy, that copy is used, so containers whose data
  volume predates an image that bundles an extension get it without network access. Otherwise
  the repository is cloned and the pinned commit checked out. Failures are reported and skipped:
  the next start tries again.
- apply_pip_constraints() points PIP_CONSTRAINT at a constraints file pinning the installed
  torch, torchvision and numpy, plus bundled-extensions-constraints.txt. Every pip run started
  from this process inherits it, including the install.py scripts and installs from the
  Extensions tab.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import subprocess
from typing import NamedTuple

from modules.paths_internal import script_path, extensions_dir

git = os.environ.get('GIT', "git")
manifest_file = os.environ.get('BUNDLED_EXTENSIONS_FILE', os.path.join(script_path, "bundled-extensions.txt"))
constraints_file = os.environ.get('BUNDLED_EXTENSIONS_CONSTRAINTS', os.path.join(script_path, "bundled-extensions-constraints.txt"))

# Names of the bundled extensions that have been in the extensions folder. One listed here but
# missing from the folder was removed by the user, and is not installed again.
state_file = os.path.join(extensions_dir, ".bundled-extensions.json")

# The code's own extensions folder. With --data-dir it differs from extensions_dir, and in the
# Docker image it holds the extensions installed at build time.
seed_dir = os.path.join(script_path, "extensions")

# Packages whose installed version is pinned while extensions install theirs.
core_packages = ["torch", "torchvision", "numpy"]

partial_suffix = ".partial"


class BundledExtension(NamedTuple):
    name: str
    url: str
    commit: str | None


def read_manifest(path=manifest_file) -> list[BundledExtension]:
    result = []
    if not os.path.isfile(path):
        return result

    with open(path, encoding="utf8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            fields = line.split()
            name = fields[0]
            if len(fields) not in (2, 3) or name in (".", "..") or name.startswith(".") or re.search(r'[\\/:]', name):
                print(f"{path}:{line_number}: expected '<folder name> <git URL> [commit]', ignoring: {line}")
                continue

            result.append(BundledExtension(name, fields[1], fields[2] if len(fields) == 3 else None))

    return result


def normalize_url(url: str) -> str:
    url = url.strip().rstrip("/").lower()
    url = re.sub(r'\.git$', "", url)
    return re.sub(r'^(https?://|git@|ssh://git@)', "", url).replace(":", "/")


def origin_urls() -> set[str]:
    """Normalized remote URLs of the git repositories in the extensions folder."""

    urls = set()
    for dirname in os.listdir(extensions_dir):
        config = os.path.join(extensions_dir, dirname, ".git", "config")
        try:
            with open(config, encoding="utf8", errors="replace") as file:
                urls.update(normalize_url(url) for url in re.findall(r'^\s*url\s*=\s*(\S+)', file.read(), re.MULTILINE))
        except OSError:
            pass

    return urls


def run_git(*args, cwd=None):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # a moved or private repository fails instead of asking for a password
    result = subprocess.run([git, *args], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def clone(extension: BundledExtension, dest: str):
    # blob:none fetches file contents only for the commits checked out, which keeps the clone
    # small; servers without support do a full clone instead.
    run_git("clone", "--filter=blob:none", "--config", "core.filemode=false", extension.url, dest)

    if extension.commit:
        # Reset the default branch rather than checking out a detached commit, so the Extensions
        # tab can still check for and apply updates.
        try:
            run_git("reset", "--hard", extension.commit, cwd=dest)
        except RuntimeError:
            print(f"Commit {extension.commit} of {extension.name} not found; using the latest commit instead.")

    if os.path.isfile(os.path.join(dest, ".gitmodules")):
        run_git("submodule", "update", "--init", "--recursive", cwd=dest)


def remove_partial_dirs():
    """Removes what an interrupted install left behind (it would load as a broken extension)."""

    for dirname in os.listdir(extensions_dir):
        if dirname.startswith(".") and dirname.endswith(partial_suffix):
            shutil.rmtree(os.path.join(extensions_dir, dirname), ignore_errors=True)


def install_one(extension: BundledExtension):
    dest = os.path.join(extensions_dir, extension.name)
    partial = os.path.join(extensions_dir, f".{extension.name}{partial_suffix}")
    shutil.rmtree(partial, ignore_errors=True)

    seed = os.path.join(seed_dir, extension.name)
    use_seed = os.path.normcase(os.path.realpath(seed_dir)) != os.path.normcase(os.path.realpath(extensions_dir)) and os.path.isdir(os.path.join(seed, ".git"))

    try:
        if use_seed:
            print(f"Installing bundled extension {extension.name} from {seed}")
            shutil.copytree(seed, partial, symlinks=True)
        else:
            print(f"Installing bundled extension {extension.name} from {extension.url}")
            clone(extension, partial)

        os.replace(partial, dest)
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def load_state() -> set[str]:
    try:
        with open(state_file, encoding="utf8") as file:
            return set(json.load(file).get("installed", []))
    except (OSError, ValueError, AttributeError):
        return set()


def save_state(installed: set[str]):
    try:
        with open(state_file, "w", encoding="utf8") as file:
            json.dump({"installed": sorted(installed)}, file, indent=4)
    except OSError as e:
        print(f"Couldn't save {state_file}: {e}")


def install_missing():
    extensions = read_manifest()
    if not extensions:
        return

    os.makedirs(extensions_dir, exist_ok=True)
    remove_partial_dirs()

    known = load_state()
    installed = set(known)
    removed = []
    failed = []
    urls = None

    for extension in extensions:
        if os.path.isdir(os.path.join(extensions_dir, extension.name)):
            installed.add(extension.name)
            continue

        if extension.name in known:
            removed.append(extension.name)
            continue

        # Already installed under a different folder name, e.g. from the Extensions tab.
        if urls is None:
            urls = origin_urls()
        if normalize_url(extension.url) in urls:
            installed.add(extension.name)
            continue

        try:
            install_one(extension)
            installed.add(extension.name)
        except Exception as e:
            failed.append(extension.name)
            print(f"Couldn't install bundled extension {extension.name}; it will be retried on the next start. {e}")

    if installed != known:
        save_state(installed)

    if removed:
        print(f"Bundled extensions not reinstalled because they were removed: {', '.join(removed)}. Delete {state_file} to reinstall them.")
    if failed:
        print(f"Bundled extensions that failed to install: {', '.join(failed)}")


def apply_pip_constraints():
    lines = []
    for package in core_packages:
        try:
            lines.append(f"{package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            pass

    if os.path.isfile(constraints_file):
        with open(constraints_file, encoding="utf8") as file:
            lines.append(file.read())

    path = os.path.join(script_path, "tmp", "extension-constraints.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as file:
        file.write("# Generated by modules/bundled_extensions.py on every start.\n" + "\n".join(lines) + "\n")

    # PIP_CONSTRAINT is a whitespace-separated list; a file: URL keeps a path with spaces whole.
    entry = path if not re.search(r'\s', path) else pathlib.Path(path).as_uri()
    existing = os.environ.get('PIP_CONSTRAINT', "").split()
    if entry not in existing:
        os.environ['PIP_CONSTRAINT'] = " ".join(existing + [entry])
