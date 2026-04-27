"""LlamaCppManager — auto-update llama.cpp release binaries.

Layout: <runtime_parent>/llama.cpp-<tag>/llama-server.exe (+ DLLs)

On backend startup, ensure_latest_available() checks GitHub for the newest
release tag and downloads it in the background if newer than what's installed.
Multiple installed versions can coexist; current_exe_path() returns the
highest-numbered build's exe.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Asset name pattern. The "llama-{tag}-bin-win-cuda-13.1-x64.zip" variant
# (~170 MB) contains llama-server.exe + DLLs (llama.dll, ggml-*.dll). It does
# NOT include CUDA runtime — relies on system having CUDA installed (which the
# user does — torch's bundled cudart works at minimum, plus most users with
# llama.cpp use cases have a real CUDA toolkit).
#
# (The companion "cudart-llama-bin-..." asset only contains 3 CUDA runtime DLLs;
#  it's *not* a standalone install — confused us once. Don't use it alone.)
ASSET_NAME_TEMPLATE = "llama-{tag}-bin-win-cuda-13.1-x64.zip"

GITHUB_LATEST_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"

RELEASE_URL_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/" + ASSET_NAME_TEMPLATE
)


def _build_number(tag: str) -> int:
    """Extract numeric suffix from a build tag like 'b8941' → 8941. Tags that
    don't match the pattern sort to the bottom (-1)."""
    m = re.match(r"^b(\d+)$", tag)
    return int(m.group(1)) if m else -1


class LlamaCppManager:
    """Manages installed llama.cpp release directories.

    Designed to be a singleton in production (instantiated once at app
    startup) but takes runtime_parent as a constructor arg for testability.
    """

    def __init__(self, runtime_parent: Path):
        self.runtime_parent = Path(runtime_parent)
        self.runtime_parent.mkdir(parents=True, exist_ok=True)

    # ---- query installed state ----

    def installed_versions(self) -> List[str]:
        """List tags of installed builds (those with llama-server.exe)."""
        if not self.runtime_parent.exists():
            return []
        out: List[str] = []
        for entry in self.runtime_parent.iterdir():
            if not entry.is_dir() or not entry.name.startswith("llama.cpp-"):
                continue
            tag = entry.name[len("llama.cpp-"):]
            if (entry / "llama-server.exe").is_file():
                out.append(tag)
        return out

    def current_version(self) -> Optional[str]:
        """The highest-build-numbered installed tag, or None if nothing installed."""
        versions = self.installed_versions()
        if not versions:
            return None
        return max(versions, key=_build_number)

    def current_exe_path(self) -> Optional[Path]:
        """Path to the current version's llama-server.exe."""
        v = self.current_version()
        if v is None:
            return None
        return self.runtime_parent / f"llama.cpp-{v}" / "llama-server.exe"

    # ---- remote check ----

    def latest_remote(self, timeout: float = 10.0) -> Optional[str]:
        """Fetch the latest release tag from GitHub. Returns None on any failure."""
        try:
            req = urllib.request.Request(
                GITHUB_LATEST_URL,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "lumenx"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            tag = data.get("tag_name")
            if not isinstance(tag, str) or not tag:
                return None
            return tag
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Could not fetch latest llama.cpp release: {e}")
            return None

    # ---- update ----

    async def ensure_latest_available(self) -> Optional[str]:
        """Check GitHub, download newer release in background if available.

        Never raises. Returns the version that ended up being current
        (could be the just-downloaded one, or the previously-installed one
        if download failed / network unreachable, or None if nothing at all).
        """
        try:
            remote_tag = self.latest_remote()
        except Exception:
            remote_tag = None

        local_tag = self.current_version()

        # No remote info → use whatever's local (could be None)
        if remote_tag is None:
            if local_tag is None:
                logger.warning("No llama.cpp installed and GitHub unreachable")
            else:
                logger.info(f"Using locally-installed llama.cpp {local_tag} (no remote check)")
            return local_tag

        # Local already at-or-newer than remote (shouldn't normally happen
        # for "latest", but be defensive)
        if local_tag is not None and _build_number(local_tag) >= _build_number(remote_tag):
            logger.info(f"llama.cpp {local_tag} is already up to date (remote: {remote_tag})")
            return local_tag

        # Need to download — run in executor since it's blocking I/O
        logger.info(f"Downloading llama.cpp {remote_tag} (replacing {local_tag or 'nothing'})")
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._download_and_extract, remote_tag
            )
            logger.info(f"Installed llama.cpp {remote_tag}")
            return remote_tag
        except Exception as e:
            logger.error(f"Failed to install llama.cpp {remote_tag}: {e}")
            # Fall back to whatever's locally available
            return self.current_version()

    def _download_and_extract(self, tag: str) -> Path:
        """Blocking. Download release zip + extract to runtime_parent/llama.cpp-<tag>/.

        Atomic-ish: extract to a temp dir, then rename. If anything fails,
        the partial extract is cleaned up. Returns the install dir.
        """
        url = RELEASE_URL_TEMPLATE.format(tag=tag)
        target = self.runtime_parent / f"llama.cpp-{tag}"
        staging = self.runtime_parent / f".llama.cpp-{tag}.staging"

        # Clean any prior staging from a previous failed attempt
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        try:
            logger.info(f"GET {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "lumenx"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Stream into memory then unzip (zips are self-contained)
                # llama.cpp zips are 100-400 MB so memory is fine on dev boxes.
                buf = io.BytesIO(resp.read())

            with zipfile.ZipFile(buf) as zf:
                zf.extractall(staging)

            # Verify llama-server.exe was extracted (may be in a subdir)
            exe = self._find_exe(staging)
            if exe is None:
                raise RuntimeError(
                    f"zip from {url} did not contain llama-server.exe"
                )

            # If the exe is in a nested subdir, flatten it up
            if exe.parent != staging:
                # Move all files from exe's parent up to staging
                src_dir = exe.parent
                for item in src_dir.iterdir():
                    shutil.move(str(item), str(staging / item.name))
                # Remove now-empty subdir tree
                try:
                    src_dir.rmdir()
                except OSError:
                    pass

            # Atomic rename. If target exists (race / re-install), wipe it.
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
            return target
        except Exception:
            # Clean up staging on failure
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _find_exe(root: Path) -> Optional[Path]:
        """Locate llama-server.exe under root (could be at top or in subdir)."""
        for p in root.rglob("llama-server.exe"):
            return p
        return None

    # ---- cleanup ----

    def cleanup_old(self, keep: int = 2) -> List[str]:
        """Delete all but the `keep` newest installed versions. Returns deleted tags."""
        versions = sorted(self.installed_versions(), key=_build_number, reverse=True)
        to_delete = versions[keep:]
        deleted: List[str] = []
        for tag in to_delete:
            d = self.runtime_parent / f"llama.cpp-{tag}"
            try:
                shutil.rmtree(d)
                deleted.append(tag)
                logger.info(f"Cleaned up old llama.cpp install: {tag}")
            except OSError as e:
                logger.warning(f"Could not delete {d}: {e}")
        return deleted
