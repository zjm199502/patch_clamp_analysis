#!/usr/bin/env python
"""升级版本号并在 CHANGELOG.md 顶部加一节。用法：python tools/bump_version.py [patch|minor|major]

每个小版本（末位 +1）都会进入软件内更新通道；末位是 10 的倍数（x.y.0、x.y.10、x.y.20…）
的版本同时发布带安装包的正式 Release。
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT = ROOT / "patchclamp_gui" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"


def bump(version: str, part: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "patch":
        patch += 1
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "major":
        major, minor, patch = major + 1, 0, 0
    else:
        raise SystemExit(__doc__)
    return f"{major}.{minor}.{patch}"


def main() -> None:
    part = sys.argv[1] if len(sys.argv) > 1 else "patch"
    text = INIT.read_text(encoding="utf-8")
    old = re.search(r'__version__ = "([^"]+)"', text).group(1)
    new = bump(old, part)
    INIT.write_text(text.replace(f'__version__ = "{old}"', f'__version__ = "{new}"'), encoding="utf-8")
    log = CHANGELOG.read_text(encoding="utf-8")
    head = f"## v{new}\n\n- \n\n"
    idx = log.index("## v")
    CHANGELOG.write_text(log[:idx] + head + log[idx:], encoding="utf-8")
    kind = "第 10 个小版本：会同时发布带安装包的正式 Release" if int(new.split(".")[-1]) % 10 == 0 else "小版本：只进入软件内更新通道"
    print(f"{old} -> {new}（{kind}）。请补全 CHANGELOG.md 中的条目。")


if __name__ == "__main__":
    main()
