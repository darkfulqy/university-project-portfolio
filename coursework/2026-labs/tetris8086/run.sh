#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if command -v nasm >/dev/null 2>&1; then
  bash ./build.sh
elif [[ ! -f build/TETRIS.COM ]]; then
  echo '缺少 NASM 和已编译程序。macOS 请执行：brew install nasm dosbox-x'
  exit 1
fi
if command -v dosbox-x >/dev/null 2>&1; then
  tetris_emulator="$(command -v dosbox-x)"
elif [[ -x /Applications/dosbox-x.app/Contents/MacOS/dosbox-x ]]; then
  tetris_emulator=/Applications/dosbox-x.app/Contents/MacOS/dosbox-x
else
  echo '缺少 DOSBox-X。macOS 请执行：brew install dosbox-x'
  echo '也可从 https://dosbox-x.com/ 下载。'
  exit 1
fi
# 以项目为工作目录，MOUNT 使用相对路径，兼容当前中文课程目录。
exec "$tetris_emulator" -conf dosbox-x.conf -fastlaunch \
  -c 'mount c "./build"' -c 'c:' -c 'TETRIS.COM' -c 'exit'
