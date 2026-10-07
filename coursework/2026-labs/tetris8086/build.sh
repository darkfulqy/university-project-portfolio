#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v nasm >/dev/null 2>&1; then
  echo '缺少 NASM。macOS 请执行：brew install nasm'
  exit 1
fi
mkdir -p build
nasm -f bin -Werror -l build/tetris.lst main.asm -o build/TETRIS.COM
echo "汇编成功：$(pwd)/build/TETRIS.COM"
wc -c < build/TETRIS.COM
