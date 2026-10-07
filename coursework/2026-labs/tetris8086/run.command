#!/usr/bin/env bash
cd -- "$(dirname -- "$0")" || exit 1
bash ./run.sh
tetris_status=$?
if [[ $tetris_status -ne 0 ]]; then
  printf '\n启动失败，请查看上面的提示。按回车关闭。\n'
  read -r
fi
exit "$tetris_status"
