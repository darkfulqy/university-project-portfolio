#!/usr/bin/env python3
"""Integration checks of the full COM file with only BIOS/DOS calls stubbed."""

from __future__ import annotations

import collections
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from unicorn import UC_HOOK_CODE, UC_HOOK_INTR, UcError
from unicorn.x86_const import (
    UC_X86_REG_AX, UC_X86_REG_BX, UC_X86_REG_CX, UC_X86_REG_DX,
    UC_X86_REG_DS, UC_X86_REG_ES, UC_X86_REG_SP, UC_X86_REG_IP,
    UC_X86_REG_EFLAGS,
)

from test_game import Engine, PROJECT, REGISTERS, SYMBOLS


MAIN_SYMBOLS = SYMBOLS + (
    "start", "main_loop", "handle_key", "render", "title_screen", "paused",
    "dirty", "last_tick", "screen_buffer", "old_mode", "old_page",
    "old_cursor", "old_cursor_pos", "shape_colors", "shapes",
)


def assemble_main() -> tuple[bytes, dict[str, int]]:
    nasm = shutil.which("nasm")
    if not nasm:
        raise RuntimeError("未找到 nasm")
    source = '\n'.join([
        '%include "main.asm"', 'db "MAINTEST"',
        *("dw " + name for name in MAIN_SYMBOLS), "",
    ])
    with tempfile.TemporaryDirectory(prefix="tetris8086-main-test-") as temporary:
        asm_path = Path(temporary) / "main_test.asm"
        output_path = Path(temporary) / "main_test.com"
        asm_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [nasm, "-f", "bin", "-Werror", "-I", str(PROJECT) + "/",
             "-o", str(output_path), str(asm_path)], capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError("完整 COM 测试入口汇编失败：\n" + result.stderr)
        binary = output_path.read_bytes()
    offset = len(binary) - 2 * len(MAIN_SYMBOLS)
    if binary[offset - 8:offset] != b"MAINTEST":
        raise RuntimeError("完整 COM 符号表损坏")
    addresses = struct.unpack_from("<" + "H" * len(MAIN_SYMBOLS), binary, offset)
    return binary, dict(zip(MAIN_SYMBOLS, addresses))


class MainEngine(Engine):
    VIDEO = 0xB8000

    def __init__(self, binary: bytes, symbols: dict[str, int]):
        self.ticks = 1234
        self.midnight = 0
        self.keyboard = collections.deque()
        self.video_calls = []
        self.exited = False
        self.idled = False
        super().__init__(binary, symbols)
        self.cpu.reg_write(UC_X86_REG_ES, self.SEGMENT)
        self.cpu.hook_add(UC_HOOK_INTR, self.interrupt)
        self.cpu.hook_add(UC_HOOK_CODE, self.instruction)
        self.cpu.mem_write(self.VIDEO - 16, b"\xa5" * 16)
        self.cpu.mem_write(self.VIDEO + 4000, b"\xa5" * 16)

    def interrupt(self, cpu, number, _data):
        ax = cpu.reg_read(UC_X86_REG_AX)
        ah = ax >> 8
        if number == 0x10:
            self.video_calls.append((ax, cpu.reg_read(UC_X86_REG_BX),
                                     cpu.reg_read(UC_X86_REG_CX),
                                     cpu.reg_read(UC_X86_REG_DX)))
            if ah == 0x0F:
                cpu.reg_write(UC_X86_REG_AX, 0x5003)
                cpu.reg_write(UC_X86_REG_BX, 0x0200)
            elif ah == 3:
                cpu.reg_write(UC_X86_REG_CX, 0x0607)
                cpu.reg_write(UC_X86_REG_DX, 0x0A0B)
        elif number == 0x1A:
            if ah != 0:
                raise AssertionError(f"未知的时钟 BIOS 功能 {ah:02x}")
            cpu.reg_write(UC_X86_REG_CX, (self.ticks >> 16) & 0xFFFF)
            cpu.reg_write(UC_X86_REG_DX, self.ticks & 0xFFFF)
            cpu.reg_write(UC_X86_REG_AX, self.midnight)
            self.midnight = 0
        elif number == 0x16:
            if ah == 1:
                flags = cpu.reg_read(UC_X86_REG_EFLAGS)
                cpu.reg_write(UC_X86_REG_EFLAGS,
                              flags & ~0x40 if self.keyboard else flags | 0x40)
                if self.keyboard:
                    cpu.reg_write(UC_X86_REG_AX, self.keyboard[0])
            elif ah == 0:
                if not self.keyboard:
                    raise AssertionError("程序在没有待处理按键时调用阻塞键盘读取")
                cpu.reg_write(UC_X86_REG_AX, self.keyboard.popleft())
            else:
                raise AssertionError(f"未知的键盘 BIOS 功能 {ah:02x}")
        elif number == 0x21 and ah == 0x4C:
            self.exited = True
            cpu.emu_stop()
        else:
            raise AssertionError(f"未模拟的中断 {number:02x}, AX={ax:04x}")

    def instruction(self, cpu, address, _size, _data):
        if bytes(cpu.mem_read(address, 1)) == b"\xf4":
            self.idled = True
            cpu.emu_stop()

    def call(self, name: str, **registers: int) -> dict[str, int]:
        initial_es = self.cpu.reg_read(UC_X86_REG_ES)
        for key, value in registers.items():
            self.cpu.reg_write(REGISTERS[key], value & 0xFFFF)
        self.cpu.reg_write(UC_X86_REG_SP, self.STACK_IP)
        self.cpu.reg_write(UC_X86_REG_EFLAGS, 2)
        self.cpu.mem_write(self.BASE + self.STACK_IP, struct.pack("<H", self.RETURN_IP))
        try:
            self.cpu.emu_start(self.address(name), self.BASE + self.RETURN_IP,
                               timeout=2_000_000, count=500_000)
        except UcError as exc:
            raise AssertionError(f"{name} 执行失败：{exc}") from exc
        if self.exited:
            return {}
        if self.cpu.reg_read(UC_X86_REG_IP) != self.RETURN_IP:
            raise AssertionError(f"{name} 没有返回")
        if self.cpu.reg_read(UC_X86_REG_SP) != self.STACK_IP + 2:
            raise AssertionError(f"{name} 没有恢复栈")
        expected_es = self.SEGMENT if name == "render" else initial_es
        if self.cpu.reg_read(UC_X86_REG_ES) != expected_es:
            raise AssertionError(f"{name} 没有恢复 ES")
        if self.cpu.reg_read(UC_X86_REG_DS) != self.SEGMENT:
            raise AssertionError(f"{name} 修改了 DS")
        return {key: self.cpu.reg_read(register) for key, register in REGISTERS.items()}

    def boot(self) -> None:
        self.cpu.emu_start(self.address("start"), self.address("main_loop"),
                           timeout=2_000_000, count=500_000)
        if self.cpu.reg_read(UC_X86_REG_IP) != self.symbols["main_loop"]:
            raise AssertionError("启动程序没有到达事件循环")

    def tick(self, ticks: int, key: int | None = None, midnight: int = 0) -> None:
        self.ticks = ticks
        self.midnight = midnight
        self.idled = False
        if key is not None:
            self.keyboard.append(key)
        self.cpu.emu_start(self.address("main_loop"), 0,
                           timeout=2_000_000, count=500_000)
        if not self.idled and not self.exited:
            raise AssertionError("事件循环没有进入 HLT 等待")

    def text_at(self, row: int, column: int, length: int) -> str:
        address = self.VIDEO + (row * 80 + column) * 2
        return bytes(self.cpu.mem_read(address, length * 2))[::2].decode("cp437")

    def video_cell(self, x: int, y: int) -> bytes:
        address = self.VIDEO + ((y + 3) * 80 + 30 + x * 2) * 2
        return bytes(self.cpu.mem_read(address, 4))

    def assert_video_guards(self) -> None:
        for address in (self.VIDEO - 16, self.VIDEO + 4000):
            if bytes(self.cpu.mem_read(address, 16)) != b"\xa5" * 16:
                raise AssertionError("文本显存复制发生越界")
        self.assert_guards()


class MainIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary, cls.symbols = assemble_main()

    def setUp(self) -> None:
        self.game = MainEngine(self.binary, self.symbols)
        self.game.boot()

    def start_play(self) -> None:
        self.game.call("handle_key", ax=13)

    def test_boot_title_gates_keys_and_gravity_until_enter(self) -> None:
        game = self.game
        self.assertEqual(game.byte("title_screen"), 1)
        self.assertEqual(game.byte("old_mode"), 3)
        self.assertEqual(game.byte("old_page"), 2)
        self.assertEqual(game.word("old_cursor"), 0x0607)
        self.assertEqual(game.word("old_cursor_pos"), 0x0A0B)
        game.tick(5000, ord("a"))
        self.assertEqual(game.word("piece_x"), 3)
        self.assertEqual(game.word("piece_y"), 0)
        self.assertEqual(game.text_at(10, 34, 14), "READY TO STACK")
        game.tick(5001, 13)
        self.assertEqual(game.byte("title_screen"), 0)
        self.assertEqual(game.word("last_tick"), 5001)
        self.assertEqual(game.word("piece_y"), 0)
        self.assertEqual(game.text_at(23, 56, 7), "PLAYING")
        game.assert_video_guards()

    def test_pause_blocks_input_and_time_and_resume_resets_clock(self) -> None:
        game = self.game
        self.start_play()
        game.piece(2, x=3, y=4)
        game.tick(2000, ord("p"))
        self.assertEqual(game.byte("paused"), 1)
        self.assertEqual(game.text_at(10, 37, 6), "PAUSED")
        for index, key in enumerate((ord("a"), ord("d"), ord("w"), ord("s"), 0x5000, 32)):
            game.tick(3000 + index * 100, key)
        self.assertEqual(game.word("piece_x"), 3)
        self.assertEqual(game.word("piece_y"), 4)
        self.assertEqual(game.byte("piece_rot"), 0)
        self.assertEqual(game.read("board", 200), bytes(200))
        self.assertEqual(game.word("score"), 0)
        game.tick(5000, ord("P"))
        self.assertEqual(game.byte("paused"), 0)
        self.assertEqual(game.word("last_tick"), 5000)
        self.assertEqual(game.word("piece_y"), 4)
        game.tick(5009)
        self.assertEqual(game.word("piece_y"), 4)
        game.tick(5010)
        self.assertEqual(game.word("piece_y"), 5)

    def test_restart_from_pause_resets_board_score_and_modes(self) -> None:
        game = self.game
        self.start_play()
        game.write("board", bytes([5] * 200))
        game.set_word("score", 1000)
        game.set_word("lines", 42)
        game.set_byte("paused", 1)
        game.tick(6000, ord("R"))
        self.assertEqual(game.read("board", 200), bytes(200))
        self.assertEqual(game.word("score"), 0)
        self.assertEqual(game.word("lines"), 0)
        self.assertEqual(game.word("level"), 1)
        self.assertEqual(game.byte("paused"), 0)
        self.assertEqual(game.byte("title_screen"), 0)
        self.assertEqual(game.word("last_tick"), 6000)
        self.assertEqual(game.text_at(7, 4, 5), "00000")

    def test_game_over_gates_movement_then_enter_restarts(self) -> None:
        game = self.game
        self.start_play()
        game.set_byte("game_over", 1)
        game.set_byte("dirty", 1)
        game.tick(3000, ord("a"))
        self.assertEqual(game.word("piece_x"), 3)
        self.assertEqual(game.text_at(10, 35, 9), "GAME OVER")
        game.tick(4000, 13)
        self.assertEqual(game.byte("game_over"), 0)
        self.assertEqual(game.word("piece_y"), 0)
        self.assertEqual(game.word("last_tick"), 4000)
        self.assertEqual(game.text_at(23, 56, 7), "PLAYING")

    def test_arrows_wasd_and_hard_drop_reach_core(self) -> None:
        game = self.game
        self.start_play()
        game.piece(2, x=3, y=0)
        for key, x in ((ord("A"), 2), (0x4D00, 3), (0x4B00, 2), (ord("d"), 3)):
            game.call("handle_key", ax=key)
            self.assertEqual(game.word("piece_x"), x)
        for key, rotation in ((ord("W"), 1), (0x4800, 2)):
            game.call("handle_key", ax=key)
            self.assertEqual(game.byte("piece_rot"), rotation)
        for key, y in ((ord("s"), 1), (0x5000, 2)):
            game.call("handle_key", ax=key)
            self.assertEqual(game.word("piece_y"), y)
        self.assertEqual(game.word("score"), 2)
        game.ticks = 7000
        game.call("handle_key", ax=32)
        self.assertEqual(sum(value != 0 for value in game.read("board", 200)), 4)
        self.assertEqual(game.word("piece_y"), 0)
        self.assertEqual(game.word("last_tick"), 7000)

    def test_frame_has_scores_board_active_ghost_and_next_preview(self) -> None:
        game = self.game
        self.start_play()
        game.piece(1, x=3, y=0)
        game.set_byte("next_type", 0)
        game.set_word("score", 65535)
        game.set_word("lines", 12345)
        game.set_word("level", 20)
        board = bytearray(200)
        board[10 * 10 + 2] = 6
        game.write("board", bytes(board))
        game.call("render")
        self.assertEqual(game.read("screen_buffer", 4000), bytes(game.cpu.mem_read(game.VIDEO, 4000)))
        self.assertEqual(game.text_at(6, 4, 9), "S C O R E")
        self.assertEqual(game.text_at(7, 4, 5), "65535")
        self.assertEqual(game.text_at(11, 4, 5), "12345")
        self.assertEqual(game.text_at(15, 4, 5), "00020")
        self.assertEqual(game.video_cell(2, 10), b"\xdb\x09\xdb\x09")
        self.assertEqual(game.video_cell(3, 10), b"\xfa\x08 \x08")
        for x in (4, 5):
            for y in (0, 1):
                self.assertEqual(game.video_cell(x, y), b"\xdb\x0e\xdb\x0e")
            for y in (18, 19):
                self.assertEqual(game.video_cell(x, y), b"\xb0\x08\xb0\x08")
        preview = bytes(game.cpu.mem_read(game.VIDEO + (7 * 80 + 58) * 2, 16))
        self.assertEqual(preview, b"\xdb\x0b" * 8)
        self.assertEqual(game.read("board", 200), bytes(board))
        game.assert_video_guards()

    def test_gravity_handles_low_word_rollover_and_midnight(self) -> None:
        game = self.game
        self.start_play()
        game.piece(1, x=3, y=0)
        game.set_word("last_tick", 65530)
        game.tick(65539)
        self.assertEqual(game.word("piece_y"), 0)
        game.tick(65540)
        self.assertEqual(game.word("piece_y"), 1)
        self.assertEqual(game.word("last_tick"), 4)
        game.set_word("last_tick", 0x00AF)
        game.tick(0, midnight=1)
        self.assertEqual(game.word("piece_y"), 1)
        self.assertEqual(game.word("last_tick"), 0)
        game.tick(9)
        self.assertEqual(game.word("piece_y"), 1)
        game.tick(10)
        self.assertEqual(game.word("piece_y"), 2)

    def test_key_traffic_does_not_starve_gravity(self) -> None:
        game = self.game
        self.start_play()
        game.piece(1, x=3, y=0)
        game.set_word("last_tick", 100)
        game.tick(110, ord("a"))
        self.assertEqual(game.word("piece_x"), 2)
        self.assertEqual(game.word("piece_y"), 1)
        self.assertEqual(game.word("last_tick"), 110)

    def test_escape_restores_saved_display_state_and_exits_dos(self) -> None:
        game = self.game
        game.call("handle_key", ax=27)
        self.assertTrue(game.exited)
        restore = game.video_calls[-4:]
        self.assertEqual(restore[0][0], 0x0003)
        self.assertEqual(restore[1][0], 0x0502)
        self.assertEqual(restore[2][0] >> 8, 1)
        self.assertEqual(restore[2][2], 0x0607)
        self.assertEqual(restore[3][0] >> 8, 2)
        self.assertEqual(restore[3][1] >> 8, 2)
        self.assertEqual(restore[3][3], 0x0A0B)


if __name__ == "__main__":
    unittest.main(verbosity=2)
