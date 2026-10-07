#!/usr/bin/env python3
"""Execute the real 8086 game engine under Unicorn, without DOS or a display."""

from __future__ import annotations

import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from unicorn import Uc, UcError, UC_ARCH_X86, UC_MODE_16
    from unicorn.x86_const import (
        UC_X86_REG_AX, UC_X86_REG_BX, UC_X86_REG_CX, UC_X86_REG_DX,
        UC_X86_REG_SI, UC_X86_REG_DI, UC_X86_REG_BP, UC_X86_REG_SP,
        UC_X86_REG_CS, UC_X86_REG_DS, UC_X86_REG_ES, UC_X86_REG_SS,
        UC_X86_REG_IP, UC_X86_REG_EFLAGS,
    )
except ImportError as exc:
    raise SystemExit(
        "缺少 Unicorn：请运行 .venv/bin/python -m pip install unicorn"
    ) from exc


PROJECT = Path(__file__).resolve().parents[1]
SYMBOLS = (
    "board", "board_guard_before", "board_guard_after",
    "piece_type", "piece_rot", "piece_x", "piece_y", "next_type",
    "score", "lines", "level", "drop_ticks", "game_over", "rng_seed", "ghost_y",
    "new_game", "move_left", "move_right", "rotate_piece", "soft_drop",
    "hard_drop", "step_down", "calc_ghost", "check_position", "get_shape",
    "lock_piece", "clear_lines",
)
REGISTERS = {
    "ax": UC_X86_REG_AX, "bx": UC_X86_REG_BX, "cx": UC_X86_REG_CX,
    "dx": UC_X86_REG_DX, "si": UC_X86_REG_SI, "di": UC_X86_REG_DI,
    "bp": UC_X86_REG_BP,
}


def assemble_engine() -> tuple[bytes, dict[str, int]]:
    """The symbol table contains NASM-resolved offsets, with no listing parser."""
    nasm = shutil.which("nasm")
    if not nasm:
        raise RuntimeError("未找到 nasm；macOS 可使用 brew install nasm")
    source = "\n".join([
        "cpu 8086", "bits 16", "org 0x100", "jmp test_entry",
        'db "TETTEST1"', *("dw " + name for name in SYMBOLS),
        "test_entry: hlt", '%include "game.inc"', "",
    ])
    with tempfile.TemporaryDirectory(prefix="tetris8086-test-") as temporary:
        asm_path = Path(temporary) / "test.asm"
        output_path = Path(temporary) / "test.com"
        asm_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [nasm, "-f", "bin", "-Werror", "-I", str(PROJECT) + "/", "-o", str(output_path),
             str(asm_path)], capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError("8086 测试入口汇编失败：\n" + result.stderr)
        binary = output_path.read_bytes()
    table_offset = binary.index(b"TETTEST1") + len(b"TETTEST1")
    offsets = struct.unpack_from("<" + "H" * len(SYMBOLS), binary, table_offset)
    return binary, dict(zip(SYMBOLS, offsets))


class Engine:
    SEGMENT = 0x1000
    BASE = SEGMENT * 16
    RETURN_IP = 0xF100
    STACK_IP = 0xEF00

    def __init__(self, binary: bytes, symbols: dict[str, int], seed: int = 12345):
        self.symbols = symbols
        self.cpu = Uc(UC_ARCH_X86, UC_MODE_16)
        self.cpu.mem_map(0, 2 * 1024 * 1024)
        self.cpu.mem_write(self.BASE + 0x100, binary)
        self.cpu.mem_write(self.BASE + self.RETURN_IP, b"\xf4")
        for register in (UC_X86_REG_CS, UC_X86_REG_DS, UC_X86_REG_SS):
            self.cpu.reg_write(register, self.SEGMENT)
        self.cpu.reg_write(UC_X86_REG_ES, 0x3400)
        self.set_word("rng_seed", seed)
        self.call("new_game")

    def address(self, name: str) -> int:
        return self.BASE + self.symbols[name]

    def read(self, name: str, length: int) -> bytes:
        return bytes(self.cpu.mem_read(self.address(name), length))

    def write(self, name: str, data: bytes) -> None:
        self.cpu.mem_write(self.address(name), data)

    def byte(self, name: str) -> int:
        return self.read(name, 1)[0]

    def word(self, name: str, signed: bool = False) -> int:
        return struct.unpack("<h" if signed else "<H", self.read(name, 2))[0]

    def set_byte(self, name: str, value: int) -> None:
        self.write(name, bytes([value & 0xFF]))

    def set_word(self, name: str, value: int) -> None:
        self.write(name, struct.pack("<H", value & 0xFFFF))

    def call(self, name: str, **registers: int) -> dict[str, int]:
        for key, value in registers.items():
            self.cpu.reg_write(REGISTERS[key], value & 0xFFFF)
        self.cpu.reg_write(UC_X86_REG_SP, self.STACK_IP)
        self.cpu.reg_write(UC_X86_REG_EFLAGS, 2)
        self.cpu.mem_write(self.BASE + self.STACK_IP, struct.pack("<H", self.RETURN_IP))
        try:
            self.cpu.emu_start(
                self.address(name), self.BASE + self.RETURN_IP,
                timeout=2_000_000, count=500_000,
            )
        except UcError as exc:
            raise AssertionError(f"{name} 执行失败：{exc}") from exc
        if self.cpu.reg_read(UC_X86_REG_IP) != self.RETURN_IP:
            raise AssertionError(f"{name} 没有在指令/时间上限内返回")
        if self.cpu.reg_read(UC_X86_REG_SP) != self.STACK_IP + 2:
            raise AssertionError(f"{name} 没有恢复栈")
        if self.cpu.reg_read(UC_X86_REG_ES) != 0x3400:
            raise AssertionError(f"{name} 修改了 ES")
        if self.cpu.reg_read(UC_X86_REG_DS) != self.SEGMENT:
            raise AssertionError(f"{name} 修改了 DS")
        answer = {key: self.cpu.reg_read(register) for key, register in REGISTERS.items()}
        answer["carry"] = self.cpu.reg_read(UC_X86_REG_EFLAGS) & 1
        return answer

    def piece(self, kind: int, rotation: int = 0, x: int = 3, y: int = 0) -> None:
        self.set_byte("piece_type", kind)
        self.set_byte("piece_rot", rotation)
        self.set_word("piece_x", x)
        self.set_word("piece_y", y)

    def cells(self, kind: int, rotation: int = 0) -> list[tuple[int, int]]:
        self.set_byte("piece_type", kind)
        self.set_byte("piece_rot", rotation)
        pointer = self.call("get_shape")["si"]
        data = bytes(self.cpu.mem_read(self.BASE + pointer, 8))
        return list(zip(data[::2], data[1::2]))

    def blocked(self, x: int, y: int, rotation: int, kind: int) -> bool:
        return bool(self.call("check_position", ax=x, bx=y, cx=rotation, dx=kind)["carry"])

    def assert_guards(self) -> None:
        for name in ("board_guard_before", "board_guard_after"):
            if self.read(name, 16) != b"\xa5" * 16:
                raise AssertionError(f"棋盘越界，{name} 被修改")


class GameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary, cls.symbols = assemble_engine()

    def setUp(self) -> None:
        self.game = Engine(self.binary, self.symbols)

    def test_new_game_and_restart_reset_state(self) -> None:
        game = self.game
        game.write("board", b"\x07" * 200)
        for name in ("score", "lines", "level", "drop_ticks"):
            game.set_word(name, 54321)
        game.set_byte("game_over", 1)
        game.call("new_game")
        self.assertEqual(game.read("board", 200), bytes(200))
        self.assertEqual([game.word(name) for name in ("score", "lines", "level", "drop_ticks")],
                         [0, 0, 1, 10])
        self.assertEqual(game.byte("game_over"), 0)
        self.assertIn(game.byte("piece_type"), range(7))
        self.assertIn(game.byte("next_type"), range(7))
        self.assertEqual(game.word("piece_x"), 3)
        self.assertEqual(game.word("piece_y"), 0)
        self.assertEqual(game.byte("piece_rot"), 0)
        game.assert_guards()

    def test_all_28_shapes_have_four_unique_connected_cells(self) -> None:
        for kind in range(7):
            for rotation in range(4):
                with self.subTest(kind=kind, rotation=rotation):
                    cells = self.game.cells(kind, rotation)
                    self.assertEqual(len(set(cells)), 4)
                    self.assertTrue(all(0 <= x < 4 and 0 <= y < 4 for x, y in cells))
                    reached = {cells[0]}
                    for _ in range(3):
                        reached |= {cell for cell in cells if any(
                            abs(cell[0] - x) + abs(cell[1] - y) == 1 for x, y in reached
                        )}
                    self.assertEqual(reached, set(cells))

    def test_every_rotation_at_all_four_boundaries(self) -> None:
        for kind in range(7):
            for rotation in range(4):
                cells = self.game.cells(kind, rotation)
                min_x, max_x = min(x for x, _ in cells), max(x for x, _ in cells)
                min_y, max_y = min(y for _, y in cells), max(y for _, y in cells)
                with self.subTest(kind=kind, rotation=rotation):
                    self.assertFalse(self.game.blocked(-min_x, -min_y, rotation, kind))
                    self.assertTrue(self.game.blocked(-min_x - 1, 4, rotation, kind))
                    self.assertFalse(self.game.blocked(9 - max_x, 4, rotation, kind))
                    self.assertTrue(self.game.blocked(10 - max_x, 4, rotation, kind))
                    self.assertFalse(self.game.blocked(3, 19 - max_y, rotation, kind))
                    self.assertTrue(self.game.blocked(3, 20 - max_y, rotation, kind))
                    self.assertTrue(self.game.blocked(3, -min_y - 1, rotation, kind))

    def test_occupied_cell_collision_and_register_preservation(self) -> None:
        game = self.game
        cells = game.cells(2, 0)
        board = bytearray(200)
        x, y = cells[0]
        board[(4 + y) * 10 + 3 + x] = 7
        game.write("board", bytes(board))
        registers = dict(ax=3, bx=4, cx=0x1200, dx=0x3402,
                         si=0x1234, di=0x5678, bp=0x4321)
        result = game.call("check_position", **registers)
        self.assertEqual(result["carry"], 1)
        self.assertEqual({key: result[key] for key in registers}, registers)
        self.assertFalse(game.blocked(3, 8, 0, 2))
        self.assertEqual(game.read("board", 200), bytes(board))

    def test_horizontal_movement_stops_at_both_walls(self) -> None:
        game = self.game
        game.piece(1, x=-1, y=3)
        game.call("move_left")
        self.assertEqual(game.word("piece_x", signed=True), -1)
        game.call("move_right")
        self.assertEqual(game.word("piece_x"), 0)
        game.set_word("piece_x", 7)
        game.call("move_right")
        self.assertEqual(game.word("piece_x"), 7)
        game.assert_guards()

    def test_rotation_uses_two_cell_wall_kick(self) -> None:
        game = self.game
        game.piece(0, rotation=1, x=-2, y=4)
        self.assertFalse(game.blocked(-2, 4, 1, 0))
        game.call("rotate_piece")
        self.assertEqual(game.byte("piece_rot"), 2)
        self.assertEqual(game.word("piece_x", signed=True), 0)
        self.assertEqual(game.word("piece_y"), 4)

    def test_rotation_cannot_pass_through_occupied_cells(self) -> None:
        game = self.game
        cells = game.cells(2, 0)
        board = bytearray([7] * 200)
        for x, y in cells:
            board[(4 + y) * 10 + 3 + x] = 0
        game.write("board", bytes(board))
        game.piece(2, 0, 3, 4)
        game.call("rotate_piece")
        self.assertEqual(game.byte("piece_rot"), 0)
        self.assertEqual(game.word("piece_x"), 3)
        self.assertEqual(game.read("board", 200), bytes(board))

    def test_ghost_is_lowest_legal_position_without_mutation(self) -> None:
        game = self.game
        board = bytearray(200)
        board[17 * 10 + 4] = 6
        game.write("board", bytes(board))
        for kind in range(7):
            for rotation in range(4):
                with self.subTest(kind=kind, rotation=rotation):
                    game.piece(kind, rotation, 3, 0)
                    game.call("calc_ghost")
                    ghost_y = game.word("ghost_y", signed=True)
                    self.assertFalse(game.blocked(3, ghost_y, rotation, kind))
                    self.assertTrue(game.blocked(3, ghost_y + 1, rotation, kind))
                    self.assertEqual(game.word("piece_y"), 0)
                    self.assertEqual(game.read("board", 200), bytes(board))
        game.assert_guards()

    def test_soft_drop_bonus_and_gravity(self) -> None:
        game = self.game
        game.piece(1, x=3, y=0)
        game.call("soft_drop")
        self.assertEqual(game.word("piece_y"), 1)
        self.assertEqual(game.word("score"), 1)
        game.call("step_down")
        self.assertEqual(game.word("piece_y"), 2)
        self.assertEqual(game.word("score"), 1)
        self.assertEqual(game.read("board", 200), bytes(200))

    def test_hard_drop_locks_four_cells_and_spawns_preview(self) -> None:
        game = self.game
        game.piece(1, x=3, y=0)
        preview = game.byte("next_type")
        game.call("hard_drop")
        board = game.read("board", 200)
        self.assertEqual(sum(value != 0 for value in board), 4)
        self.assertEqual([board[y * 10 + x] for y in (18, 19) for x in (4, 5)], [2] * 4)
        self.assertEqual(game.word("score"), 36)
        self.assertEqual(game.byte("piece_type"), preview)
        self.assertEqual(game.word("piece_y"), 0)
        game.assert_guards()

    def set_full_rows(self, count: int) -> bytes:
        board = bytearray(200)
        board[(19 - count) * 10] = 2
        board[(19 - count) * 10 + 9] = 3
        for row in range(20 - count, 20):
            board[row * 10:(row + 1) * 10] = bytes([row % 7 + 1] * 10)
        self.game.write("board", bytes(board))
        return bytes(board)

    def test_clear_one_to_four_lines_and_compact_board(self) -> None:
        for count, score in enumerate((100, 300, 500, 800), start=1):
            with self.subTest(count=count):
                self.game.call("new_game")
                self.set_full_rows(count)
                self.game.call("clear_lines")
                expected = bytearray(200)
                expected[190], expected[199] = 2, 3
                self.assertEqual(self.game.read("board", 200), bytes(expected))
                self.assertEqual(self.game.word("score"), score)
                self.assertEqual(self.game.word("lines"), count)
                self.assertEqual(self.game.word("level"), 1)
                self.game.assert_guards()

    def test_hard_drop_clears_four_rows_and_combines_scoring(self) -> None:
        game = self.game
        board = bytearray(200)
        for row in range(16, 20):
            board[row * 10:(row + 1) * 10] = bytes([3] * 10)
            board[row * 10 + 5] = 0
        game.write("board", bytes(board))
        game.piece(0, rotation=1, x=3, y=0)
        game.call("hard_drop")
        self.assertEqual(game.read("board", 200), bytes(200))
        self.assertEqual(game.word("lines"), 4)
        self.assertEqual(game.word("score"), 800 + 2 * 16)
        self.assertEqual(game.byte("game_over"), 0)
        game.assert_guards()

    def test_level_transition_uses_preclear_multiplier(self) -> None:
        game = self.game
        game.set_word("lines", 9)
        self.set_full_rows(1)
        game.call("clear_lines")
        self.assertEqual(game.word("score"), 100)
        self.assertEqual(game.word("lines"), 10)
        self.assertEqual(game.word("level"), 2)
        self.assertEqual(game.word("drop_ticks"), 9)
        self.set_full_rows(2)
        game.call("clear_lines")
        self.assertEqual(game.word("score"), 100 + 300 * 2)

    def test_score_lines_and_speed_saturate(self) -> None:
        game = self.game
        game.set_word("score", 65000)
        game.set_word("lines", 65534)
        game.set_word("level", 20)
        self.set_full_rows(4)
        game.call("clear_lines")
        self.assertEqual(game.word("score"), 65535)
        self.assertEqual(game.word("lines"), 65535)
        self.assertEqual(game.word("level"), 20)
        self.assertEqual(game.word("drop_ticks"), 2)
        game.set_word("score", 65535)
        game.piece(1, x=3, y=0)
        game.call("soft_drop")
        self.assertEqual(game.word("score"), 65535)
        game.call("hard_drop")
        self.assertEqual(game.word("score"), 65535)
        game.assert_guards()

    def test_spawn_collision_sets_game_over(self) -> None:
        game = self.game
        board = bytearray(200)
        for row in (0, 1):
            for column in range(2, 8):
                board[row * 10 + column] = 7
        game.write("board", bytes(board))
        game.piece(1, x=3, y=18)
        game.call("hard_drop")
        self.assertEqual(game.byte("game_over"), 1)
        self.assertEqual(game.word("lines"), 0)
        game.assert_guards()

    def test_randomized_gameplay_is_deterministic_and_stays_in_bounds(self) -> None:
        second = Engine(self.binary, self.symbols)
        chooser = random.Random(0x8086)
        actions = ("move_left", "move_right", "rotate_piece", "soft_drop",
                   "step_down", "hard_drop", "calc_ghost")
        compared_words = ("piece_x", "piece_y", "score", "lines", "level",
                          "drop_ticks", "rng_seed", "ghost_y")
        compared_bytes = ("piece_type", "piece_rot", "next_type", "game_over")
        restarts = 0
        for index in range(2500):
            if self.game.byte("game_over"):
                self.game.call("new_game")
                second.call("new_game")
                restarts += 1
            action = chooser.choice(actions)
            self.game.call(action)
            second.call(action)
            with self.subTest(index=index, action=action):
                board = self.game.read("board", 200)
                self.assertEqual(board, second.read("board", 200))
                self.assertTrue(all(0 <= value <= 7 for value in board))
                for name in compared_words:
                    self.assertEqual(self.game.word(name), second.word(name))
                for name in compared_bytes:
                    self.assertEqual(self.game.byte(name), second.byte(name))
                self.assertIn(self.game.byte("piece_type"), range(7))
                self.assertIn(self.game.byte("piece_rot"), range(4))
                if not self.game.byte("game_over"):
                    self.assertFalse(self.game.blocked(
                        self.game.word("piece_x", signed=True),
                        self.game.word("piece_y", signed=True),
                        self.game.byte("piece_rot"), self.game.byte("piece_type"),
                    ))
                self.game.assert_guards()
                second.assert_guards()
        self.assertGreater(restarts, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
