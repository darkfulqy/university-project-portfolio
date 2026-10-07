; 俄罗斯方块 — IBM-PC / 8086 / DOS
; 汇编：nasm -f bin -Werror main.asm -o build/TETRIS.COM
; 游戏及绘图均由 16 位汇编实现。CPU 指令限制为原始 8086。
bits 16
cpu 8086
org 100h

BOARD_COL equ 30
BOARD_ROW equ 3

%macro TEXT 4
    mov dh, %1
    mov dl, %2
    mov bl, %3
    mov si, %4
    call put_text
%endmacro

jmp start

start:
    ; COM 的代码、数据及栈共用一个段，栈位于显式预留的区域。
    cli
    mov ax, cs
    mov ss, ax
    mov sp, stack_top
    sti
    mov ds, ax
    mov es, ax
    cld
    ; 保存进入前的显示模式、活动页及光标形状，退出时恢复。
    mov ah, 0Fh
    int 10h
    mov [old_mode], al
    mov [old_page], bh
    mov ah, 03h
    int 10h
    mov [old_cursor], cx
    mov [old_cursor_pos], dx
    mov ax, 0003h
    int 10h
    ; 关闭字符闪烁，允许属性字节使用全部背景色位（VGA BIOS）。
    mov ax, 1003h
    xor bx, bx
    int 10h
    mov ah, 01h
    mov cx, 2000h
    int 10h
    xor ah, ah
    int 1Ah
    xor dx, cx
    or dx, 1
    mov [rng_seed], dx
    call new_game
    call reset_clock
    mov byte [title_screen], 1
    mov byte [dirty], 1

; 非阻塞事件循环：每次最多处理一个按键，再检查计时器，避免长按饿死下落。
main_loop:
    mov ah, 01h
    int 16h
    jz .timer
    xor ah, ah
    int 16h
    call handle_key
.timer:
    cmp byte [title_screen], 0
    jne .render
    cmp byte [paused], 0
    jne .render
    cmp byte [game_over], 0
    jne .render
    xor ah, ah
    int 1Ah                       ; CX:DX = 自午夜起的 BIOS 时钟滴答
    ; 午夜归零时重设基准，避免无符号差误判。低字回绕通过 SUB 处理。
    or al, al
    jnz .clock_reset
    mov ax, dx
    sub ax, [last_tick]
    cmp ax, [drop_ticks]
    jb .render
    mov [last_tick], dx
    call step_down
    mov byte [dirty], 1
    jmp .render
.clock_reset:
    mov [last_tick], dx
.render:
    cmp byte [dirty], 0
    je .idle
    call render
    mov byte [dirty], 0
.idle:
    ; 有 BIOS 时钟/键盘中断才唤醒，避免无意义的忙等。
    sti
    hlt
    jmp main_loop

; 输入 AX 来自 INT 16h：AL=ASCII，AH=扫描码。允许破坏通用寄存器。
handle_key:
    cmp al, 27
    je exit_game
    cmp al, 'q'
    je exit_game
    cmp al, 'Q'
    je exit_game
    cmp al, 'r'
    je restart_game
    cmp al, 'R'
    je restart_game
    cmp byte [title_screen], 0
    je .not_title
    cmp al, 13
    je begin_game
    cmp al, ' '
    je begin_game
    ret
.not_title:
    cmp byte [game_over], 0
    je .not_over
    cmp al, 13
    je restart_game
    ret
.not_over:
    cmp al, 'p'
    je toggle_pause
    cmp al, 'P'
    je toggle_pause
    cmp byte [paused], 0
    jne .done
    cmp ah, 4Bh
    je .left
    cmp ah, 4Dh
    je .right
    cmp ah, 48h
    je .rotate
    cmp ah, 50h
    je .down
    cmp al, ' '
    je .drop
    or al, 20h                    ; A-Z 转小写；扫描码已优先检查
    cmp al, 'a'
    je .left
    cmp al, 'd'
    je .right
    cmp al, 'w'
    je .rotate
    cmp al, 's'
    je .down
.done:
    ret
.left:
    call move_left
    jmp .changed
.right:
    call move_right
    jmp .changed
.rotate:
    call rotate_piece
    jmp .changed
.down:
    call soft_drop
    call reset_clock
    jmp .changed
.drop:
    call hard_drop
    call reset_clock
.changed:
    mov byte [dirty], 1
    ret

begin_game:
    mov byte [title_screen], 0
    call reset_clock
    mov byte [dirty], 1
    ret

restart_game:
    call new_game
    mov byte [paused], 0
    mov byte [title_screen], 0
    call reset_clock
    mov byte [dirty], 1
    ret

toggle_pause:
    xor byte [paused], 1
    call reset_clock
    mov byte [dirty], 1
    ret

reset_clock:
    xor ah, ah
    int 1Ah
    mov [last_tick], dx
    ret

exit_game:
    xor ah, ah
    mov al, [old_mode]
    int 10h
    mov ah, 05h
    mov al, [old_page]
    int 10h
    mov ah, 01h
    mov cx, [old_cursor]
    int 10h
    mov ah, 02h
    mov bh, [old_page]
    mov dx, [old_cursor_pos]
    int 10h
    mov ax, 4C00h
    int 21h

; --------------------- 文本显存绘图 ---------------------
; 一个文本单元 = 字符字节 + 颜色属性字节。
; 字符地址 B800:(行 * 80 + 列) * 2。用两个字符组成一个近似正方形方格。
; 先在本段 screen_buffer 绘图，再一次复制到 B800h，减少中间画面闪烁。
render:
    push cs
    pop es
    mov di, screen_buffer
    mov ax, 0720h
    mov cx, 2000
    rep stosw

    TEXT 1, 4, 0Bh, str_title
    TEXT 2, 4, 08h, str_subtitle
    TEXT 4, 4, 07h, str_tag
    TEXT 6, 4, 08h, str_score
    mov ax, [score]
    mov dx, 0704h
    mov bl, 0Fh
    call put_number
    TEXT 10, 4, 08h, str_lines
    mov ax, [lines]
    mov dx, 0B04h
    mov bl, 0Fh
    call put_number
    TEXT 14, 4, 08h, str_level
    mov ax, [level]
    mov dx, 0F04h
    mov bl, 0Eh
    call put_number
    TEXT 19, 4, 08h, str_hardware1
    TEXT 20, 4, 08h, str_hardware2
    TEXT 21, 4, 08h, str_hardware3
    TEXT 1, 34, 07h, str_playfield
    TEXT 4, 56, 08h, str_next
    TEXT 12, 56, 0Bh, str_controls
    TEXT 14, 56, 07h, str_move
    TEXT 15, 56, 07h, str_rotate
    TEXT 16, 56, 07h, str_down
    TEXT 17, 56, 07h, str_drop
    TEXT 19, 56, 08h, str_pause
    TEXT 20, 56, 08h, str_restart
    TEXT 21, 56, 08h, str_exit
    TEXT 24, 4, 08h, str_footer

    ; 边框：20 格高，每格两个字符宽。
    mov dx, 021Dh                ; 行2，列29
    mov ax, 08DAh                ; CP437 左上角
    call put_char
    mov dx, 0232h
    mov ax, 08BFh
    call put_char
    mov dx, 171Dh
    mov ax, 08C0h
    call put_char
    mov dx, 1732h
    mov ax, 08D9h
    call put_char
    mov dl, BOARD_COL
    mov cx, 20
.horizontal:
    mov dh, 2
    mov ax, 08C4h
    call put_char
    mov dh, 23
    call put_char
    inc dl
    loop .horizontal
    mov dh, BOARD_ROW
    mov cx, 20
.vertical:
    mov dl, 29
    mov ax, 08B3h
    call put_char
    mov dl, 50
    call put_char
    inc dh
    loop .vertical

    ; 固定的方块来自 board；活动块和落点投影不写进棋盘。
    mov si, board
    xor dx, dx
.row:
    xor cx, cx
.col:
    xor bx, bx
    mov bl, [si]
    mov ax, 08FAh                ; 空格中的淡灰色网格点
    or bl, bl
    jz .cell
    dec bx
    mov ah, [shape_colors + bx]
    mov al, 219
.cell:
    call draw_cell
    inc si
    inc cx
    cmp cx, 10
    jb .col
    inc dx
    cmp dx, 20
    jb .row

    cmp byte [game_over], 0
    jne .next
    call calc_ghost
    call get_shape
    mov ax, 08B0h                ; 落点投影：灰色浅阴影
    mov bx, [ghost_y]
    call draw_active
    call get_shape
    xor bx, bx
    mov bl, [piece_type]
    mov ah, [shape_colors + bx]
    mov al, 219
    mov bx, [piece_y]
    call draw_active
.next:
    ; 预览永远取下一个方块的初始朝向。
    xor bx, bx
    mov bl, [next_type]
    mov ah, [shape_colors + bx]
    mov al, 219
    mov cl, 5
    shl bx, cl                   ; 每种形状 32 字节
    lea si, [shapes + bx]
    mov cx, 4
.preview:
    mov dl, [si]
    shl dl, 1
    add dl, 58
    mov dh, [si+1]
    add dh, 6
    call put_char
    inc dl
    call put_char
    add si, 2
    loop .preview

    cmp byte [title_screen], 0
    jne .title
    cmp byte [game_over], 0
    jne .over
    cmp byte [paused], 0
    jne .paused
    TEXT 23, 56, 0Ah, str_playing
    jmp .copy
.title:
    call overlay_panel
    TEXT 10, 34, 1Fh, str_ready
    TEXT 12, 32, 1Eh, str_start
    TEXT 14, 32, 17h, str_start2
    jmp .copy
.over:
    call overlay_panel
    TEXT 10, 35, 1Ch, str_gameover
    TEXT 12, 32, 1Fh, str_again
    TEXT 14, 34, 17h, str_quit
    jmp .copy
.paused:
    call overlay_panel
    TEXT 10, 37, 1Eh, str_paused
    TEXT 12, 33, 1Fh, str_resume
    TEXT 14, 34, 17h, str_quit
.copy:
    mov ax, 0B800h
    mov es, ax
    xor di, di
    mov si, screen_buffer
    mov cx, 2000
    rep movsw
    push ds
    pop es
    ret

; put_char: DH=屏幕行，DL=列，AX=属性:字符；保持所有通用寄存器。
put_char:
    push ax
    push bx
    push dx
    push di
    mov bx, ax
    xor ax, ax
    mov al, dh
    mov di, 160
    push dx
    mul di
    pop dx
    mov di, ax
    xor dh, dh
    shl dx, 1
    add di, dx
    mov [screen_buffer + di], bx
    pop di
    pop dx
    pop bx
    pop ax
    ret

; put_text: DS:SI=零结尾字符串，DH/DL=行列，BL=颜色；保持寄存器。
put_text:
    push ax
    push dx
    push si
    mov ah, bl
.next:
    lodsb
    or al, al
    jz .done
    call put_char
    inc dl
    jmp .next
.done:
    pop si
    pop dx
    pop ax
    ret

; put_number: AX=0..65535；固定五位十进制，DIV 10 从右向左生成。
put_number:
    push ax
    push bx
    push cx
    push dx
    push si
    push di
    push dx
    push bx
    mov di, number_buffer + 4
    mov bx, 10
    mov cx, 5
.digit:
    xor dx, dx
    div bx
    add dl, '0'
    mov [di], dl
    dec di
    loop .digit
    pop bx
    pop dx
    mov si, number_buffer
    call put_text
    pop di
    pop si
    pop dx
    pop cx
    pop bx
    pop ax
    ret

; draw_cell: CX=棋盘x，DX=棋盘y，AX=属性:字符；保持所有寄存器。
draw_cell:
    push ax
    push bx
    push dx
    cmp cx, 10
    jae .done
    cmp dx, 20
    jae .done
    mov bx, cx
    shl bx, 1
    add bl, BOARD_COL
    mov dh, dl
    add dh, BOARD_ROW
    mov dl, bl
    call put_char
    inc dl
    cmp al, 250
    jne .second
    mov al, ' '
.second:
    call put_char
.done:
    pop dx
    pop bx
    pop ax
    ret

; draw_active: SI=四组局部坐标，BX=块原点y，AX=属性:字符。
draw_active:
    push ax
    push bx
    push cx
    push dx
    push si
    push di
    mov di, 4
.cell:
    xor cx, cx
    mov cl, [si]
    add cx, [piece_x]
    xor dx, dx
    mov dl, [si+1]
    add dx, bx
    call draw_cell
    add si, 2
    dec di
    jnz .cell
    pop di
    pop si
    pop dx
    pop cx
    pop bx
    pop ax
    ret

overlay_panel:
    mov dh, 9
.row:
    mov dl, BOARD_COL
.col:
    mov ax, 1720h
    call put_char
    inc dl
    cmp dl, 50
    jb .col
    inc dh
    cmp dh, 16
    jb .row
    ret

old_mode db 3
old_page db 0
old_cursor dw 0607h
old_cursor_pos dw 0
last_tick dw 0
paused db 0
title_screen db 1
dirty db 1
number_buffer db '00000', 0
str_title db 'T E T R I S', 0
str_subtitle db '8086 / IBM-PC', 0
str_tag db 'ASSEMBLY EDITION', 0
str_score db 'S C O R E', 0
str_lines db 'L I N E S', 0
str_level db 'L E V E L', 0
str_hardware1 db '16-BIT REAL MODE', 0
str_hardware2 db 'VIDEO  B800:0000', 0
str_hardware3 db '7-BAG RANDOMIZER', 0
str_playfield db 'PLAYFIELD', 0
str_next db 'N E X T', 0
str_controls db 'CONTROLS', 0
str_move db 'A / D    Move', 0
str_rotate db 'W / UP   Rotate', 0
str_down db 'S / DOWN Soft drop', 0
str_drop db 'SPACE    Hard drop', 0
str_pause db 'P        Pause', 0
str_restart db 'R        Restart', 0
str_exit db 'ESC / Q  Exit', 0
str_footer db 'ARROW KEYS ALSO WORK      SHADED CELLS = LANDING POSITION', 0
str_playing db 'PLAYING', 0
str_ready db 'READY TO STACK', 0
str_start db 'ENTER / SPACE', 0
str_start2 db '     TO START', 0
str_gameover db 'GAME OVER', 0
str_again db 'ENTER / R: RETRY', 0
str_quit db 'ESC: EXIT', 0
str_paused db 'PAUSED', 0
str_resume db 'P: RESUME', 0

%include "game.inc"

align 2, db 0
screen_buffer times 4000 db 0
stack_bottom:
    times 1024 db 0
stack_top:

; 保留 PSP、代码、数据、缓冲区和栈后，仍必须装进一个 64 KiB 段。
%if ($-$$) > 0FE00h
    %error "COM program and stack exceed available segment space"
%endif
