; ============================================================================
; 俄罗斯方块规则核心 —— NASM，纯 8086 指令，16 位实模式。
; 由 main.asm 在同一个 COM 程序内包含；本文件不执行 BIOS/DOS 调用。
;
; 约定：DS 指向程序段。所有公共过程保留 DS、ES；其余寄存器可被修改。
; check_position 额外保留全部通用寄存器，只通过 CF 返回碰撞结果。
; 棋盘为 10 列 × 20 行，按行存放：0 是空格，1..7 是方块类型 + 1。
; 当前活动块没有写入 board；落地后才通过 lock_piece 写入。
; ============================================================================

; 两侧哨兵供自动测试检查越界；正常游戏永远不会改动这些字节。
board_guard_before: times 16 db 0a5h
board:              times 200 db 0
board_guard_after:  times 16 db 0a5h

piece_type: db 0                  ; I、O、T、S、Z、J、L，编号 0..6
piece_rot:  db 0                  ; 顺时针方向编号 0..3
piece_x:    dw 3                  ; 局部 4×4 方框左上角，可为负数
piece_y:    dw 0
next_type:  db 0
score:      dw 0                  ; 无符号，超过 65535 时保持 65535
lines:      dw 0
level:      dw 1                  ; 每消除 10 行升一级，上限 20
drop_ticks: dw 10                 ; 每格间隔 BIOS tick；下限 2
game_over:  db 0
rng_seed:   dw 0ace1h             ; 主程序可在启动时用 BIOS 时钟初始化
ghost_y:    dw 0

; 文字模式显示属性使用的颜色；显示代码决定放入前景还是背景半字节。
shape_colors: db 11, 14, 13, 10, 12, 9, 6

; 每种方块 4 个方向，每方向 4 对 (x,y)，每对 2 字节。
; 固定大小使地址 = shapes + type*32 + rot*8，8086 用逐次移位实现。
shapes:
    ; I
    db 0,1, 1,1, 2,1, 3,1
    db 2,0, 2,1, 2,2, 2,3
    db 0,2, 1,2, 2,2, 3,2
    db 1,0, 1,1, 1,2, 1,3
    ; O：旋转不改变位置
    db 1,0, 2,0, 1,1, 2,1
    db 1,0, 2,0, 1,1, 2,1
    db 1,0, 2,0, 1,1, 2,1
    db 1,0, 2,0, 1,1, 2,1
    ; T
    db 1,0, 0,1, 1,1, 2,1
    db 1,0, 1,1, 2,1, 1,2
    db 0,1, 1,1, 2,1, 1,2
    db 1,0, 0,1, 1,1, 1,2
    ; S
    db 1,0, 2,0, 0,1, 1,1
    db 1,0, 1,1, 2,1, 2,2
    db 1,1, 2,1, 0,2, 1,2
    db 0,0, 0,1, 1,1, 1,2
    ; Z
    db 0,0, 1,0, 1,1, 2,1
    db 2,0, 1,1, 2,1, 1,2
    db 0,1, 1,1, 1,2, 2,2
    db 1,0, 0,1, 1,1, 0,2
    ; J
    db 0,0, 0,1, 1,1, 2,1
    db 1,0, 2,0, 1,1, 1,2
    db 0,1, 1,1, 2,1, 2,2
    db 1,0, 1,1, 0,2, 1,2
    ; L
    db 2,0, 0,1, 1,1, 2,1
    db 1,0, 1,1, 1,2, 2,2
    db 0,1, 1,1, 2,1, 0,2
    db 0,0, 1,0, 1,1, 1,2

bag:           times 7 db 0
bag_index:     db 7
rotation_try:  db 0
rotation_kicks: dw 0, -1, 1, -2, 2
cleared_count: db 0
line_scores:   dw 0, 100, 300, 500, 800

; 开始一局。保留随机数状态，使重新开始时不会反复出现相同序列。
new_game:
    push es
    push ds
    pop es
    cld
    xor ax, ax
    mov di, board
    mov cx, 200
    rep stosb
    pop es
    mov [score], ax
    mov [lines], ax
    mov [game_over], al
    mov word [level], 1
    mov word [drop_ticks], 10
    mov byte [bag_index], 7
    call next_piece_type
    mov [next_type], al
    call spawn_piece
    ret

; 取当前活动块的 4 个局部坐标，返回 DS:SI。
get_shape:
    xor ax, ax
    mov al, [piece_type]
    mov bx, ax
    shl bx, 1
    shl bx, 1
    shl bx, 1
    shl bx, 1
    shl bx, 1
    mov al, [piece_rot]
    shl ax, 1
    shl ax, 1
    shl ax, 1
    add bx, ax
    mov si, shapes
    add si, bx
    ret

; 碰撞检测：AX=x，BX=y，CL=方向，DL=类型。
; 返回 CF=1 表示越界或与已有方块重叠，CF=0 表示可放置。
; 保留全部通用寄存器，允许调用者重复尝试不同候选位置。
; 无符号比较同时拒绝负的实际格坐标：例如 -1 = FFFFh >= 10。
check_position:
    push ax
    push bx
    push cx
    push dx
    push si
    push di
    push bp
    mov bp, ax                       ; 候选方框 x
    push bx                          ; 暂存候选方框 y
    xor dh, dh
    mov si, dx
    shl si, 1
    shl si, 1
    shl si, 1
    shl si, 1
    shl si, 1
    xor ch, ch
    mov di, cx
    shl di, 1
    shl di, 1
    shl di, 1
    add si, di
    add si, shapes
    pop dx                           ; DX = 候选方框 y
    mov cx, 4
.cell:
    xor ax, ax
    mov al, [si]
    add ax, bp
    cmp ax, 10
    jae .blocked
    mov di, ax                       ; DI = 这一格的绝对 x
    xor ax, ax
    mov al, [si+1]
    add ax, dx
    cmp ax, 20
    jae .blocked
    mov bx, ax
    shl bx, 1
    shl bx, 1
    shl bx, 1                        ; BX = y*8
    shl ax, 1                        ; AX = y*2
    add bx, ax
    add bx, di                       ; BX = y*10+x
    cmp byte [board+bx], 0
    jne .blocked
    add si, 2
    loop .cell
    clc
    jmp short .restore
.blocked:
    stc
.restore:
    pop bp
    pop di
    pop si
    pop dx
    pop cx
    pop bx
    pop ax
    ret

move_left:
    cmp byte [game_over], 0
    jne .done
    mov ax, [piece_x]
    dec ax
    mov bx, [piece_y]
    mov cl, [piece_rot]
    mov dl, [piece_type]
    call check_position
    jc .done
    mov [piece_x], ax
.done:
    ret

move_right:
    cmp byte [game_over], 0
    jne .done
    mov ax, [piece_x]
    inc ax
    mov bx, [piece_y]
    mov cl, [piece_rot]
    mov dl, [piece_type]
    call check_position
    jc .done
    mov [piece_x], ax
.done:
    ret

; 顺时针旋转，依次尝试水平偏移 0、-1、+1、-2、+2。
; 这是易解释的简化踢墙规则，不包含向上踢墙或完整 SRS 规则表。
rotate_piece:
    cmp byte [game_over], 0
    jne .done
    mov al, [piece_rot]
    inc al
    and al, 3
    mov [rotation_try], al
    mov si, rotation_kicks
    mov di, 5
.try:
    mov ax, [piece_x]
    add ax, [si]
    mov bx, [piece_y]
    mov cl, [rotation_try]
    mov dl, [piece_type]
    call check_position
    jnc .accept
    add si, 2
    dec di
    jnz .try
    ret
.accept:
    mov [piece_x], ax
    mov al, [rotation_try]
    mov [piece_rot], al
.done:
    ret

; 自动下落：成功则下降一格；无法下降则立即固定，无锁定延迟。
step_down:
    cmp byte [game_over], 0
    jne .done
    mov ax, [piece_x]
    mov bx, [piece_y]
    inc bx
    mov cl, [piece_rot]
    mov dl, [piece_type]
    call check_position
    jc .lock
    mov [piece_y], bx
.done:
    ret
.lock:
    jmp lock_piece

; 软降：每成功下降一格加 1 分，撞到底部时固定。
soft_drop:
    cmp byte [game_over], 0
    jne .done
    mov ax, [piece_x]
    mov bx, [piece_y]
    inc bx
    mov cl, [piece_rot]
    mov dl, [piece_type]
    call check_position
    jc .lock
    mov [piece_y], bx
    mov ax, 1
    call add_score
.done:
    ret
.lock:
    jmp lock_piece

; 硬降：先计算落点，按下降距离 ×2 加分，然后立即固定。
hard_drop:
    cmp byte [game_over], 0
    jne .done
    call calc_ghost
    mov ax, [ghost_y]
    sub ax, [piece_y]
    shl ax, 1
    call add_score
    mov ax, [ghost_y]
    mov [piece_y], ax
    call lock_piece
.done:
    ret

; 落点预览。只修改 ghost_y 和通用寄存器，不改变棋盘及活动块。
calc_ghost:
    mov bx, [piece_y]
    cmp byte [game_over], 0
    jne .save
    mov ax, [piece_x]
    mov cl, [piece_rot]
    mov dl, [piece_type]
.descend:
    inc bx
    call check_position
    jnc .descend
    dec bx
.save:
    mov [ghost_y], bx
    ret

; 固定活动块，消行，再生成下一块。额外检查当前位置以避免越界写。
lock_piece:
    cmp byte [game_over], 0
    jne .done
    mov ax, [piece_x]
    mov bx, [piece_y]
    mov cl, [piece_rot]
    mov dl, [piece_type]
    call check_position
    jc .invalid
    call get_shape
    mov cx, 4
.cell:
    xor ax, ax
    mov al, [si+1]
    add ax, [piece_y]
    mov bx, ax
    shl bx, 1
    shl bx, 1
    shl bx, 1
    shl ax, 1
    add bx, ax
    xor ax, ax
    mov al, [si]
    add ax, [piece_x]
    add bx, ax
    mov al, [piece_type]
    inc al
    mov [board+bx], al
    add si, 2
    loop .cell
    call clear_lines
    call spawn_piece
.done:
    ret
.invalid:
    mov byte [game_over], 1
    ret

; 自底向上检查满行。删除第 r 行时，把前 r 行逆向复制下一行，
; 然后清空顶行，并再次检查原来的 r 行，防止漏掉连续的满行。
; r=0 时复制长度为 0，仍然会清空顶行，因此顶行也能正确消除。
; 本过程可单独调用；它更新分数/等级，但不生成活动块。
clear_lines:
    push es
    push ds
    pop es
    cld
    mov byte [cleared_count], 0
    mov bp, 19
.row:
    mov ax, bp
    mov bx, ax
    shl bx, 1
    shl bx, 1
    shl bx, 1
    shl ax, 1
    add bx, ax                       ; BX = r*10
    mov si, board
    add si, bx
    mov cx, 10
.inspect:
    cmp byte [si], 0
    je .next_row
    inc si
    loop .inspect
    inc byte [cleared_count]
    mov cx, bx
    mov si, board-1
    add si, bx
    mov di, board+9
    add di, bx
    std                              ; 重叠复制：从高地址向低地址读取
    rep movsb
    cld                              ; 过程出口及显示代码均使用正方向
    xor ax, ax
    mov di, board
    mov cx, 10
    rep stosb
    jmp short .row                   ; 新移下来的这一行还需要检查
.next_row:
    dec bp
    jns .row
    pop es
    cmp byte [cleared_count], 0
    je .done

    ; 得分使用消行前的等级。正常游戏一次最多消除四行。
    xor bx, bx
    mov bl, [cleared_count]
    cmp bx, 4
    jbe .score_index
    mov bx, 4                        ; 防御性处理人工构造的 >4 行棋盘
.score_index:
    shl bx, 1
    mov ax, [line_scores+bx]
    mul word [level]
    or dx, dx
    jnz .max_score
    call add_score
    jmp short .add_lines
.max_score:
    mov word [score], 0ffffh
.add_lines:
    xor ax, ax
    mov al, [cleared_count]
    add ax, [lines]
    jnc .store_lines
    mov ax, 0ffffh
.store_lines:
    mov [lines], ax
    xor dx, dx
    mov bx, 10
    div bx                           ; AX = 总消行数 /10
    inc ax
    cmp ax, 20
    jbe .store_level
    mov ax, 20
.store_level:
    mov [level], ax
    mov bx, 11
    sub bx, ax
    cmp ax, 9
    jb .store_ticks
    mov bx, 2
.store_ticks:
    mov [drop_ticks], bx
.done:
    ret

; 分数饱和加法：输入 AX=本次加分，进位意味着超过 16 位上限。
add_score:
    add ax, [score]
    jnc .store
    mov ax, 0ffffh
.store:
    mov [score], ax
    ret

; 将预告块变为当前块，再从七袋中抽取新的预告块。
; 出生时与已有格重叠即结束，避免方块覆盖堆积区域。
spawn_piece:
    mov al, [next_type]
    mov [piece_type], al
    mov byte [piece_rot], 0
    mov word [piece_x], 3
    mov word [piece_y], 0
    call next_piece_type
    mov [next_type], al
    mov ax, [piece_x]
    mov bx, [piece_y]
    mov cl, [piece_rot]
    mov dl, [piece_type]
    call check_position
    jnc .valid
    mov byte [game_over], 1
.valid:
    call calc_ghost
    ret

; 七袋随机：每袋包含七种方块各一个，Fisher-Yates 原地洗牌。
; 返回 AL=下一个类型。每 7 次抽取构成一个七种类型的排列。
next_piece_type:
    push bx
    push cx
    push dx
    push si
    push di
    cmp byte [bag_index], 7
    jb .take
    xor si, si
.fill:
    mov ax, si
    mov [bag+si], al
    inc si
    cmp si, 7
    jb .fill
    mov si, 6
.shuffle:
    call random_word
    xor dx, dx
    mov bx, si
    inc bx
    div bx                           ; 余数 DX 在 [0,i] 范围
    mov di, dx
    mov al, [bag+si]
    mov ah, [bag+di]
    mov [bag+si], ah
    mov [bag+di], al
    dec si
    jnz .shuffle
    mov byte [bag_index], 0
.take:
    xor bx, bx
    mov bl, [bag_index]
    mov al, [bag+bx]
    inc byte [bag_index]
    pop di
    pop si
    pop dx
    pop cx
    pop bx
    ret

; 16 位线性同余伪随机数：seed = seed*25173 + 13849 (mod 65536)。
; 乘法只保存低 16 位，溢出即模运算；返回 AX，保留 BX 和 DX。
random_word:
    push bx
    push dx
    mov ax, [rng_seed]
    mov bx, 25173
    mul bx
    add ax, 13849
    mov [rng_seed], ax
    pop dx
    pop bx
    ret
