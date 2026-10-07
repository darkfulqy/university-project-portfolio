module encoder16_4 (
    input  [15:0] Dn,     // 低有效输入 D15 优先级最高
    output reg [3:0] An,   // 低有效输出
    output reg GSn         // 低有效：是否有有效输入
);

always @(*) begin
    GSn = 1'b1;
    An  = 4'b1111;

    if (Dn[15] == 0) begin An = 4'b0000; GSn = 0; end
    else if (Dn[14] == 0) begin An = 4'b0001; GSn = 0; end
    else if (Dn[13] == 0) begin An = 4'b0010; GSn = 0; end
    else if (Dn[12] == 0) begin An = 4'b0011; GSn = 0; end
    else if (Dn[11] == 0) begin An = 4'b0100; GSn = 0; end
    else if (Dn[10] == 0) begin An = 4'b0101; GSn = 0; end
    else if (Dn[9]  == 0) begin An = 4'b0110; GSn = 0; end
    else if (Dn[8]  == 0) begin An = 4'b0111; GSn = 0; end
    else if (Dn[7]  == 0) begin An = 4'b1000; GSn = 0; end
    else if (Dn[6]  == 0) begin An = 4'b1001; GSn = 0; end
    else if (Dn[5]  == 0) begin An = 4'b1010; GSn = 0; end
    else if (Dn[4]  == 0) begin An = 4'b1011; GSn = 0; end
    else if (Dn[3]  == 0) begin An = 4'b1100; GSn = 0; end
    else if (Dn[2]  == 0) begin An = 4'b1101; GSn = 0; end
    else if (Dn[1]  == 0) begin An = 4'b1110; GSn = 0; end
    else if (Dn[0]  == 0) begin An = 4'b1111; GSn = 0; end
end

endmodule
