#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

#!/usr/bin/python3
# coding=utf-8
import numpy as np
import os

TOTAL_SIZE = 16384  # 8 * 2048

def gen_golden_data_simple():
    # Создаем папки, если их нет
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    # Генерируем вектор
    input_data = np.random.uniform(-5, 5, [TOTAL_SIZE]).astype(np.float32)
    
    # Вычисляем softmax по блокам (как делает ядро)
    golden = np.zeros_like(input_data)
    for i in range(0, TOTAL_SIZE, 2048):
        block = input_data[i:i+2048]
        # Стабильный softmax для python (вычитаем max)
        max_val = np.max(block)
        exp_block = np.exp(block - max_val)
        sum_exp = np.sum(exp_block)
        golden[i:i+2048] = exp_block / sum_exp
    
    # Сохраняем файлы
    input_data.tofile("./input/softmax_input.bin")
    golden.tofile("./output/golden.bin")
    print("Generated input/softmax_input.bin and output/golden.bin")

if __name__ == '__main__':
    gen_golden_data_simple()