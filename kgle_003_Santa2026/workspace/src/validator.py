"""
Elf Calculator Zero-Trust Validator
48個の記憶レジスタと四則演算ルールに基づく厳格なハードウェア検証器
"""

import json
from typing import List, Tuple, Dict, Any, Set, Optional


class ElfCalculatorValidator:
    def __init__(self, puzzles_path: str):
        self.puzzles: Dict[str, Dict[str, Any]] = {}
        self._load_puzzles(puzzles_path)

    def _load_puzzles(self, path: str):
        import pandas as pd
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            pid = str(row['id']).strip()
            if pid.isdigit():
                pid = f"scored_{int(pid):04d}"
            
            starts = json.loads(row['starting_values']) if isinstance(row['starting_values'], str) else list(row['starting_values'])
            targets = json.loads(row['targets']) if isinstance(row['targets'], str) else list(row['targets'])
            
            self.puzzles[pid] = {
                'starting_values': starts,
                'targets': targets,
                'target_set': set(targets),
                'value_limit': int(row['value_limit']),
                'registers': int(row.get('registers', 48))
            }

    def validate_puzzle(self, pid: str, program: List[List[Any]]) -> Tuple[bool, str, int]:
        """
        指定したパズルに対するプログラムを検証する。
        戻り値: (is_valid, error_message, step_count)
        """
        pinfo = self.puzzles.get(pid)
        if pinfo is None:
            return False, f"Unknown puzzle ID {pid}", 0
        
        if not isinstance(program, list):
            return False, "Program must be a list of instructions", 0

        step_count = len(program)
        if step_count > 4096:
            return False, f"Instruction count {step_count} exceeds limit of 4096", step_count

        regs_cap = pinfo['registers']
        limit = pinfo['value_limit']
        
        # レジスタ配列の初期化 (None: 未定義)
        regs: List[Optional[int]] = [None] * regs_cap
        for idx, val in enumerate(pinfo['starting_values']):
            if idx < regs_cap:
                regs[idx] = val

        remaining = set(pinfo['target_set']) - set(pinfo['starting_values'])

        for step, ins in enumerate(program):
            if not isinstance(ins, (list, tuple)) or len(ins) != 4:
                return False, f"Malformed instruction at step {step}: {ins}", step_count
            
            op, ra, rb, rd = ins
            if not (isinstance(ra, int) and isinstance(rb, int) and isinstance(rd, int)):
                return False, f"Register indices must be integers at step {step}: {ins}", step_count
            
            if not (0 <= ra < regs_cap and 0 <= rb < regs_cap and 0 <= rd < regs_cap):
                return False, f"Register index out of bounds [0, {regs_cap-1}] at step {step}: {ins}", step_count
            
            va, vb = regs[ra], regs[rb]
            if va is None or vb is None:
                return False, f"Read from uninitialized register (ra={ra}:{va}, rb={rb}:{vb}) at step {step}", step_count

            # 演算の実行
            if op == '+':
                out = va + vb
            elif op == '-':
                out = va - vb
                if out <= 0:
                    return False, f"Non-positive subtraction result ({va} - {vb} = {out}) at step {step}", step_count
            elif op == '*':
                out = va * vb
            elif op == '/':
                if vb <= 0:
                    return False, f"Division by non-positive value ({vb}) at step {step}", step_count
                if va % vb != 0:
                    return False, f"Inexact division ({va} / {vb}, remainder {va % vb}) at step {step}", step_count
                out = va // vb
            else:
                return False, f"Invalid opcode '{op}' at step {step}", step_count

            if out <= 0 or out > limit:
                return False, f"Value {out} out of bounds (1, {limit}) at step {step}", step_count

            # デスティネーションへの書き込み
            regs[rd] = out
            remaining.discard(out)

        if remaining:
            return False, f"{len(remaining)} targets unreached (e.g. {list(remaining)[:3]})", step_count

        return True, "OK", step_count
