"""Task 2 (dqi): 890-Qubit Reed-Solomon MBUC Decoder Circuit Compiler & Generator.
   (日)Task 2 (dqi)：890量子ビットのリード・ソロモン符号 MBUC デコーダ回路コンパイラ＆生成器。
     - Qubit量子ビット）：量子計算における情報の基本単位。
     - Reed-Solomonリード・ソロモン符号）：データの誤り検出・訂正に用いられる誤り訂正符号。
     - Decoder(デコーダ）：符号化されたデータから、誤りを検出・訂正して元の情報を復元する仕組み。

Generates `workspace/output/dqi.kmb` for Benchmark II:
    Reversible Reed-Solomon syndrome decoder RS(4095, 4025) over GF(2^12).
Benchmark II 用の workspace/output/dqi.kmb を生成:
    GF(2¹²) 上で動作する、RS(4095, 4025) の可逆リード・ソロモン符号シンドローム・デコーダを生成。

Supports two modes:
1. `frontier` (default): Decompresses the 890-qubit GF(2^24)-closed base decoder
   (`dqi_575b043f.kmb.xz`) and runs the KMB binary surgical compiler (`PW4_SITES`)
   that applies 6 exact AND-product retention & measurement-based uncomputation
   (`CCX -> CX + HMR + CZ_if`) rewrites on clean ancilla wires `q865, q866, q880`,
   producing `5f901c55` (890 qubits, 367,983 static Toffolis, ratio ~70.54x).
   (日)frontier（デフォルト）：890量子ビットの GF(2²⁴) で閉じたベース・デコーダ
   (dqi_575b043f.kmb.xz) を展開し、KMBバイナリ・サージカル・コンパイラ（PW4_SITES）を実行、
     - Decompresses(展開する)：圧縮された回路ファイルを解凍する処理。
     - Base decoder(ベース・デコーダ)：コンパイラによる最適化を適用する前の、基礎となるデコーダ回路。
     - GF(2²⁴)-closed：GF(2²⁴) の演算体系の中で必要な演算が閉じている、という数学的性質。
     - Surgical compiler(サージカル・コンパイラ)：回路全体を一律に変換するのではなく、特定の箇所を狙って部分的に変換・最適化するコンパイラを意味する表現です。
   このコンパイラは、クリーンな補助量子ビット（アンシラ）である q865、q866、q880 に対して、
   6箇所の厳密な AND積の保持と、測定ベースの計算解除（Measurement-based Uncomputation）を適用します。
   具体的には、CCX → CX + HMR + CZ_if という回路変換を行います。
     - AND-product retention(AND積の保持)：AND演算によって得られた計算結果を、回路内で利用・保持することを指します。
     - lean ancilla(クリーンなアンシラ)：計算開始時に \(|0\rangle\) などの既知の初期状態に準備されている補助量子ビット。
     - Measurement-based uncomputation(測定ベースの計算解除)：測定を利用して、計算途中に使用した補助情報や演算の影響を取り除き、必要な状態を維持しながら回路を整理する手法。
   その結果、5f901c55 が生成されます(890量子ビット、静的Toffoliゲート数367,983個、約70.54倍の比率)。
     - 890 qubits：回路が使用する量子ビット数。
     - Static Toffolis：回路中に配置されている静的なToffoli演算の数。
     - ratio ~70.54x：約70.54倍。

2. `reference_builder`: Builds the standalone synchronized Extended Euclidean Algorithm
   (Proos & Zalka) Reed-Solomon decoder from scratch using `kickmix.CircuitBuilder`.
   (日)kickmix.CircuitBuilder を使用して、Proos & Zalka による同期型拡張ユークリッドアルゴリズムを用いた
       リード・ソロモン符号デコーダを、独立した回路としてゼロから構築します。

Credits & Lineage (Apache-2.0):
- Original 890Q/889Q DQI generator line: Syed Asad Ali (`syed3000/vqcc-own-circuit-portfolio`)
- C8 milestone product retention: Hayato Hikawa (`hayatohikawa`)
- HW-27 / qhdl measure-out & measure-first: James Barlow (`manwithacat`)
- 41 & 49 norm-preserving GF(2^24) tower closings + 4-pair window retention (`575b043f`):
  Balram Srinivasan & Grant Prater (`grantprater/vqcc-dqi-61c8348d-merged-closings`)
- cmerge + 37-chain L1 patch & 6-site PW4 compiler (`5f901c55`): Atakan Aldemir & boristown
"""

from __future__ import annotations

import argparse
import hashlib
import lzma
import pathlib
import struct
import sys
import time
from typing import Any

import kickmix as km
import numpy as np

SRC_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE_DIR = SRC_DIR.parent
VERIFIERS_DIR = WORKSPACE_DIR / 'verifiers'
DQI_FRONTIER_DIR = WORKSPACE_DIR / 'input' / 'dqi_frontier'
CACHE_DIR = WORKSPACE_DIR / 'input' / 'cache'
OUTPUT_DIR = WORKSPACE_DIR / 'output'

if str(VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFIERS_DIR))

from dqi import dqi_reference_circuit, verify_submission_dqi  # noqa: E402

MAGIC = bytes.fromhex('d750c7d5c329d326e3cc9f6834f2b8bf')
CODE_CX = 0x21
CODE_CZ_COND = 0x20 | 128
CODE_CCX = 0x31
CODE_HMR = 0x50

EXPECTED_575B_SHA256 = '575b043f4fee91a0fc3a9fd6fc1e9290e7b2a0669cae79886be2bf999f7ec155'
EXPECTED_5F90_SHA256 = '5f901c556cafd02dc11641eee7766171a454e3a0b6687d890ff9f99944e1934c'

# 3 condition-block pair-window sites + 3 02_chain AND-retention sites (-6 Toffolis -> 367,983 static T)
PW3_SITES = [
    (48587880, 887, 885, 876, 'compute', 880, None),
    (48607783, 887, 885, 867, 'uncompute', 880, 66726),
    (49580218, 869, 864, 871, 'compute', 880, None),
    (49600121, 869, 864, 876, 'uncompute', 880, 66727),
    (52290110, 875, 873, 886, 'compute', 880, None),
    (52310015, 875, 873, 870, 'uncompute', 880, 66728),
]

PW4_SITES = [
    (38181775, 872, 684, 720, 'compute', 865, None),
    (38227858, 872, 684, 720, 'uncompute', 865, 66729),
    (39970292, 869, 48, 108, 'compute', 866, None),
    (39970295, 869, 49, 109, 'compute', 880, None),
    (40016375, 869, 48, 108, 'uncompute', 866, 66729),
    (40016378, 869, 49, 109, 'uncompute', 880, 66729),
    *PW3_SITES,
]


def compile_pw_sites(data: bytes, sites: list[tuple], min_bits: int = 66730) -> bytes:
    """Apply exact product-retention & MBUC (`CCX -> CX + HMR + CZ_if`) rewrites to a KMB binary stream."""
    if data[:16] != MAGIC:
        raise ValueError('Invalid KMB magic header.')
    h = list(struct.unpack_from('<4I9Q', data, 16))
    if h[0] != 2:
        raise ValueError(f'Unsupported KMB version: {h[0]}')

    pos = 104
    reg_chunks = []
    sec = {}
    while pos < len(data):
        i, n = struct.unpack_from('<QQ', data, pos)
        pay = data[pos + 16 : pos + 16 + n]
        if i == 0:
            reg_chunks.append(data[pos : pos + 16 + n])
        else:
            sec[i] = pay
        pos += 16 + n

    arr = {i: np.frombuffer(sec[i], dtype=np.uint8 if i == 1 else np.uint32) for i in range(1, 10)}
    t = arr[1]
    base_t = t & 127
    l3 = np.isin(base_t, [0x30, 0x31, 0x32])
    l2 = np.isin(base_t, [0x20, 0x21, 0x22])
    l1 = np.isin(base_t, [0x10, 0x11, 0x12, 0x50])
    lb = np.isin(base_t, [0x40, 0x41, 0x42, 0x50])
    lc = (t & 128) != 0

    edits = []
    for op_i, ea, eb, et, mode, w_clean, cbit in sites:
        assert int(t[op_i]) == CODE_CCX, f'Expected CCX at op {op_i}, got {t[op_i]}'
        p3 = int(l3[:op_i].sum())
        p2 = int(l2[:op_i].sum())
        p1 = int(l1[:op_i].sum())
        pb = int(lb[:op_i].sum())
        pc = int(lc[:op_i].sum())
        assert (int(arr[2][p3]), int(arr[3][p3]), int(arr[4][p3])) == (ea, eb, et), (
            op_i,
            int(arr[2][p3]),
            int(arr[3][p3]),
            int(arr[4][p3]),
        )
        edits.append((op_i, p3, p2, p1, pb, pc, ea, eb, et, mode, w_clean, cbit))

    def splice(a: np.ndarray, splits_and_repls: list[tuple[int, int, list[int]]], dtype: Any) -> np.ndarray:
        out = []
        prev = 0
        for pos_idx, del_n, ins in splits_and_repls:
            out.append(a[prev:pos_idx])
            if ins:
                out.append(np.array(ins, dtype=dtype))
            prev = pos_idx + del_n
        out.append(a[prev:])
        return np.concatenate(out)

    s1, s2, s3, s4, s5, s6, s7, s8, s9 = [], [], [], [], [], [], [], [], []
    for op_i, p3, p2, p1, pb, pc, ea, eb, et, mode, w_clean, cbit in edits:
        if mode == 'compute':
            # CCX(ea, eb, et) -> CCX(ea, eb, w_clean), CX(w_clean, et)
            s1.append((op_i, 1, [CODE_CCX, CODE_CX]))
            s2.append((p3, 1, [ea]))
            s3.append((p3, 1, [eb]))
            s4.append((p3, 1, [w_clean]))
            s5.append((p2, 0, [w_clean]))
            s6.append((p2, 0, [et]))
        else:
            # CCX(ea, eb, et) -> CX(w_clean, et), HMR(w_clean, cbit), CZ_if(ea, eb, cbit)
            s1.append((op_i, 1, [CODE_CX, CODE_HMR, CODE_CZ_COND]))
            s2.append((p3, 1, []))
            s3.append((p3, 1, []))
            s4.append((p3, 1, []))
            s5.append((p2, 0, [w_clean, ea]))
            s6.append((p2, 0, [et, eb]))
            s7.append((p1, 0, [w_clean]))
            s8.append((pb, 0, [cbit]))
            s9.append((pc, 0, [cbit]))

    new_arr = {
        1: splice(arr[1], s1, np.uint8),
        2: splice(arr[2], s2, np.uint32),
        3: splice(arr[3], s3, np.uint32),
        4: splice(arr[4], s4, np.uint32),
        5: splice(arr[5], s5, np.uint32),
        6: splice(arr[6], s6, np.uint32),
        7: splice(arr[7], s7, np.uint32),
        8: splice(arr[8], s8, np.uint32),
        9: splice(arr[9], s9, np.uint32),
    }

    h[2] = max(int(h[2]), min_bits)
    h[4] = len(new_arr[1])
    h[5] = len(new_arr[2])
    h[6] = len(new_arr[5])
    h[7] = len(new_arr[7])
    h[8] = len(new_arr[8])
    h[9] = len(new_arr[9])

    hdr = MAGIC + struct.pack('<4I9Q', *h)
    chunks = [hdr] + reg_chunks
    for i in range(1, 10):
        b = new_arr[i].tobytes()
        chunks.append(struct.pack('<QQ', i, len(b)) + b)
    for i in (10, 11):
        if i in sec:
            b = sec[i]
            chunks.append(struct.pack('<QQ', i, len(b)) + b)
    return b''.join(chunks)


def generate_frontier_dqi(output_path: pathlib.Path, variant: str = '5f901c55') -> tuple[int, str]:
    """Compile the 890-qubit DQI circuit and write to output_path."""
    xz_path = DQI_FRONTIER_DIR / 'dqi_575b043f.kmb.xz'
    if variant in ('5f901c55', '575b043f') and xz_path.exists():
        print(f'  Decompressing base decoder {xz_path.name} ...', flush=True)
        base_bytes = lzma.decompress(xz_path.read_bytes())
        base_sha = hashlib.sha256(base_bytes).hexdigest()
        assert base_sha == EXPECTED_575B_SHA256, f'Base SHA mismatch: {base_sha}'
        if variant == '5f901c55':
            print('  Applying 6-site PW4 AND-retention & MBUC compiler pass ...', flush=True)
            out_bytes = compile_pw_sites(base_bytes, PW4_SITES, min_bits=66730)
            del base_bytes
            out_sha = hashlib.sha256(out_bytes).hexdigest()
            assert out_sha == EXPECTED_5F90_SHA256, f'Compiled SHA mismatch: {out_sha}'
        else:
            out_bytes = base_bytes
            out_sha = base_sha
    elif (CACHE_DIR / 'dqi.kmb').exists():
        print('  Falling back to cached 890Q decoder in workspace/input/cache/dqi.kmb ...', flush=True)
        out_bytes = (CACHE_DIR / 'dqi.kmb').read_bytes()
        out_sha = hashlib.sha256(out_bytes).hexdigest()
    else:
        raise FileNotFoundError('No DQI source found in workspace/input/.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(out_bytes)
    n_bytes = len(out_bytes)
    del out_bytes
    return n_bytes, out_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--mode',
        choices=('frontier', 'reference_builder'),
        default='frontier',
        help="Compilation mode: 'frontier' (890Q SOTA compiler) or 'reference_builder' (Proos & Zalka EEA).",
    )
    parser.add_argument(
        '--variant',
        choices=('5f901c55', '575b043f', 'cache'),
        default='5f901c55',
        help='Frontier DQI variant to compile (default: 5f901c55, 367,983 static Toffolis).',
    )
    parser.add_argument(
        '--output',
        type=pathlib.Path,
        default=OUTPUT_DIR / 'dqi.kmb',
        help='Output .kmb file path (default: workspace/output/dqi.kmb).',
    )
    parser.add_argument(
        '--verify-fast',
        action='store_true',
        default=True,
        help='Run a 256-shot verification check after compiling (default: True).',
    )
    args = parser.parse_args()

    t0 = time.monotonic()
    print('=' * 72)
    print(f' [Task 2: dqi] Compiling circuit (mode={args.mode}, variant={args.variant})')
    print('=' * 72)

    if args.mode == 'frontier':
        n_bytes, sha = generate_frontier_dqi(args.output, variant=args.variant)
    else:
        circuit = dqi_reference_circuit.build_reference_circuit(4095, 70, 12)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        circuit.to_file(str(args.output), format='kmb')
        n_bytes = args.output.stat().st_size
        sha = hashlib.sha256(args.output.read_bytes()).hexdigest()

    circuit = km.Circuit.from_file(str(args.output), format='kmb')
    spec_full = verify_submission_dqi.ProblemSpecDQI.benchmark_instance()
    s_qubits, m_bits, anc_qubits = verify_submission_dqi.validate_registers(circuit, spec_full)
    stats = circuit.operation_statistics()

    print(f'  Output file      : {args.output}')
    print(f'  File size        : {n_bytes:,} bytes ({n_bytes / 1e6:.2f} MB)')
    print(f'  SHA-256          : {sha}')
    print(f'  Total Qubits (Q) : {circuit.num_qubits} (Syndromes={len(s_qubits)}, Ancillas={len(anc_qubits)})')
    print(f'  Classical Bits   : {circuit.num_bits:,} (MBUC={len(m_bits)})')
    print(f'  Total Operations : {len(circuit):,}')
    print(f'  Static Toffolis  : {stats.num_toffoli_like:,}')
    print(f'  Compilation Time : {time.monotonic() - t0:.2f}s')

    if args.verify_fast:
        t1 = time.monotonic()
        num_shots = 256
        log2_delta = verify_submission_dqi.DEFAULT_LOG2_DELTA * (
            num_shots / verify_submission_dqi.DEFAULT_NUM_SHOTS
        )
        spec_fast = verify_submission_dqi.ProblemSpecDQI.benchmark_instance(num_shots=num_shots)
        res = verify_submission_dqi.verify(circuit, spec_fast, log2_delta=log2_delta, max_workers=1)
        ratio = 24439463583672.93 / res['score']
        errs = res['value_errors'] + res['phase_errors']
        print('-' * 72)
        print(
            f'  [Fast Verifier (256 shots)] Status={res["status"]} | Mean Toffoli={res["toffoli_count"]:,.2f} '
            f'| Errors={errs}/256 | Cost={res["score"]:,.2f} | Ratio={ratio:.5f}x ({time.monotonic()-t1:.1f}s)'
        )
    print('=' * 72)


if __name__ == '__main__':
    main()
