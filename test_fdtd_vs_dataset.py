import os
import subprocess
import numpy as np

# =====================================================
# ================= PATH CONFIG =======================
# =====================================================
LUMERICAL_EXE = r"C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe"
WORKDIR = r"C:/holo/fdtd"

FSP_FILE = "Double-sided.fsp"
LSF_TEMPLATE = "Double-sided_template.lsf"
LSF_RUN = "Double-sided_run.lsf"

LOG_FILE = "fdtd.log"
OUTPUT_FILE = "single_data.txt"

# =====================================================
# =============== BASIC UTILITIES =====================
# =====================================================
def bit_matrix_to_index(bit_matrix):
    """
    6x6 binary matrix -> pixel_index (1~63)
    """
    import numpy as np

    bit_matrix = np.asarray(bit_matrix, dtype=np.float32)  # ★关键修复
    flat = bit_matrix.reshape(-1)

    index = []
    for i, v in enumerate(flat):
        if v > 0.5:
            index.append(i + 1)
    return index


# =====================================================
# ================= LSF HANDLING ======================
# =====================================================
def write_lsf_with_index(pixel_index):
    index_str = ",".join(str(i) for i in pixel_index)
    replacement = f"pixel_index=[{index_str}];\n"

    with open(os.path.join(WORKDIR, LSF_TEMPLATE), "r", encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    replaced = False
    for line in lines:
        if "pixel_index=[" in line:
            out.append(replacement)
            replaced = True
        else:
            out.append(line)

    if not replaced:
        raise RuntimeError("pixel_index=[...] not found in LSF")

    with open(os.path.join(WORKDIR, LSF_RUN), "w", encoding="utf-8") as f:
        f.writelines(out)


# =====================================================
# ================= RUN LUMERICAL =====================
# =====================================================
def run_lumerical():
    log_path = os.path.join(WORKDIR, LOG_FILE)
    if os.path.exists(log_path):
        os.remove(log_path)

    cmd = [
        LUMERICAL_EXE,
        FSP_FILE,
        "-nw",
        "-run", LSF_RUN
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=WORKDIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print(stdout)
        print(stderr)
        raise RuntimeError("Lumerical crashed")

# =====================================================
# ============ EXTRACT 60-D SPECTRUM ==================
# =====================================================
def debug_line_lengths(path, n=2):
    print("\n===== single_data.txt line lengths =====")
    with open(path, "r") as f:
        for i in range(n):
            line = f.readline()
            if not line:
                break
            count = len(line.split())
            print(f"line {i}: num_count = {count}")
    print("========================================\n")

def extract_full_spectrum():
    path = os.path.join(WORKDIR, OUTPUT_FILE)
    if not os.path.exists(path):
        raise RuntimeError("single_data.txt not generated")

    # debug_line_lengths(path, n=30)

    # ---------- 核心修复 ----------
    with open(path, "r") as f:
        lines = f.readlines()

    # 第一行是 pixel_index，跳过
    data_lines = lines[1:]
    data = np.array(
        [[complex(x.replace("i", "j")) for x in line.split()]
         for line in data_lines],
        dtype=np.complex64
    )

    # data shape: (103,16)
    # take 3 wavelengths → indices you used in dataset
    wl_idx = [530//10 - 52, 670//10 - 52, 800//10 - 52]

    result = []
    for idx in wl_idx:
        row = data[idx]
        # (Exx, Eyy, Exy, Eyx) forward only
        T = np.array([[row[0], row[2]],
                      [row[3], row[1]]], dtype=np.complex64)

        # poli = 10
        poli = 10
        A = np.array([[np.cos(i/poli*np.pi),
                       np.sin(i/poli*np.pi)] for i in range(poli)])

        for i in range(poli):
            val = A[i] @ T @ A[i].T
            result.append(np.real(val))
            result.append(np.imag(val))

    return np.array(result, dtype=np.float32)


# =====================================================
# ============ PUBLIC SINGLE SAMPLE API ================
# =====================================================
def run_lumerical_single(pixel_index):
    write_lsf_with_index(pixel_index)
    # run_lumerical()
    return extract_full_spectrum()


# =====================================================
# ===================== DATASET =======================
# =====================================================
def read_txt_to_2d_list(file_path, symbol,dataset):
    with open(file_path, 'r') as file:
        data = []
        line_count = 0
        for line in file:
            if line_count >= dataset:
                break
            line = line.strip()
            if line:
                # 假设数值由逗号分隔
                row = [float(num) for num in line.split(symbol)]
                data.append(row)
            line_count += 1
        return data
def binary_string_to_bit_list(file_path,dataset):
    with open(file_path, 'r') as file:
        data = []
        line_count = 0
        for line in file:
            if line_count >= dataset:
                break
            line = line.strip()
            bit_list = [float(bit) for bit in line]
            bit_matrix = [bit_list[i*6:i*6+6] for i in range(len(bit_list) // 6)]
            data.append(bit_matrix)
            line_count += 1
    return data


# =====================================================
# ======================= MAIN ========================
# =====================================================
if __name__ == "__main__":
    tn = 5  # test samples

    st = binary_string_to_bit_list('dataset/st_36.txt', tn)
    opr1 = read_txt_to_2d_list('dataset/opr_530.txt', ' ', tn)
    opr2 = read_txt_to_2d_list('dataset/opr_670.txt', ' ', tn)
    opr3 = read_txt_to_2d_list('dataset/opr_800.txt', ' ', tn)
    opr = [a+b+c for a,b,c in zip(opr1,opr2,opr3)]

    for i in range(tn):
        bit_matrix = st[i]
        index = bit_matrix_to_index(bit_matrix)

        fdtd_y = run_lumerical_single(index)
        dataset_y = np.array(opr[i], dtype=np.float32)

        error = np.mean(np.abs(fdtd_y - dataset_y))
        print(f"Sample {i}: MAE = {error:.6e}")
