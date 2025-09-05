# --- 放在文件最顶端 ---
import os, sys
# 把项目根目录加入 sys.path（snr_scan.py 在 evaluate/ 下，所以上一级是根）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 兼容两种目录结构：有 models/ 子包 或直接放在根目录
try:
    from models.rescnn import ResCNN
    from models.transformer import MIMOTransformer
except ModuleNotFoundError:
    from rescnn import ResCNN
    from transformer import MIMOTransformer


import os, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# === 你的模型与指标 ===
from models.rescnn import ResCNN                         # :contentReference[oaicite:0]{index=0}
from models.transformer import MIMOTransformer          # :contentReference[oaicite:1]{index=1}
from evaluate.metrics import compute_nmse               # :contentReference[oaicite:2]{index=2}

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def smart_load_weights(model, ckpt_path, device):
    """兼容两种保存格式：直接 state_dict 或 {'model_state_dict': ...}"""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model

def add_awgn_on_real_imag(x, snr_db, real_ch=0, imag_ch=1):
    """
    只对实部/虚部加 AWGN，保持其它通道（如位置编码）不变。
    x: np.ndarray, [N, n_observed, C]  (C>=2, 你的输入是 3 维: Re/Im/pos)
    """
    x_noisy = x.copy()
    real = x[:, :, real_ch]
    imag = x[:, :, imag_ch]
    # 以整个数据的平均功率估计噪声方差（也可改成每样本自适应）
    signal_power = np.mean(real**2 + imag**2)
    sigma = np.sqrt(signal_power / (10.0**(snr_db/10.0))) / np.sqrt(2.0)

    noise_r = np.random.normal(0.0, sigma, size=real.shape)
    noise_i = np.random.normal(0.0, sigma, size=imag.shape)
    x_noisy[:, :, real_ch] = real + noise_r
    x_noisy[:, :, imag_ch] = imag + noise_i
    return x_noisy

@torch.no_grad()
def batched_predict(model, X, device, batch_size=256):
    """
    X: np.ndarray [N, n_observed, input_dim]
    return: np.ndarray [N, n_predict, 2]
    """
    model.eval()
    out_list = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        yb = model(xb).detach().cpu().numpy()
        out_list.append(yb)
    return np.concatenate(out_list, axis=0)

def nmse_db_np(y_pred, y_true):
    """和你的 metrics 一致的定义，只是 numpy 版用于画图"""
    # 转成 torch 用你现成函数也行；这里避免反复 tensor->numpy
    pred_c = y_pred[..., 0] + 1j * y_pred[..., 1]
    true_c = y_true[..., 0] + 1j * y_true[..., 1]
    num = np.mean(np.abs(pred_c - true_c)**2)
    den = np.mean(np.abs(true_c)**2) + 1e-12
    return 10.0 * np.log10(num / den)

def main():
    ap = argparse.ArgumentParser(description="SNR Robustness Scan for ResCNN / Transformer / Ensemble")
    ap.add_argument("--rescnn_ckpt", default="checkpoints/rescnn_best.pth")
    ap.add_argument("--trans_ckpt",  default="checkpoints/transformer_best.pth")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="Ensemble weight: y = alpha * ResCNN + (1-alpha) * Transformer")
    ap.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save", default="results/snr_scan.png")
    ap.add_argument("--test_dir", default="data/test", help="若目录存在，则优先从此处加载 X/Y")
    ap.add_argument("--x_path", default="X_data.npy", help="备用：根目录下的测试 X")
    ap.add_argument("--y_path", default="Y_data.npy", help="备用：根目录下的测试 Y")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    cfg = load_config("config.json")              # 读到 n_observed/n_predict/input 维度等 :contentReference[oaicite:3]{index=3}
    n_obs = cfg["n_observed"]
    n_pre = cfg["n_predict"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------ 加载数据（两条路径任选其一） ------------
    if os.path.isdir(args.test_dir) and \
       os.path.exists(os.path.join(args.test_dir, "X_data.npy")) and \
       os.path.exists(os.path.join(args.test_dir, "Y_data.npy")):
        X = np.load(os.path.join(args.test_dir, "X_data.npy"))
        Y = np.load(os.path.join(args.test_dir, "Y_data.npy"))
    else:
        X = np.load(args.x_path)   # 你已上传的 X_data.npy
        Y = np.load(args.y_path)   # 你已上传的 Y_data.npy

    # 形状自检（你的模型 forward 约定：输入 [B, n_observed, 3]；输出 [B, n_predict, 2]） :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    assert X.ndim == 3 and X.shape[1] == n_obs, f"X shape expect (*,{n_obs},C) but got {X.shape}"
    assert Y.ndim == 3 and Y.shape[1] == n_pre and Y.shape[2] == 2, f"Y shape expect (*,{n_pre},2) but got {Y.shape}"
    print(f"Loaded test: X={X.shape}, Y={Y.shape}")

    # ------------ 构建与加载模型 ------------
    rescnn = ResCNN(input_dim=X.shape[2], n_observed=n_obs, n_predict=n_pre,
                    hidden_dim=128, n_layers=6).to(device)      # 超参与你训练一致 :contentReference[oaicite:6]{index=6}
    trans  = MIMOTransformer(input_dim=X.shape[2], n_observed=n_obs, n_predict=n_pre,
                             d_model=96, n_heads=3, n_layers=2, d_ff=192, dropout=0.1).to(device)  # 你常用配置

    if os.path.exists(args.rescnn_ckpt):
        smart_load_weights(rescnn, args.rescnn_ckpt, device)
        print("Loaded:", args.rescnn_ckpt)
    else:
        print("WARNING: ResCNN ckpt not found -> using randomly initialized model")

    if os.path.exists(args.trans_ckpt):
        smart_load_weights(trans, args.trans_ckpt, device)
        print("Loaded:", args.trans_ckpt)
    else:
        print("WARNING: Transformer ckpt not found -> using randomly initialized model")

    # ------------ SNR 扫描 ------------
    nmse_res_list, nmse_trs_list, nmse_ens_list = [], [], []

    np.random.seed(42)
    for snr in args.snrs:
        Xn = add_awgn_on_real_imag(X, snr)  # 只对 Re/Im 加噪
        Y_res = batched_predict(rescnn, Xn, device, args.batch)
        Y_trs = batched_predict(trans,  Xn, device, args.batch)
        Y_ens = args.alpha * Y_res + (1.0 - args.alpha) * Y_trs

        nmse_res = nmse_db_np(Y_res, Y)
        nmse_trs = nmse_db_np(Y_trs, Y)
        nmse_ens = nmse_db_np(Y_ens, Y)

        nmse_res_list.append(nmse_res)
        nmse_trs_list.append(nmse_trs)
        nmse_ens_list.append(nmse_ens)

        print(f"SNR {snr:>4} dB | ResCNN {nmse_res:6.2f} dB | Transformer {nmse_trs:6.2f} dB | Ensemble {nmse_ens:6.2f} dB")

    # ------------ 画图 ------------
    plt.figure(figsize=(8.2, 4.6), dpi=140)
    plt.plot(args.snrs, nmse_res_list, marker="o", label="ResCNN")
    plt.plot(args.snrs, nmse_trs_list, marker="o", label="Transformer")
    plt.plot(args.snrs, nmse_ens_list, marker="o", label=f"Ensemble (α={args.alpha})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("Robustness: NMSE vs SNR")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save)
    print("Saved:", args.save)

if __name__ == "__main__":
    main()
