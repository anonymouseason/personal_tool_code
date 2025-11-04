import os
import subprocess
import threading
import time
import logging
from typing import List, Optional
from dataclasses import dataclass

import torch

# ===================== 参数区域 =====================
PAUSE_THRESHOLD = 40     # 任意 GPU 利用率超过该阈值 => 暂停
POLL_INTERVAL   = 1.0    # nvidia-smi 轮询间隔（秒）
SYNC_INTERVAL_S = 2.0    # 同步线程触发同步/清理的最小间隔（秒）
STREAMS_PER_GPU = 4      # 每块 GPU 的并行 CUDA streams 数量（提升利用率）
BATCH_PER_STREAM = 8     # 每条 stream 连续提交的 matmul 批次数（增大以提高吃满）
# 默认矩阵规模和数量（越小越省显存；利用率不足时再慢慢加）
DEFAULT_MATRIX_SIZE     = 1024
DEFAULT_MATRICES_PER_GPU = 2
# ===================================================

def run_cmd(cmd: List[str]) -> str:
    completed = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )
    return completed.stdout.strip()

def get_gpu_utils_via_nvidia_smi() -> List[int]:
    out = run_cmd([
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    utils = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        utils.append(0 if line.upper() == "N/A" else int(line))
    return utils

class GPUUtilMonitor(threading.Thread):
    """
    后台监视 GPU 利用率。
    - 任一GPU > PAUSE_THRESHOLD => 设置 pause_event
    - 全部GPU == 0             => 清除 pause_event
    """
    def __init__(self, pause_event: threading.Event, stop_event: threading.Event, logger: logging.Logger):
        super().__init__(daemon=True)
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.logger = logger
        self.last_state = "INIT"

    def run(self):
        while not self.stop_event.is_set():
            try:
                utils = get_gpu_utils_via_nvidia_smi()
            except Exception as e:
                self.logger.error(f"[Monitor] Failed to query nvidia-smi: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            if not utils:
                self.logger.warning("[Monitor] No GPU utils found from nvidia-smi")
                time.sleep(POLL_INTERVAL)
                continue

            any_over = any(u > PAUSE_THRESHOLD for u in utils)
            all_zero = all(u == 0 for u in utils)

            if any_over:
                self.pause_event.set()
                state = "PAUSED"
            elif all_zero:
                self.pause_event.clear()
                state = "RUNNING"
            else:
                state = "PAUSED" if self.pause_event.is_set() else "RUNNING"

            utils_str = ", ".join(f"GPU{i}={u}%" for i, u in enumerate(utils))
            if state != self.last_state:
                self.logger.info(f"[Monitor] {utils_str} -> STATE={state}")
                self.last_state = state
            else:
                self.logger.debug(f"[Monitor] {utils_str} (STATE={state})")

            time.sleep(POLL_INTERVAL)

class SyncThread(threading.Thread):
    """
    同步/清理线程：
    - 当 pause 中：执行 torch.cuda.synchronize + empty_cache，尽快“刹车”
    - 当运行中：按固定间隔少量同步，既让任务排队长一些，又避免无限排队导致 OOM
    """
    def __init__(self, devices: List[torch.device], pause_event: threading.Event, stop_event: threading.Event, logger: logging.Logger):
        super().__init__(daemon=True)
        self.devices = devices
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.logger = logger
        self._last_sync = 0.0

    def run(self):
        while not self.stop_event.is_set():
            now = time.time()
            if self.pause_event.is_set():
                for d in self.devices:
                    try:
                        torch.cuda.synchronize(d)
                    except Exception as e:
                        self.logger.warning(f"[Sync] synchronize failed on {d}: {e}")
                # 回收未使用显存，降低峰值
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(0.2)
                continue

            if now - self._last_sync >= SYNC_INTERVAL_S:
                # 少量同步，防止任务无限积压
                for d in self.devices:
                    try:
                        # 使用 Event 精细同步也可，这里简单全卡同步
                        torch.cuda.synchronize(d)
                    except Exception as e:
                        self.logger.debug(f"[Sync] non-critical sync failed on {d}: {e}")
                self._last_sync = now

            time.sleep(0.05)


class ComputeWorker(threading.Thread):
    """
    计算线程（每个进程一个线程，内部多 streams 并行）
    - 复用矩阵缓冲区（降低显存）
    - 多 stream 并行提交 matmul（提高利用率）
    - 混合精度/TF32（进一步降显存、提吞吐）
    """
    def __init__(
        self,
        devices: List[torch.device],
        matrix_size: int,
        matrices_per_gpu: int,
        pause_event: threading.Event,
        stop_event: threading.Event,
        logger: logging.Logger,
    ):
        super().__init__(daemon=True)
        self.devices = devices
        self.matrix_size = matrix_size
        self.matrices_per_gpu = matrices_per_gpu
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.logger = logger

        # ======== 精度/后端优化 ========
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # 优先使用 FP16；不支持时自动回退 FP32
        self.dtype = torch.float16
        try:
            _ = torch.zeros(1, device=devices[0], dtype=self.dtype)
        except Exception:
            self.dtype = torch.float32

        # ======== 为每块 GPU 准备数据和 streams ========
        self.streams_per_gpu = STREAMS_PER_GPU
        self.batch_per_stream = BATCH_PER_STREAM

        # [(device, [A_i], [tmp_i_per_stream], [streams]), ...]
        self._per_gpu_data = []
        for d in self.devices:
            with torch.cuda.device(d):
                # 确保 A 的数量 >= streams 数，避免多 stream 写同一个 A
                num_As = max(self.matrices_per_gpu, self.streams_per_gpu)
                As = [
                    torch.randn(self.matrix_size, self.matrix_size, device=d, dtype=self.dtype)
                    for _ in range(num_As)
                ]
                # 为每条 stream 准备独立 tmp，避免并发写冲突
                streams = [torch.cuda.Stream(device=d) for _ in range(self.streams_per_gpu)]
                tmps = [
                    torch.empty(self.matrix_size, self.matrix_size, device=d, dtype=self.dtype)
                    for _ in range(self.streams_per_gpu)
                ]
                self._per_gpu_data.append((d, As, tmps, streams))

        self.iteration = 0

    @staticmethod
    def _submit_batch_on_stream(A: torch.Tensor, tmp: torch.Tensor):
        """
        提交一小段计算：tmp = A @ A^T ; A = tmp @ A
        说明：
        - 使用 copy_ 复用缓冲，避免产生长期悬挂的大张量。
        - 不用 matmul_（版本兼容问题），统一用 copy_。
        """
        tmp.copy_(A @ A.t())   # tmp = A @ A^T
        A.copy_(tmp @ A)       # A  = tmp @ A

    def run(self):
        self.logger.info(
            f"[Compute] dtype={self.dtype}, streams/GPU={self.streams_per_gpu}, batch/stream={self.batch_per_stream}"
        )
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue

            try:
                self.iteration += 1
                # 遍历每块 GPU
                for (d, As, tmps, streams) in self._per_gpu_data:
                    with torch.cuda.device(d):
                        # 每条 stream 绑定一个独立 A 和 tmp
                        for si, stream in enumerate(streams):
                            A = As[si]
                            tmp = tmps[si]
                            # 把这条批次的活提交到该 stream
                            with torch.cuda.stream(stream):
                                for _ in range(self.batch_per_stream):
                                    self._submit_batch_on_stream(A, tmp)

                if self.iteration % 200 == 0:
                    self.logger.info(f"[Compute] queued iteration={self.iteration}")

            except RuntimeError as e:
                self.logger.error(f"[Compute] RuntimeError: {e}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"[Compute] Unexpected error: {e}")
                time.sleep(0.5)

class GPUManager:
    """GPU管理：启动监控、计算、同步三线程"""
    def __init__(self, matrix_size=DEFAULT_MATRIX_SIZE, matrices_per_gpu=DEFAULT_MATRICES_PER_GPU):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("GPUManager")

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPU available")

        self.devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        self.logger.info(f"Found {num_gpus} GPU(s): {', '.join(map(str, self.devices))}")

        self.pause_event = threading.Event()
        self.stop_event  = threading.Event()

        self.monitor = GPUUtilMonitor(self.pause_event, self.stop_event, self.logger)
        self.syncer   = SyncThread(self.devices, self.pause_event, self.stop_event, self.logger)
        self.worker   = ComputeWorker(
            self.devices,
            matrix_size=matrix_size,
            matrices_per_gpu=matrices_per_gpu,
            pause_event=self.pause_event,
            stop_event=self.stop_event,
            logger=self.logger,
        )

    def run(self):
        self.logger.info("Starting Monitor/Sync/Compute threads...")
        self.monitor.start()
        self.syncer.start()
        self.worker.start()
        try:
            # 主线程仅保持存活（可扩展热键/命令处理）
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received, stopping...")
        finally:
            self.logger.info("Stopping threads...")
            self.stop_event.set()
            # 先解除暂停，防止线程卡住
            self.pause_event.clear()
            for t in (self.worker, self.syncer, self.monitor):
                t.join(timeout=2.0)
            self.logger.info("All threads stopped.")

def main():
    try:
        mgr = GPUManager()
        mgr.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()