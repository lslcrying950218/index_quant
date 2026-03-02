"""
ONNX Runtime 推理引擎
- 将PyTorch/LightGBM模型导出为ONNX格式
- 高性能推理 (GPU/CPU)
- 延迟 < 5ms
- 模型热更新
"""
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime 未安装, 将回退到PyTorch推理")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ONNXInferenceEngine:
    """ONNX Runtime 推理引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda")
        self.model_dir = Path(config.get("model_dir", "models"))

        # 推理会话
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._lock = threading.Lock()

        # 性能统计
        self._inference_count = 0
        self._total_latency_ms = 0.0
        self._max_latency_ms = 0.0

        # 初始化ONNX Runtime
        if ORT_AVAILABLE:
            self._init_providers()

    def _init_providers(self):
        """初始化执行提供者"""
        available = ort.get_available_providers()
        logger.info(f"ONNX Runtime可用Provider: {available}")

        if "CUDAExecutionProvider" in available and "cuda" in self.device:
            self.providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                }),
                "CPUExecutionProvider",
            ]
            logger.info("✅ ONNX Runtime 使用 CUDA 加速")
        else:
            self.providers = ["CPUExecutionProvider"]
            logger.info("ONNX Runtime 使用 CPU")

    # ==================== 模型加载 ====================

    def load_model(self, model_name: str, model_path: str) -> bool:
        """加载ONNX模型"""
        if not ORT_AVAILABLE:
            logger.error("onnxruntime 未安装")
            return False

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2
            sess_options.enable_mem_pattern = True

            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=self.providers,
            )

            with self._lock:
                self._sessions[model_name] = session

            # 打印模型信息
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            logger.info(
                f"✅ ONNX模型加载: {model_name}\n"
                f"   输入: {[(i.name, i.shape, i.type) for i in inputs]}\n"
                f"   输出: {[(o.name, o.shape, o.type) for o in outputs]}"
            )
            return True

        except Exception as e:
            logger.error(f"ONNX模型加载失败 {model_name}: {e}")
            return False

    def unload_model(self, model_name: str):
        """卸载模型"""
        with self._lock:
            if model_name in self._sessions:
                del self._sessions[model_name]
                logger.info(f"模型已卸载: {model_name}")

    # ==================== 推理 ====================

    def predict(self, model_name: str,
                inputs: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """
        执行推理

        Args:
            model_name: 模型名称
            inputs: {"input_name": ndarray, ...}

        Returns:
            {"output_name": ndarray, ...} 或 None
        """
        with self._lock:
            session = self._sessions.get(model_name)

        if session is None:
            logger.error(f"模型未加载: {model_name}")
            return None

        try:
            t0 = time.perf_counter()

            # 确保输入类型正确
            feed = {}
            for inp in session.get_inputs():
                if inp.name in inputs:
                    arr = inputs[inp.name]
                    if inp.type == "tensor(float)":
                        arr = arr.astype(np.float32)
                    elif inp.type == "tensor(double)":
                        arr = arr.astype(np.float64)
                    elif inp.type == "tensor(int64)":
                        arr = arr.astype(np.int64)
                    feed[inp.name] = arr

            # 执行推理
            output_names = [o.name for o in session.get_outputs()]
            results = session.run(output_names, feed)

            latency_ms = (time.perf_counter() - t0) * 1000

            # 统计
            self._inference_count += 1
            self._total_latency_ms += latency_ms
            self._max_latency_ms = max(self._max_latency_ms, latency_ms)

            if latency_ms > 10:
                logger.warning(
                    f"推理延迟偏高: {model_name} {latency_ms:.2f}ms"
                )

            return dict(zip(output_names, results))

        except Exception as e:
            logger.error(f"ONNX推理异常 {model_name}: {e}")
            return None

    def predict_alpha(self, factor_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alpha模型推理 (封装接口)

        Args:
            factor_matrix: (n_stocks, seq_len, n_factors)

        Returns:
            (pred_returns, pred_vols) each shape (n_stocks,)
        """
        result = self.predict("alpha_transformer", {
            "factor_input": factor_matrix.astype(np.float32)
        })

        if result is None:
            n = factor_matrix.shape[0]
            return np.zeros(n), np.ones(n) * 0.02

        pred_returns = result.get("pred_return", np.zeros(factor_matrix.shape[0]))
        pred_vols = result.get("pred_volatility", np.ones(factor_matrix.shape[0]) * 0.02)

        return pred_returns.squeeze(), pred_vols.squeeze()

    # ==================== 模型导出 ====================

    @staticmethod
    def export_pytorch_to_onnx(model: 'torch.nn.Module',
                                dummy_input: 'torch.Tensor',
                                output_path: str,
                                input_names: List[str] = None,
                                output_names: List[str] = None,
                                dynamic_axes: dict = None):
        """将PyTorch模型导出为ONNX"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch 未安装, 无法导出")
            return False

        try:
            if input_names is None:
                input_names = ["factor_input"]
            if output_names is None:
                output_names = ["pred_return", "pred_volatility"]
            if dynamic_axes is None:
                dynamic_axes = {
                    "factor_input": {0: "batch_size"},
                    "pred_return": {0: "batch_size"},
                    "pred_volatility": {0: "batch_size"},
                }

            model.eval()
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
            )
            logger.info(f"✅ ONNX模型导出成功: {output_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            return False

    @staticmethod
    def export_lgb_to_onnx(lgb_model, n_features: int, output_path: str):
        """将LightGBM模型导出为ONNX"""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnxmltools
            from onnxmltools.convert import convert_lightgbm

            initial_type = [("X", FloatTensorType([None, n_features]))]
            onnx_model = convert_lightgbm(
                lgb_model, initial_types=initial_type,
                target_opset=17,
            )

            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"✅ LightGBM ONNX导出成功: {output_path}")
            return True

        except ImportError:
            logger.error("需要安装 onnxmltools: pip install onnxmltools")
            return False
        except Exception as e:
            logger.error(f"LightGBM ONNX导出失败: {e}")
            return False

    # ==================== 热更新 ====================

    def hot_reload(self, model_name: str, new_model_path: str) -> bool:
        """模型热更新 (不中断服务)"""
        logger.info(f"🔄 模型热更新: {model_name} <- {new_model_path}")

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            new_session = ort.InferenceSession(
                new_model_path,
                sess_options=sess_options,
                providers=self.providers,
            )

            # 原子替换
            with self._lock:
                old_session = self._sessions.get(model_name)
                self._sessions[model_name] = new_session

            # 旧session自动GC
            logger.info(f"✅ 模型热更新完成: {model_name}")
            return True

        except Exception as e:
            logger.error(f"模型热更新失败: {e}")
            return False

    # ==================== 统计 ====================

    def get_stats(self) -> dict:
        avg_latency = (
            self._total_latency_ms / self._inference_count
            if self._inference_count > 0 else 0
        )
        return {
            "inference_count": self._inference_count,
            "avg_latency_ms": round(avg_latency, 3),
            "max_latency_ms": round(self._max_latency_ms, 3),
            "loaded_models": list(self._sessions.keys()),
        }
