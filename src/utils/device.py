import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_available_device():
    """
    检测并返回可用的最佳设备

    Returns:
        device: PyTorch设备 (cuda, mps, 或 cpu)
    """
    mps_available = False
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                test_tensor = torch.zeros(1, device="mps")
                mps_available = True
                logger.info("MPS acceleration available and working")
                return torch.device("mps")
            except RuntimeError as e:
                logger.warning(f"MPS reported available but failed: {e}")
                logger.warning("Falling back to CPU")
        else:
            if torch.backends.mps.is_available():
                logger.warning("MPS available but PyTorch was not built with MPS support")
            else:
                logger.info("MPS acceleration not available on this system")
    except AttributeError:
        logger.warning("PyTorch version does not support MPS")

    if torch.cuda.is_available():
        logger.info("CUDA acceleration available")
        return torch.device("cuda")

    logger.info("Using CPU for computation (no GPU acceleration available)")
    return torch.device("cpu")