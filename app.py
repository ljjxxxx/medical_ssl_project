import argparse
from src.app.interface import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='胸部X光疾病筛查应用')
    parser.add_argument('--model', type=str, default='./checkpoints/best_classifier_model.pt',
                        help='训练好的模型检查点路径')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (auto, mps, cuda, 或 cpu)')
    parser.add_argument('--port', type=int, default=7860, help='Gradio服务器端口')
    args = parser.parse_args()

    app = create_app(args)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=False)