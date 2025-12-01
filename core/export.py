import argparse
from checkpoints import export_onnx_from_saved
from model import CONTEXT_SIZE

def main():
    parser = argparse.ArgumentParser(description="Export model from saved MoEGPT to ONNX and other formats")
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/model_final.pt",
        help="Path to saved model file",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="saved_models/tokenizer.json",
        help="Path to saved tokenizer.json",
    )
    parser.add_argument(
        "--export_onnx",
        type=str,
        default=None,
        help="If set, export the model to this ONNX path and exit",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Precision for inference: fp32, fp16, or bf16",
    )

    args = parser.parse_args()

    # ONNX export mode: no text generation
    if args.export_onnx is not None:
        export_onnx_from_saved(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            onnx_path=args.export_onnx,
            seq_len=CONTEXT_SIZE,
            precision=args.precision,
        )
        return


if __name__ == "__main__":
    main()
