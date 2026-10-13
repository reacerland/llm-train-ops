"""Post-training sampling: greedy generation from the trained weights.

Used for smoke-run human review: after training, each configured prompt is
continued greedily and written to samples.txt in the output directory.
"""

from pathlib import Path

import torch

from trainops.mix.tokenizer import TokenizerLoadError

from .config import PromptsConfig


def require_decode(tokenizer: object) -> None:
    """[prompts] needs a tokenizer that can decode; fail fast if it can't."""
    if not callable(getattr(tokenizer, "decode", None)):
        raise TokenizerLoadError(
            f"tokenizer {type(tokenizer).__name__!r} has no 'decode' method; "
            f"[prompts] sampling requires decode(ids) -> str")


def generate_samples(model: torch.nn.Module, tokenizer,
                     prompts: PromptsConfig, device: torch.device,
                     out_path: Path) -> None:
    """Greedily continue each prompt; write ">>> prompt\\n<generated>" blocks."""
    model.eval()
    max_seq_len = getattr(model, "max_seq_len", None)
    blocks = []
    with torch.no_grad():
        for text in prompts.texts:
            ids = list(tokenizer.encode(text))
            if max_seq_len is not None:
                ids = ids[:max(1, max_seq_len - prompts.max_new_tokens)]
            generated: list[int] = []
            for _ in range(prompts.max_new_tokens):
                if max_seq_len is not None and \
                        len(ids) + len(generated) >= max_seq_len:
                    break
                x = torch.tensor([ids + generated], dtype=torch.long,
                                 device=device)
                next_id = int(model(x)[0, -1].argmax().item())
                if next_id == tokenizer.eos_token_id:
                    break
                generated.append(next_id)
            blocks.append(f">>> {text}\n{tokenizer.decode(generated)}\n")
    Path(out_path).write_text("\n".join(blocks), encoding="utf-8")
