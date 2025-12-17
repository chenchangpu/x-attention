import time
import torch
from transformers import StaticCache
from xattn.src.load_llama import load_model,FastPrefillConfig
import argparse
from tqdm import tqdm


def generate_prompt(tokenizer,target_len,):
    context = "A quick brown fox jumps over the lazy dog. \n"
    with open("demo/xattention.txt", "r") as f:
        needle = f.read()

    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_repetitions = target_len // num_tokens_context

    text = (
        "This is a very long story book with knowledge of XAttention, which you need to remember for later question: <book> "
        + context * int(num_repetitions * 0.5)
        + needle
        + context * int(num_repetitions * 0.5)
        + "</book>\n Based on the content of the book, please briefly tell me about XAttention.\nAnswer:"
    )

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    suffix_len = len(tokenizer("</book>\n Based on the content of the book, please briefly tell me about XAttention.\nAnswer:", add_special_tokens=False))
    over_len = input_ids.shape[1] - target_len
    input_ids = torch.cat([input_ids[:, :-suffix_len-100-over_len], input_ids[:, -suffix_len-100:]], dim=1) if over_len > 0 else input_ids
    return input_ids

if __name__ == "__main__":
    # load model and tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--len", type=int, default=131072)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--metric", type=str, default='xattn')
    parser.add_argument("--use-pooling", action="store_true")
    parser.add_argument("--block-sparse-kernel", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    config = FastPrefillConfig(metric = args.metric,stride = args.stride, threshold = args.threshold, 
                               use_pooling = True if args.use_pooling else False,
                               block_sparse_kernel= args.block_sparse_kernel)
    
    model, tokenizer = load_model(name_or_path="/data3/Llama-3-8B-Instruct-Gradient-1048k", fastprefillconfig=config)
    input_ids = generate_prompt(tokenizer,args.len)
    # -------------------
    # 1. Prefill
    # -------------------
    past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=131172, device=model.device, dtype=model.dtype)
    # warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            for i in range(0, input_ids.size(1), args.chunk_size):
                chunk = input_ids[:, i: i + args.chunk_size]
                output = model(
                    input_ids=chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                    num_logits_to_keep=1,
                )
                past_key_values = output.past_key_values 
        past_key_values.reset()   
    
    start_prefill = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, input_ids.size(1), args.chunk_size), desc="Prefilling", unit="chunk"):
            chunk = input_ids[:, i: i + args.chunk_size]
            output = model(
                input_ids=chunk,
                past_key_values=past_key_values,
                use_cache=True,
                num_logits_to_keep=1,
            )
            past_key_values = output.past_key_values
    torch.cuda.synchronize()
    end_prefill = time.time()
    prefill_time = end_prefill - start_prefill
    print(f"Prefill Time: {prefill_time:.4f} s")

     # -------------------
    # 2. Decode
    # -------------------
    start_decode = time.time()
    eos_token_id = tokenizer.eos_token_id  # 获取eos token ID
    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_content = [pred_token_idx.item()]
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in tqdm(range(50), desc="Decoding", unit="token"):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
                num_logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_token = pred_token_idx.item()
            
            if generated_token == eos_token_id:
                break  # 如果生成EOS token，提前结束
            
            generated_content += [generated_token]
    torch.cuda.synchronize()
    end_decode = time.time()
    decode_time = end_decode - start_decode
    print(f"Prefill Time: {prefill_time:.4f} s")
    print(f"Decode Time: {decode_time:.4f} s")
    output_text = tokenizer.decode(generated_content, skip_special_tokens=True)
    print("Generated Text:", output_text)
