import click
import pandas as pd
from memorag.memorag import MemoRAG
import os
from tqdm import tqdm

PROMPT_TEMPLATE = "You are a helpful financial AI assistant. You will be provided with a financial document and a question.\nYour task is to generate a comprehensive and accurate answer.\n\nContext:\n{context}\n\nQuestion: {input}"

@click.command()
@click.option("--qa_data_path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help="Path to the QA data file.")
@click.option("--memory_dir", type=click.Path(file_okay=False, dir_okay=True, exists=True), required=True, help="Directory where the memory data is stored.")
@click.option("--compress_ratio", type=int, default=4, help="Compression ratio for the memory.")
@click.option("--result_save_path", type=click.Path(dir_okay=False, exists=False))
def cli(qa_data_path, memory_dir, compress_ratio: int, result_save_path: str):
    if not result_save_path.endswith(".parquet"):
        raise ValueError("result_save_path must end with .parquet")

    qa_df = pd.read_parquet(qa_data_path)

    pipe = MemoRAG(
        mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
        ret_model_name_or_path="intfloat/multilingual-e5-large-instruct",
        gen_model_name_or_path="Qwen/Qwen3-8B",
        beacon_ratio=compress_ratio,
    )
    if not os.path.exists(os.path.join(memory_dir, "memory.bin")):
        raise ValueError(f"Memory data not found in {memory_dir}. Please run the memory creation step first.")
    if not os.path.exists(os.path.join(memory_dir, "index.bin")):
        raise ValueError(f"Index data not found in {memory_dir}. Please run the memory creation step first.")

    pipe.load(memory_dir, print_stats=True)
    
    queries = qa_df["query"].tolist()
    answers = []
    for query in tqdm(queries):
        res = pipe(query=query,
                   task_type="memorag", max_new_tokens=4096, use_memory_answer=False,
                   prompt_template=PROMPT_TEMPLATE)
        answers.append(res)
    qa_df["generated_answer"] = answers
    qa_df.to_parquet(result_save_path, index=False)
    print(f"Results saved to {result_save_path}")


if __name__ == "__main__":
    cli()
