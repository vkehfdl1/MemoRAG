import os

from tqdm import tqdm
from memorag.memorag import Memory
import click
import pandas as pd

PROMPT_TEMPLATE = "You are a helpful financial AI assistant.\nYour task is to generate a comprehensive and accurate answer.\n\nQuestion: {question}"


@click.command()
@click.option("--corpus_data_path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help="Path to the corpus data file.")
@click.option("--qa_data_path", type=click.Path(exists=True, file_okay=True, dir_okay=False), help="Path to the QA data file.")
@click.option("--memory_dir", type=click.Path(file_okay=False, dir_okay=True), required=True, help="Directory to save the memory data.")
@click.option("--compress_ratio", type=int, default=4, help="Compression ratio for the memory.")
@click.option("--mem_model_name", type=str, default="TommyChien/memorag-qwen2-7b-inst", help="Memory model name or path.")
@click.option("--result_save_path", type=click.Path(dir_okay=False, exists=False))
def cli(corpus_data_path, qa_data_path, memory_dir, compress_ratio, mem_model_name, result_save_path):
    """
    Command-line interface to create and save a MemoRAG memory from a corpus data file.
    """
    if not result_save_path.endswith(".parquet"):
        raise ValueError("result_save_path must end with .parquet")

    
    qa_df = pd.read_parquet(qa_data_path)
    queries = qa_df['query'].tolist()

    model = Memory(mem_model_name, beacon_ratio=compress_ratio,
                    load_in_4bit=False, enable_flash_attn=False)
    
    memory_path = os.path.join(memory_dir, "memory.bin")

    if not os.path.exists(memory_path):
        print("Creating memory from corpus data...")
        corpus_df = pd.read_parquet(corpus_data_path)
        contents = "\n\n".join(corpus_df['contents'].tolist())
        model.memorize(contents)
        if memory_dir:
            model.save(memory_path)
            print(f"Memory saved to {memory_path}")
    else:
        print(f"Loading existing memory from {memory_path}...")
        model.load(memory_path)
        print("Memory loaded.")
   
    answers = []
    for query in tqdm(queries):
        res = model.generate(PROMPT_TEMPLATE, query,
        max_new_tokens=4096, temperature=0.7, top_p=0.9, do_sample=False,
        with_cache=True)[0]
        answers.append(res)
    qa_df["generated_answer"] = answers
    qa_df.to_parquet(result_save_path, index=False)
    print(f"Results saved to {result_save_path}")

if __name__ == "__main__":
    cli()
