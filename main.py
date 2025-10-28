from memorag.memorag import MemoRAG
import click
import pandas as pd

@click.command()
@click.option("--corpus_data_path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help="Path to the corpus data file.")
@click.option("--memory_dir", type=click.Path(file_okay=False, dir_okay=True), required=True, help="Directory to save the memory data.")
@click.option("--compress_ratio", type=int, default=4, help="Compression ratio for the memory.")
def cli(corpus_data_path, memory_dir, compress_ratio):
    """
    Command-line interface to create and save a MemoRAG memory from a corpus data file.
    """
    corpus_df = pd.read_parquet(corpus_data_path)
    contents = "\n\n".join(corpus_df['contents'].tolist())

    pipe = MemoRAG(
        mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
        ret_model_name_or_path="intfloat/multilingual-e5-large-instruct",
        gen_model_name_or_path="Qwen/Qwen3-8B",
        beacon_ratio=compress_ratio,
    )
    pipe.memorize(contents, save_dir=memory_dir, print_stats=True)
    print(f"Memory saved to {memory_dir}")
    query = "What are the central allegations linking these multiple lawsuits against Prudential Financial, and how have they progressed through the courts?"

    res = pipe(context=contents, query=query,
    task_type="memorag", max_new_tokens=4096)
    print(f"MemoRAG generated answer: \n{res}")


if __name__ == "__main__":
    cli()
