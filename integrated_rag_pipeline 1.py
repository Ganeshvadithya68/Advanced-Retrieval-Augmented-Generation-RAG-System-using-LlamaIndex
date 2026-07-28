import argparse 
import os 
from pathlib import Path 
from typing import Optional 
 
from dotenv import load_dotenv 
 
from llama_index.core import ( 
    Document, 
    SimpleDirectoryReader, 
    StorageContext, 
    VectorStoreIndex, 
    load_index_from_storage, 
) 
from llama_index.core.node_parser import ( 
    HierarchicalNodeParser, 
    SentenceWindowNodeParser, 
    SimpleNodeParser, 
    get_leaf_nodes, 
) 
from llama_index.core.postprocessor import ( 
    MetadataReplacementPostProcessor, 
    SentenceTransformerRerank, 
) 
from llama_index.core.query_engine import RetrieverQueryEngine 
from llama_index.core.retrievers import AutoMergingRetriever 
from llama_index.llms.openai import OpenAI 
 
# same embedding model name as in the notebooks (local HF model) 
EMBED = "local:BAAI/bge-small-en-v1.5" 
RERANK_MODEL = "BAAI/bge-reranker-base" 
 
 
def get_openai_key(from_cli: Optional[str]) -> str: 
    if from_cli: 
        return from_cli 
    key = os.environ.get("OPENAI_API_KEY") or 
os.environ.get("openai_api_key") 
    if not key: 
        raise ValueError( 
            "Missing OPENAI_API_KEY. Put it in .env or use --openai-api
key." 
        ) 
    return key 
 
 
def read_text_file(path: Path): 
    if not path.exists(): 
        raise FileNotFoundError(f"No such file: {path}") 
    return SimpleDirectoryReader(input_files=[str(path)]).load_data() 
 
 
def build_basic_engine(documents, llm): 
    parser = SimpleNodeParser.from_defaults(chunk_size=64, 
chunk_overlap=2) 
    nodes = parser.get_nodes_from_documents(documents) 
    index = VectorStoreIndex(nodes, embed_model=EMBED) 
    return index.as_query_engine(similarity_top_k=2, llm=llm) 
 
 
def build_sentence_window_engine(big_doc, llm, save_folder: Path, 
use_save: bool): 
    node_parser = SentenceWindowNodeParser.from_defaults( 
        window_size=2, 
        window_metadata_key="window", 
        original_text_metadata_key="original_text", 
    ) 
 
    if use_save and save_folder.exists(): 
        sc = StorageContext.from_defaults(persist_dir=str(save_folder)) 
        index = load_index_from_storage(sc) 
    else: 
        index = VectorStoreIndex.from_documents( 
            [big_doc], 
            embed_model=EMBED, 
            transformations=[node_parser], 
        ) 
        if use_save: 
            index.storage_context.persist(persist_dir=str(save_folder)) 
 
    postproc = 
MetadataReplacementPostProcessor(target_metadata_key="window") 
    rerank = SentenceTransformerRerank(top_n=2, model=RERANK_MODEL) 
 
    return index.as_query_engine( 
        similarity_top_k=6, 
        node_postprocessors=[postproc, rerank], 
        llm=llm, 
    ) 
 
 
def build_auto_merge(engine_llm, big_doc, save_folder: Path, use_save: 
bool): 
    node_parser = HierarchicalNodeParser.from_defaults( 
        chunk_sizes=[256, 64, 16], 
        chunk_overlap=8, 
    ) 
    nodes = node_parser.get_nodes_from_documents([big_doc]) 
    leaf_nodes = get_leaf_nodes(nodes) 
 
    if use_save and save_folder.exists(): 
        sc = StorageContext.from_defaults(persist_dir=str(save_folder)) 
        auto_index = load_index_from_storage(sc) 
    else: 
        sc = StorageContext.from_defaults() 
        sc.docstore.add_documents(nodes) 
        auto_index = VectorStoreIndex( 
            leaf_nodes, 
            storage_context=sc, 
            embed_model=EMBED, 
        ) 
        if use_save: 
            
auto_index.storage_context.persist(persist_dir=str(save_folder)) 
 
    base = auto_index.as_retriever(similarity_top_k=12) 
    retriever = AutoMergingRetriever(base, auto_index.storage_context, 
verbose=True) 
    rerank = SentenceTransformerRerank(top_n=2, model=RERANK_MODEL) 
    return RetrieverQueryEngine.from_args( 
        retriever, 
        node_postprocessors=[rerank], 
        llm=engine_llm, 
    ) 
 
 
def print_answer_block(title, response, max_src=2): 
    print("\n" + "=" * 60) 
    print(title) 
    print("=" * 60) 
    print(response) 
    srcs = getattr(response, "source_nodes", None) or [] 
    if srcs: 
        print("\nTop chunks:") 
        for j in range(min(max_src, len(srcs))): 
            n = srcs[j] 
            t = n.node.get_text() if hasattr(n, "node") else str(n) 
            t = t.replace("\n", " ").strip() 
            if len(t) > 300: 
                t = t[:300] + "..." 
            print(f"  {j + 1}. {t}") 
 
 
def print_retrieval_block(title, nodes, max_n=4): 
    print("\n" + "=" * 60) 
    print(title) 
    print("=" * 60) 
    for j in range(min(max_n, len(nodes))): 
        n = nodes[j] 
        t = n.node.get_text() if hasattr(n, "node") else str(n) 
        t = t.replace("\n", " ").strip() 
        if len(t) > 400: 
            t = t[:400] + "..." 
        sc = getattr(n, "score", None) 
        if isinstance(sc, (int, float)): 
            print(f"{j + 1}. (score={sc:.4f}) {t}") 
        else: 
            print(f"{j + 1}. {t}") 
 
 
def main(): 
    ap = argparse.ArgumentParser() 
    ap.add_argument("--input-file", default="data/Henry.txt") 
    ap.add_argument("--query", default="Who is the beautiful person in 
Hong Kong?") 
    ap.add_argument("--llm-model", default="gpt-3.5-turbo") 
    ap.add_argument("--temperature", type=float, default=0.1) 
    ap.add_argument("--openai-api-key", default=None) 
    ap.add_argument("--persist", action="store_true") 
    ap.add_argument("--cache-dir", default=".rag_cache") 
    ap.add_argument( 
        "--retrieval-only", 
        action="store_true", 
        help="Only show retrieved text, no OpenAI chat (good when you 
have no credits)", 
    ) 
    args = ap.parse_args() 
 
    repo = Path(__file__).resolve().parent 
    env_path = repo / ".env" 
    if env_path.exists(): 
        load_dotenv(str(env_path)) 
 
    data_path = (repo / args.input_file).resolve() 
    cache = (repo / args.cache_dir).resolve() 
    cache.mkdir(parents=True, exist_ok=True) 
 
    llm = None 
    if not args.retrieval_only: 
        k = get_openai_key(args.openai_api_key) 
        os.environ["OPENAI_API_KEY"] = k 
        llm = OpenAI(model=args.llm_model, temperature=args.temperature) 
 
    print("Loading:", data_path) 
    docs = read_text_file(data_path) 
    one_doc = Document(text="\n\n".join(d.text for d in docs)) 
 
    print("\n[1/3] Basic RAG...") 
    basic_qe = build_basic_engine(docs, llm) 
 
    print("[2/3] Sentence window...") 
    sw_qe = build_sentence_window_engine( 
        one_doc, llm, cache / "sentence_index", args.persist 
    ) 
 
    print("[3/3] Auto-merging...") 
    am_qe = build_auto_merge(llm, one_doc, cache / "merging_index", 
args.persist) 
 
    q = args.query 
    if args.retrieval_only: 
        print_retrieval_block("1) Basic RAG", 
basic_qe.retriever.retrieve(q)) 
        print_retrieval_block("2) Sentence window", 
sw_qe.retriever.retrieve(q)) 
        print_retrieval_block("3) Auto-merging", 
am_qe.retriever.retrieve(q)) 
    else: 
        print_answer_block("1) Basic RAG", basic_qe.query(q)) 
        print_answer_block("2) Sentence window", sw_qe.query(q)) 
        print_answer_block("3) Auto-merging", am_qe.query(q)) 
 
 
if __name__ == "__main__": 
main() 