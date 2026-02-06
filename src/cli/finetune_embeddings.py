#!/usr/bin/env python3
"""
finetune_embeddings - CLI para fine-tuning del modelo de embeddings.

Uso:
    python -m src.cli.finetune_embeddings generate [opciones]
    python -m src.cli.finetune_embeddings validate <archivo>
    python -m src.cli.finetune_embeddings split <archivo>
    python -m src.cli.finetune_embeddings launch <archivo>
    python -m src.cli.finetune_embeddings status <job_id>
    python -m src.cli.finetune_embeddings list
    python -m src.cli.finetune_embeddings cancel <job_id>
    python -m src.cli.finetune_embeddings test <model_id>
"""

import argparse
import logging
import sys
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Configura rutas del proyecto."""
    project_root = Path(__file__).parent.parent.parent
    return {
        "indices_dir": project_root / "indices",
        "finetuning_dir": project_root / "finetuning_data",
        "config_dir": project_root / "config",
    }


def cmd_generate(args):
    """Genera pares de entrenamiento sintéticos."""
    paths = setup_paths()
    chunks_path = paths["indices_dir"] / "chunks.pkl"

    if not chunks_path.exists():
        print("No hay chunks indexados. Ejecuta ingest_library primero.")
        sys.exit(1)

    with open(chunks_path, "rb") as f:
        chunks_store = pickle.load(f)

    # Filtrar solo micro chunks
    from ..ingestion.chunker import ChunkLevel

    micro_chunks = [
        c for c in chunks_store.values()
        if c.level == ChunkLevel.MICRO
    ]

    # Limitar si se especifica
    if args.max_chunks:
        import random
        random.seed(42)
        micro_chunks = random.sample(
            micro_chunks, min(args.max_chunks, len(micro_chunks))
        )

    print(f"Generando pares desde {len(micro_chunks)} chunks...")

    from ..finetuning.pair_generator import SyntheticPairGenerator

    generator = SyntheticPairGenerator(
        queries_per_chunk=args.queries_per_chunk,
        batch_size=args.batch_size,
    )

    pairs = generator.generate_from_chunks(micro_chunks)
    print(f"Generados {len(pairs)} pares")

    # Generar hard negatives si se solicita
    negatives = None
    if args.hard_negatives:
        print("Generando hard negatives...")
        negatives = generator.generate_hard_negatives(pairs, micro_chunks)
        print(f"Hard negatives generados para {len(negatives)} pares")

    # Formatear y guardar
    from ..finetuning.formatter import OpenAIFineTuneFormatter

    formatter = OpenAIFineTuneFormatter()
    output_path = Path(args.output) if args.output else (
        paths["finetuning_dir"] / "training_data.jsonl"
    )

    stats = formatter.format_pairs(pairs, output_path, negatives)
    print(f"\nDatos guardados en: {output_path}")
    print(f"Total pares: {stats['total_pairs']}")
    print(f"Formato: {stats['format']}")


def cmd_validate(args):
    """Valida un archivo de entrenamiento."""
    from ..finetuning.formatter import OpenAIFineTuneFormatter

    formatter = OpenAIFineTuneFormatter()
    stats = formatter.validate_format(Path(args.file))

    print("\nResultado de validación:")
    print(f"  Válido: {'Sí' if stats['valid'] else 'No'}")
    print(f"  Líneas totales: {stats['total_lines']}")
    print(f"  Líneas válidas: {stats['valid_lines']}")
    print(f"  Duplicados: {stats['duplicates']}")
    print(f"  Formato: {stats['format_detected']}")

    if stats["errors"]:
        print(f"\n  Errores ({len(stats['errors'])}):")
        for err in stats["errors"][:10]:
            print(f"    - {err}")


def cmd_split(args):
    """Divide datos en train/val."""
    from ..finetuning.formatter import OpenAIFineTuneFormatter

    formatter = OpenAIFineTuneFormatter()
    input_path = Path(args.file)
    parent = input_path.parent

    train_path = parent / f"{input_path.stem}_train.jsonl"
    val_path = parent / f"{input_path.stem}_val.jsonl"

    stats = formatter.split_train_val(
        input_path, train_path, val_path,
        val_fraction=args.val_fraction,
    )

    print(f"\nSplit completado:")
    print(f"  Train: {stats['train']} ({train_path})")
    print(f"  Val: {stats['val']} ({val_path})")
    print(f"  Fracción val: {stats['val_fraction_actual']:.1%}")


def cmd_launch(args):
    """Lanza un job de fine-tuning."""
    from ..finetuning.launcher import FineTuneLauncher

    launcher = FineTuneLauncher()

    # Subir archivo
    print(f"Subiendo {args.file}...")
    file_id = launcher.upload_training_file(Path(args.file))
    print(f"Archivo subido: {file_id}")

    # Subir validación si existe
    val_file_id = None
    if args.validation_file:
        print(f"Subiendo validación {args.validation_file}...")
        val_file_id = launcher.upload_training_file(
            Path(args.validation_file)
        )

    # Crear job
    print(f"Creando job de fine-tuning...")
    job_id = launcher.create_job(
        training_file_id=file_id,
        model=args.model,
        suffix=args.suffix,
        n_epochs=args.epochs,
        validation_file_id=val_file_id,
    )
    print(f"Job creado: {job_id}")
    print(f"Monitorizar con: python -m src.cli.finetune_embeddings status {job_id}")


def cmd_status(args):
    """Muestra estado de un job."""
    from ..finetuning.launcher import FineTuneLauncher

    launcher = FineTuneLauncher()
    status = launcher.get_job_status(args.job_id)

    print(f"\nEstado del job {status['id']}:")
    print(f"  Status: {status['status']}")
    print(f"  Modelo base: {status['model']}")
    print(f"  Modelo fine-tuned: {status['fine_tuned_model'] or 'N/A'}")
    print(f"  Tokens entrenados: {status['trained_tokens'] or 'N/A'}")
    if status["error"]:
        print(f"  Error: {status['error']}")


def cmd_list(args):
    """Lista jobs recientes."""
    from ..finetuning.launcher import FineTuneLauncher

    launcher = FineTuneLauncher()
    jobs = launcher.list_jobs(limit=args.limit)

    if not jobs:
        print("No hay jobs de fine-tuning")
        return

    print(f"\nJobs recientes ({len(jobs)}):")
    for j in jobs:
        model = j["fine_tuned_model"] or j["model"]
        print(f"  {j['id']}: {j['status']} ({model})")


def cmd_cancel(args):
    """Cancela un job."""
    from ..finetuning.launcher import FineTuneLauncher

    launcher = FineTuneLauncher()
    result = launcher.cancel_job(args.job_id)
    print(f"Job {result['id']}: {result['status']}")


def cmd_test(args):
    """Compara modelo fine-tuned vs base."""
    from ..finetuning.launcher import FineTuneLauncher

    launcher = FineTuneLauncher()

    test_queries = [
        "¿Qué es el algoritmo de Shor?",
        "Explica el entrelazamiento cuántico",
        "¿Cómo funciona la teleportación cuántica?",
        "Diferencias entre BB84 y E91",
        "¿Qué es la decoherencia?",
    ]

    if args.queries_file:
        with open(args.queries_file) as f:
            test_queries = [l.strip() for l in f if l.strip()]

    print(f"Comparando {args.model_id} vs {args.base_model}...")
    result = launcher.test_model(
        model_id=args.model_id,
        test_queries=test_queries,
        base_model=args.base_model,
    )

    print(f"\nResultados:")
    print(f"  Modelo base: {result['base_model']}")
    print(f"  Modelo FT: {result['finetuned_model']}")
    print(f"  Queries: {result['num_queries']}")
    print(f"  Similitud base vs FT: {result['avg_similarity_base_vs_ft']}")


def main():
    """Punto de entrada principal."""
    # Cargar variables de entorno
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Fine-tuning del modelo de embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcomando")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generar pares de entrenamiento")
    p_gen.add_argument("--output", "-o", help="Archivo JSONL de salida")
    p_gen.add_argument("--max-chunks", type=int, help="Máximo de chunks a procesar")
    p_gen.add_argument("--queries-per-chunk", type=int, default=3)
    p_gen.add_argument("--batch-size", type=int, default=5)
    p_gen.add_argument("--hard-negatives", action="store_true",
                       help="Generar hard negatives con BM25")

    # validate
    p_val = subparsers.add_parser("validate", help="Validar archivo de entrenamiento")
    p_val.add_argument("file", help="Archivo JSONL")

    # split
    p_spl = subparsers.add_parser("split", help="Dividir en train/val")
    p_spl.add_argument("file", help="Archivo JSONL de entrada")
    p_spl.add_argument("--val-fraction", type=float, default=0.1)

    # launch
    p_lau = subparsers.add_parser("launch", help="Lanzar job de fine-tuning")
    p_lau.add_argument("file", help="Archivo JSONL de entrenamiento")
    p_lau.add_argument("--model", default="text-embedding-3-small")
    p_lau.add_argument("--suffix", help="Sufijo del modelo")
    p_lau.add_argument("--epochs", type=int, help="Número de epochs")
    p_lau.add_argument("--validation-file", help="Archivo de validación")

    # status
    p_sta = subparsers.add_parser("status", help="Estado de un job")
    p_sta.add_argument("job_id", help="ID del job")

    # list
    p_lst = subparsers.add_parser("list", help="Listar jobs")
    p_lst.add_argument("--limit", type=int, default=10)

    # cancel
    p_can = subparsers.add_parser("cancel", help="Cancelar un job")
    p_can.add_argument("job_id", help="ID del job")

    # test
    p_tst = subparsers.add_parser("test", help="Comparar modelo FT vs base")
    p_tst.add_argument("model_id", help="ID del modelo fine-tuned")
    p_tst.add_argument("--base-model", default="text-embedding-3-large")
    p_tst.add_argument("--queries-file", help="Archivo con queries de test")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "split": cmd_split,
        "launch": cmd_launch,
        "status": cmd_status,
        "list": cmd_list,
        "cancel": cmd_cancel,
        "test": cmd_test,
    }

    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
