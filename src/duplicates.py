"""Detecção de imagens duplicadas e correção de vazamento entre splits.

Se a mesma imagem (ou uma cópia quase igual) está no treino e no teste, o modelo
"já viu" a resposta e as métricas ficam infladas. Este módulo:

1. Calcula, para cada imagem:
   - MD5 dos bytes            -> duplicatas exatas (arquivo idêntico)
   - dHash perceptual 64 bits -> candidatas a quase iguais (mesma estrutura)
   - miniatura colorida 16x16 -> confirma as candidatas pela cor
   Uma imagem é "quase igual" a outra só se passar nos DOIS testes. Só o dHash
   não basta: neste dataset ele junta frutas diferentes fotografadas no mesmo
   fundo (ex.: mangas diferentes no mesmo papel branco).
2. Agrupa as duplicatas da mesma classe.
3. Com --fix, coloca cada grupo inteiro num só split (prioridade train > val > test)
   e reescreve data/splits.csv mudando só essas linhas.

Uso:
    python -m src.duplicates                 # só relatório (não muda nada)
    python -m src.duplicates --fix           # corrige data/splits.csv
    python -m src.duplicates --threshold 2   # critério de "quase igual" mais rígido

Saídas:
    data/duplicates.csv                 grupos de duplicatas (versionado)
    data/duplicates_entre_classes.csv   mesma imagem em classes diferentes (só aviso)
    duplicates_preview.png              exemplos para conferir visualmente
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from src import config
from src.data_loader import SPLITS, download_dataset, read_splits

DEFAULT_THRESHOLD = 4         # bits diferentes no dHash (de 64) para ser candidata
DEFAULT_MAX_COLOR_DIFF = 6.0  # diferença média por pixel (0-255) na miniatura 16x16
# Os cortes foram escolhidos olhando pares deste dataset: até ~6 são praticamente
# sempre o mesmo objeto; acima de ~7 começam a aparecer frutas diferentes no mesmo fundo.
THUMB_SIZE = 16
SPLIT_PRIORITY = {name: i for i, name in enumerate(SPLITS)}  # train > val > test
HASH_CACHE_PATH = config.CACHE_DIR / "image_hashes.npz"
REPORT_PATH = config.DATA_DIR / "duplicates.csv"
CROSS_CLASS_PATH = config.DATA_DIR / "duplicates_entre_classes.csv"
NO_HASH = np.uint64(0)  # imagem que não abriu; fica de fora da busca por quase iguais


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------

def dhash(img: Image.Image) -> int:
    """Hash perceptual de 64 bits: compara cada pixel com o vizinho da direita
    numa versão 9x8 em tons de cinza. Imagens visualmente iguais dão hashes iguais
    ou com poucos bits diferentes."""
    small = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _hash_file(path: str):
    # Alguns PNGs com paleta geram um aviso inofensivo do Pillow ao converter para RGB
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    with open(path, "rb") as f:
        data = f.read()
    md5 = hashlib.md5(data).hexdigest()
    try:
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (64, 64))  # JPEG: decodifica já reduzido, bem mais rápido
        img = img.convert("RGB")
        h = dhash(img)
        thumb = np.asarray(img.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BOX), dtype=np.uint8)
    except Exception:
        h, thumb = 0, np.zeros((THUMB_SIZE, THUMB_SIZE, 3), np.uint8)
    return md5, h, thumb


def compute_hashes(root: Path, rows: list[dict], workers: int | None = None):
    """Retorna (md5s, dhashes, thumbs) na mesma ordem de `rows`.

    Guarda os resultados em data/cache/image_hashes.npz; nas próximas execuções
    só recalcula imagens novas ou que mudaram de tamanho.
    """
    cache = {}
    if HASH_CACHE_PATH.exists():
        z = np.load(HASH_CACHE_PATH)
        cache = {
            (p, int(s)): (m, int(h), t)
            for p, s, m, h, t in zip(z["paths"], z["sizes"], z["md5s"], z["dhashes"], z["thumbs"])
        }

    sizes = [os.path.getsize(root / r["path"]) for r in rows]
    todo = [i for i, r in enumerate(rows) if (r["path"], sizes[i]) not in cache]
    if todo:
        print(f"Calculando hashes de {len(todo)} imagens ({workers or os.cpu_count()} processos)...")
        with Pool(workers) as pool:
            paths = [str(root / rows[i]["path"]) for i in todo]
            for k, (i, result) in enumerate(zip(todo, pool.imap(_hash_file, paths, chunksize=64)), 1):
                cache[(rows[i]["path"], sizes[i])] = result
                if k % 5000 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}")
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        keys, values = list(cache), list(cache.values())
        np.savez(
            HASH_CACHE_PATH,
            paths=np.array([p for p, _ in keys]),
            sizes=np.array([s for _, s in keys], dtype=np.int64),
            md5s=np.array([v[0] for v in values]),
            dhashes=np.array([v[1] for v in values], dtype=np.uint64),
            thumbs=np.stack([v[2] for v in values]),
        )

    entries = [cache[(r["path"], sizes[i])] for i, r in enumerate(rows)]
    md5s = [e[0] for e in entries]
    dhashes = np.array([e[1] for e in entries], dtype=np.uint64)
    thumbs = np.stack([e[2] for e in entries])
    return md5s, dhashes, thumbs


# ---------------------------------------------------------------------------
# Busca de pares
# ---------------------------------------------------------------------------

def find_exact_pairs(md5s: list[str]) -> list[tuple[int, int]]:
    by_md5 = defaultdict(list)
    for i, m in enumerate(md5s):
        by_md5[m].append(i)
    return [(idx[0], j) for idx in by_md5.values() for j in idx[1:]]


def color_diff(thumbs: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Diferença média absoluta (0-255) entre as miniaturas coloridas dos pares (i, j)."""
    return np.abs(thumbs[i].astype(np.int16) - thumbs[j].astype(np.int16)).mean(axis=(1, 2, 3))


def find_near_pairs(
    dhashes: np.ndarray,
    thumbs: np.ndarray,
    threshold: int = DEFAULT_THRESHOLD,
    max_color_diff: float = DEFAULT_MAX_COLOR_DIFF,
    block: int = 512,
) -> list[tuple[int, int]]:
    """Pares (i, j), i < j, quase iguais: dHash com até `threshold` bits
    diferentes E miniaturas com diferença de cor <= `max_color_diff`.

    Compara todos contra todos (O(n²)), mas cada comparação é um XOR de 64 bits
    vetorizado com numpy, então leva só alguns segundos para ~30 mil imagens.
    """
    valid = np.where(dhashes != NO_HASH)[0]
    h = dhashes[valid]
    pairs = []
    for start in range(0, len(h), block):
        a = h[start:start + block]
        dist = np.bitwise_count(a[:, None] ^ h[None, start:])  # só o triângulo superior
        ii, jj = np.nonzero(dist <= threshold)
        keep = jj > ii  # tira a diagonal (imagem com ela mesma)
        i, j = valid[ii[keep] + start], valid[jj[keep] + start]
        ok = color_diff(thumbs, i, j) <= max_color_diff
        pairs.extend(zip(i[ok].tolist(), j[ok].tolist()))
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        self.parent[self.find(a)] = self.find(b)


# ---------------------------------------------------------------------------
# Análise e correção
# ---------------------------------------------------------------------------

def find_duplicates(
    root: Path,
    rows: list[dict],
    threshold: int = DEFAULT_THRESHOLD,
    max_color_diff: float = DEFAULT_MAX_COLOR_DIFF,
    workers=None,
):
    """Retorna (grupos, pares_entre_classes).

    - grupos: lista de dicts {members: [índices], kind: 'exata'|'quase'}, só
      imagens da mesma classe, com 2+ membros.
    - pares_entre_classes: (i, j, kind) com a mesma imagem em classes diferentes.
    """
    md5s, dhashes, thumbs = compute_hashes(root, rows, workers)
    unreadable = int((dhashes == NO_HASH).sum())
    if unreadable:
        print(f"Aviso: {unreadable} imagens não abriram (só entram na busca exata)")

    exact = find_exact_pairs(md5s)
    near = find_near_pairs(dhashes, thumbs, threshold, max_color_diff)

    uf = UnionFind(len(rows))
    cross_class = []
    seen = set()
    for pairs, kind in ((exact, "exata"), (near, "quase")):
        for i, j in pairs:
            if rows[i]["class_name"] == rows[j]["class_name"]:
                uf.union(i, j)
            elif (min(i, j), max(i, j)) not in seen:
                seen.add((min(i, j), max(i, j)))
                cross_class.append((i, j, "exata" if md5s[i] == md5s[j] else kind))

    members = defaultdict(list)
    for i in range(len(rows)):
        members[uf.find(i)].append(i)
    groups = [
        {"members": m, "kind": "exata" if len({md5s[i] for i in m}) == 1 else "quase"}
        for m in members.values() if len(m) > 1
    ]
    groups.sort(key=lambda g: (-len(g["members"]), rows[g["members"][0]]["path"]))
    return groups, cross_class


def plan_fix(rows: list[dict], groups: list[dict]) -> dict[int, str]:
    """{índice: novo_split} para que cada grupo fique num só split (train > val > test)."""
    changes = {}
    for g in groups:
        splits = {rows[i]["split"] for i in g["members"]}
        if len(splits) > 1:
            target = min(splits, key=SPLIT_PRIORITY.__getitem__)
            changes.update({i: target for i in g["members"] if rows[i]["split"] != target})
    return changes


def write_report(rows, groups, cross_class, changes):
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "kind", "path", "class_name", "split_before", "split_after"])
        for gid, g in enumerate(groups):
            for i in sorted(g["members"], key=lambda i: rows[i]["path"]):
                r = rows[i]
                w.writerow([gid, g["kind"], r["path"], r["class_name"], r["split"], changes.get(i, r["split"])])
    with open(CROSS_CLASS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "path_a", "class_a", "split_a", "path_b", "class_b", "split_b"])
        for i, j, kind in cross_class:
            a, b = rows[i], rows[j]
            w.writerow([kind, a["path"], a["class_name"], a["split"], b["path"], b["class_name"], b["split"]])


def apply_fix(rows: list[dict], changes: dict[int, str], splits_path: Path | None = None):
    """Reescreve data/splits.csv mudando só a coluna split das linhas em `changes`."""
    new_rows = [{**r, "split": changes.get(i, r["split"])} for i, r in enumerate(rows)]
    with open(splits_path or config.SPLITS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class_name", "split"])
        w.writeheader()
        w.writerows(new_rows)
    return new_rows


def print_summary(rows, groups, cross_class, changes):
    in_groups = sum(len(g["members"]) for g in groups)
    kinds = Counter(g["kind"] for g in groups)
    leaking = [g for g in groups if len({rows[i]["split"] for i in g["members"]}) > 1]

    print(f"\n{len(rows)} imagens analisadas")
    print(f"Grupos de duplicatas (mesma classe): {len(groups)} "
          f"({kinds['exata']} exatos, {kinds['quase']} quase iguais), {in_groups} imagens")
    print(f"Grupos espalhados em mais de um split (vazamento): {len(leaking)}")
    print(f"Imagens que precisam mudar de split: {len(changes)}")
    for (src, dst), n in sorted(Counter((rows[i]["split"], s) for i, s in changes.items()).items()):
        print(f"  {src:5s} -> {dst:5s}: {n}")

    before = Counter(r["split"] for r in rows)
    after = Counter(changes.get(i, r["split"]) for i, r in enumerate(rows))
    print("\nImagens por split (antes -> depois):")
    for s in SPLITS:
        print(f"  {s:5s}: {before[s]:6d} -> {after[s]:6d}")

    if cross_class:
        print(f"\nAviso: {len(cross_class)} pares de imagens iguais em CLASSES diferentes "
              f"(possível erro de rótulo; não alterados). Veja {CROSS_CLASS_PATH.name}")


def save_preview(root, rows, groups, path="duplicates_preview.png", n_groups=8, max_per_group=5,
                 class_name: str | None = None):
    """Grade com alguns grupos (primeiro os que vazam entre splits) para conferir a olho.

    class_name: mostra só grupos dessa classe (ex.: 'Mango__Healthy').
    """
    import matplotlib.pyplot as plt  # só salva arquivo; não troca o backend (quebraria o notebook)

    if class_name:
        groups = [g for g in groups if rows[g["members"][0]]["class_name"] == class_name]
    leaking = [g for g in groups if len({rows[i]["split"] for i in g["members"]}) > 1]
    shown = (leaking + [g for g in groups if g not in leaking])[:n_groups]
    if not shown:
        print(f"Nenhum grupo de duplicatas{' em ' + class_name if class_name else ''}.")
        return
    fig, axes = plt.subplots(len(shown), max_per_group, figsize=(2.4 * max_per_group, 2.6 * len(shown)),
                             squeeze=False)
    for row_axes, g in zip(axes, shown):
        for ax, i in zip(row_axes, g["members"][:max_per_group]):
            with Image.open(root / rows[i]["path"]) as img:
                img.thumbnail((200, 200))
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{rows[i]['split']} | {g['kind']}\n{Path(rows[i]['path']).name[:24]}", fontsize=7)
        for ax in row_axes:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"Prévia salva em {path}")


def save_cross_class_preview(root, rows, cross_class, path="duplicates_entre_classes.png"):
    """Pares de imagens iguais em classes diferentes, lado a lado (possíveis erros de rótulo)."""
    import matplotlib.pyplot as plt  # só salva arquivo; não troca o backend (quebraria o notebook)

    if not cross_class:
        return
    fig, axes = plt.subplots(len(cross_class), 2, figsize=(5, 2.6 * len(cross_class)), squeeze=False)
    for (i, j, kind), row_axes in zip(cross_class, axes):
        for ax, k in zip(row_axes, (i, j)):
            with Image.open(root / rows[k]["path"]) as img:
                img.thumbnail((200, 200))
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{rows[k]['class_name']} | {kind}\n{Path(rows[k]['path']).name}", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"Prévia dos pares entre classes salva em {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Procura imagens duplicadas entre os splits.")
    parser.add_argument("--fix", action="store_true", help="aplica a correção em data/splits.csv")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"bits diferentes no dHash para ser candidata (padrão {DEFAULT_THRESHOLD})")
    parser.add_argument("--max-color-diff", type=float, default=DEFAULT_MAX_COLOR_DIFF,
                        help=f"diferença de cor máxima, 0-255 (padrão {DEFAULT_MAX_COLOR_DIFF:g}); "
                             "-1 = só duplicatas exatas")
    parser.add_argument("--data-dir", help="pasta do dataset (padrão: kagglehub)")
    parser.add_argument("--workers", type=int, help="processos para calcular os hashes")
    parser.add_argument("--no-preview", action="store_true", help="não gera as imagens de prévia")
    parser.add_argument("--preview-class", help="prévia só desta classe, ex.: Mango__Healthy")
    parser.add_argument("--preview-groups", type=int, default=8, help="quantos grupos mostrar (padrão 8)")
    args = parser.parse_args(argv)

    root = Path(args.data_dir) if args.data_dir else download_dataset()
    rows = read_splits()
    groups, cross_class = find_duplicates(root, rows, args.threshold, args.max_color_diff, args.workers)
    changes = plan_fix(rows, groups)

    print_summary(rows, groups, cross_class, changes)
    write_report(rows, groups, cross_class, changes)
    print(f"Relatório salvo em {REPORT_PATH}")
    if not args.no_preview:
        suffix = f"_{args.preview_class}" if args.preview_class else ""
        save_preview(root, rows, groups, f"duplicates_preview{suffix}.png",
                     n_groups=args.preview_groups, class_name=args.preview_class)
        save_cross_class_preview(root, rows, cross_class)

    if args.fix and changes:
        apply_fix(rows, changes)
        print(f"\n{config.SPLITS_PATH} atualizado ({len(changes)} imagens mudaram de split).")
        print("Apague data/cache/ (exceto image_hashes.csv) para liberar o cache antigo.")
    elif changes:
        print("\nNada foi alterado. Rode com --fix para corrigir data/splits.csv.")
    else:
        print("\nNenhum vazamento entre splits.")


if __name__ == "__main__":
    main()
