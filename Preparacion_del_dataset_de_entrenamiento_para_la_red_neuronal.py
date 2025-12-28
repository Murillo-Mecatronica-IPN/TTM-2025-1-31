# Código para preparación del dataset de entrenamiento para la red neuronal
import argparse
import shutil
from pathlib import Path
import random
import sys
from typing import List, Tuple, Dict

# Extensiones de imagen soportadas
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def parse_args():
    """
    Parsea los argumentos de línea de comandos.
    Define las opciones para directorios de entrada/salida, ratios de split, etc.
    """
    ap = argparse.ArgumentParser(
        description="Organiza un dataset YOLO existente (imágenes + etiquetas .txt) en la estructura estándar con splits (train/val/test) y genera data.yaml"
    )
    ap.add_argument('--images', required=True, help='Directorio con las imágenes fuente (puede estar plano o recursivo)')
    ap.add_argument('--labels', required=True, help='Directorio raíz que contiene los archivos .txt YOLO (puede tener subdirectorios)')
    ap.add_argument('--out-dir', default='yolo_dataset', help='Directorio de salida con estructura final')
    ap.add_argument('--val-ratio', type=float, default=0.2, help='Proporción validación')
    ap.add_argument('--test-ratio', type=float, default=0.0, help='Proporción test (opcional)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--names', help='Lista separada por comas de nombres de clases (si se quiere generar data.yaml)')
    ap.add_argument('--names-file', help='Archivo con un nombre de clase por línea (alternativa a --names)')
    ap.add_argument('--strategy', choices=['copy', 'move', 'symlink'], default='copy', help='Cómo trasladar archivos')
    ap.add_argument('--dry-run', action='store_true', help='Sólo mostrar lo que se haría, sin copiar/mover')
    ap.add_argument('--no-recursive-labels', action='store_true', help='Desactiva búsqueda recursiva de labels (por defecto es recursiva)')
    ap.add_argument('--show-missing', type=int, default=15, help='Muestra hasta N nombres base de imágenes sin label (0 para ocultar)')
    return ap.parse_args()

def read_class_names(args) -> List[str]:
    """
    Lee los nombres de las clases desde un archivo o desde el argumento de línea de comandos.
    Retorna una lista de strings con los nombres de las clases.
    """
    if args.names and args.names_file:
        print('[WARN] Ignorando --names porque se proporcionó --names-file')
    if args.names_file:
        p = Path(args.names_file)
        if not p.is_file():
            sys.exit(f'No existe names-file: {p}')
        return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]
    if args.names:
        return [n.strip() for n in args.names.split(',') if n.strip()]
    return []

def build_label_index(labels_dir: Path, recursive: bool) -> Dict[str, Path]:
    """
    Construye un índice de archivos de etiquetas para búsqueda rápida.
    Mapea el nombre base del archivo (sin extensión) a su ruta completa.
    """
    pattern_iter = labels_dir.rglob('*.txt') if recursive else labels_dir.glob('*.txt')
    index: Dict[str, Path] = {}
    collisions = 0
    for p in pattern_iter:
        stem = p.stem.lower()
        if stem in index:
            collisions += 1
        else:
            index[stem] = p
    if collisions:
        print(f'[WARN] {collisions} colisiones de nombres base de labels (se usa el primero encontrado). Considera aplanar nombres únicos.')
    print(f'[INFO] Index de labels construido: {len(index)} entradas')
    return index

def collect_pairs(images_dir: Path, labels_dir: Path, recursive_labels: bool, show_missing: int) -> List[Tuple[Path, Path]]:
    """
    Empareja imágenes con sus correspondientes archivos de etiquetas.
    Busca recursivamente imágenes y encuentra su etiqueta usando el índice.
    Retorna una lista de tuplas (ruta_imagen, ruta_label).
    """
    label_index = build_label_index(labels_dir, recursive_labels)
    pairs: List[Tuple[Path, Path]] = []
    missing = []
    total_imgs = 0
    for img_path in images_dir.rglob('*'):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        total_imgs += 1
        stem = img_path.stem.lower()
        lbl = label_index.get(stem)
        if not lbl:
            missing.append(stem)
            continue
        pairs.append((img_path, lbl))
    print(f'[INFO] Imágenes con extensión soportada: {total_imgs}')
    print(f'[INFO] Pares válidos imagen+label: {len(pairs)}')
    if missing:
        print(f'[INFO] Imágenes sin label: {len(missing)}')
        if show_missing > 0:
            preview = missing[:show_missing]
            print('[DEBUG] Ejemplos sin label:', ', '.join(preview))
    if not pairs:
        sys.exit('No se encontraron pares imagen+etiqueta válidos. Verifica que los nombres base coincidan y que las extensiones de imagen estén soportadas.')
    return pairs

def split_pairs(pairs: List[Tuple[Path, Path]], val_ratio: float, test_ratio: float, seed: int):
    """
    Divide aleatoriamente los pares de imagen/etiqueta en conjuntos de entrenamiento, validación y prueba.
    """
    random.seed(seed)
    random.shuffle(pairs)
    total = len(pairs)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test

def ensure_dirs(base: Path):
    """
    Crea la estructura de directorios necesaria para YOLO (images/train, labels/train, etc.).
    """
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val', 'images/test', 'labels/test']:
        (base / sub).mkdir(parents=True, exist_ok=True)

def transfer(src: Path, dst: Path, strategy: str, dry: bool):
    """
    Transfiere un archivo desde src a dst usando la estrategia especificada (copiar, mover, enlace simbólico).
    """
    if dry:
        print(f'[DRY] {strategy} {src} -> {dst}')
        return
    if strategy == 'copy':
        shutil.copy2(src, dst)
    elif strategy == 'move':
        shutil.move(str(src), dst)
    elif strategy == 'symlink':
        if dst.exists():
            return
        dst.symlink_to(src.resolve())

def validate_label_file(path: Path, class_count_hint: int | None = None) -> bool:
    """
    Valida el formato de un archivo de etiqueta YOLO.
    Verifica que tenga 5 campos por línea, valores numéricos y coordenadas normalizadas [0, 1].
    """
    ok = True
    for ln_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f'[WARN] {path.name}: línea {ln_no} no tiene 5 campos -> {line}')
            ok = False
            continue
        try:
            cls = int(parts[0])
            nums = [float(x) for x in parts[1:]]
        except ValueError:
            print(f'[WARN] {path.name}: línea {ln_no} contiene valores no numéricos')
            ok = False
            continue
        if any(not (0.0 <= v <= 1.0) for v in nums):
            print(f'[WARN] {path.name}: línea {ln_no} bbox fuera de rango [0,1]: {nums}')
            ok = False
        if class_count_hint is not None and cls >= class_count_hint:
            print(f'[WARN] {path.name}: clase {cls} >= número declarado de clases ({class_count_hint})')
            ok = False
    return ok

def generate_data_yaml(out_dir: Path, class_names: List[str]):
    """
    Genera el archivo data.yaml requerido por YOLO para el entrenamiento.
    Incluye rutas a los directorios de imágenes y nombres de clases.
    """
    if not class_names:
        print('[INFO] No se proporcionaron nombres de clase; omitiendo data.yaml (debes crear uno manualmente).')
        return
    yaml_path = out_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write('path: ./yolo_dataset\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        if any((out_dir / 'images/test').iterdir()):
            f.write('test: images/test\n')
        f.write('names:\n')
        for i, name in enumerate(class_names):
            f.write(f'  {i}: {name}\n')
    print(f'[OK] data.yaml generado en {yaml_path}')

def main():
    """
    Función principal que orquesta todo el proceso de preparación del dataset.
    """
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir = Path(args.out_dir)

    if not images_dir.is_dir():
        sys.exit(f'No existe directorio de imágenes: {images_dir}')
    if not labels_dir.is_dir():
        sys.exit(f'No existe directorio de labels: {labels_dir}')
    if args.val_ratio + args.test_ratio >= 1.0:
        sys.exit('val_ratio + test_ratio debe ser < 1.0')

    class_names = read_class_names(args)

    print('[INFO] Recolectando pares imagen+label...')
    pairs = collect_pairs(images_dir, labels_dir, not args.no_recursive_labels, args.show_missing)

    train, val, test = split_pairs(pairs, args.val_ratio, args.test_ratio, args.seed)
    print(f'[SPLIT] train={len(train)} val={len(val)} test={len(test)}')

    ensure_dirs(out_dir)

    def place(group: List[Tuple[Path, Path]], split: str):
        """
        Función auxiliar para transferir un grupo de archivos (train/val/test) a su destino.
        """
        for img, lbl in group:
            dst_img = out_dir / 'images' / split / img.name
            dst_lbl = out_dir / 'labels' / split / (lbl.name)
            transfer(img, dst_img, args.strategy, args.dry_run)
            transfer(lbl, dst_lbl, args.strategy, args.dry_run)
            validate_label_file(lbl, class_count_hint=len(class_names) if class_names else None)

    place(train, 'train')
    place(val, 'val')
    if test:
        place(test, 'test')

    if not args.dry_run:
        generate_data_yaml(out_dir, class_names)

    print('[DONE] Dataset preparado. Puedes entrenar con:')
    print(f"  python yolo_pipeline/train_yolo.py --data {out_dir / 'data.yaml'} --model yolov8n.pt --epochs 50")

if __name__ == '__main__':
    main()