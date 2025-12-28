# Código para entrenamiento de la red neuronal
import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import textwrap
from typing import Dict

def run(cmd_list):
    """
    Ejecuta un comando del sistema en una subshell.
    Imprime el comando antes de ejecutarlo y lanza una excepción si falla.
    """
    print("[RUN]", " ".join(cmd_list))
    r = subprocess.run(cmd_list, stdout=sys.stdout, stderr=sys.stderr)
    if r.returncode != 0:
        raise SystemExit(f"Comando falló: {' '.join(cmd_list)}")

def ensure_ultralytics():
    """
    Verifica si la librería 'ultralytics' está instalada.
    Si no lo está, intenta instalarla automáticamente usando pip.
    También maneja errores comunes de importación en Windows (dependencias de PyTorch).
    """
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:  # pragma: no cover
        print("ultralytics no instalado. Instalando...")
        run([sys.executable, "-m", "pip", "install", "ultralytics"])
    # Reintentar import después de instalar
    try:
        import ultralytics  # noqa: F401
    except OSError as e:
        # Errores típicos en Windows cuando torch falla por dependencias nativas (c10.dll, WinError 126)
        print("[ERROR] Fallo al importar dependencias nativas (PyTorch) requeridas por ultralytics:")
        print(f"        {e}")
        print("\nCausa común en Windows: Falta Microsoft Visual C++ Redistributable (x64).\n"
              "Descárgalo e instálalo desde:\n"
              "  https://aka.ms/vs/16/release/vc_redist.x64.exe\n"
              "Luego cierra y reabre la terminal, y vuelve a ejecutar el script.")
        print("\nAlternativa (si SIGUE fallando): forzar instalación de PyTorch CPU-only y dependencias:")
        print(f"  {sys.executable} -m pip uninstall -y torch torchvision torchaudio")
        print(f"  {sys.executable} -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio")
        raise

def resolve_dataset(args):
    """
    Determina qué archivo data.yaml usar para el entrenamiento.

    Lógica de prioridad:
    1. Si se pasa --data (ruta explícita a data.yaml) -> usarla directamente.
    2. Si ya existe out_dir/data.yaml y no se fuerza la reconstrucción (--force-rebuild) -> usarla.
    3. Si NO hay --coco / --images-dir y tampoco --data -> error (faltan datos).
    4. Si hay datos COCO, ejecutar el script de conversión COCO -> YOLO.
    """
    # 1) Ruta directa a data.yaml proporcionada por el usuario
    if args.data:
        data_yaml = Path(args.data)
        if not data_yaml.is_file():
            raise SystemExit(f"--data apunta a un archivo inexistente: {data_yaml}")
        print(f"[INFO] Usando dataset YOLO existente: {data_yaml}")
        return data_yaml
    # 2) Verificar si ya existe un dataset preparado en el directorio de salida
    out_dir = Path(args.out_dir)
    data_yaml = out_dir / 'data.yaml'
    if data_yaml.exists() and not args.force_rebuild:
        print(f"[INFO] Dataset YOLO detectado: {data_yaml} (usa --force-rebuild para regenerar)")
        return data_yaml
    # 3) Validar que tenemos los datos necesarios para convertir desde COCO si no hay dataset YOLO
    if not args.coco or not args.images_dir:
        raise SystemExit("Falta --data (data.yaml existente) o bien --coco y --images-dir para convertir desde COCO.")
    # 4) Ejecutar conversión COCO -> YOLO llamando al script externo
    converter = Path(__file__).parent / 'convert_coco_to_yolo.py'
    if not converter.exists():
        raise SystemExit("No se encuentra convert_coco_to_yolo.py. Asegúrate de que está en yolo_pipeline/.")

    cmd = [
        sys.executable,
        str(converter),
        '--coco', args.coco,
        '--images-dir', args.images_dir,
        '--out-dir', args.out_dir,
        '--val-ratio', str(args.val_ratio),
        '--seed', str(args.seed)
    ]
    run(cmd)
    if not data_yaml.exists():
        raise SystemExit("Falló la conversión: no se generó data.yaml")
    return data_yaml

def parse_extra(pairs) -> Dict[str, str]:
    """
    Parsea argumentos extra pasados en formato clave=valor.
    Útil para pasar hiperparámetros adicionales a YOLO que no están explícitamente definidos en argparse.
    """
    extras: Dict[str, str] = {}
    if not pairs:
        return extras
    for p in pairs:
        if '=' not in p:
            print(f"[WARN] Ignorando --extra '{p}' (falta '=')")
            continue
        k, v = p.split('=', 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            print(f"[WARN] Ignorando --extra '{p}' (clave o valor vacío)")
            continue
        extras[k] = v
    return extras

def train(args, data_yaml):
    """
    Inicia el proceso de entrenamiento usando la librería Ultralytics YOLO.
    Configura los parámetros de entrenamiento basándose en los argumentos.
    """
    from ultralytics import YOLO
    print(f"[INFO] Cargando modelo base: {args.model}")
    model = YOLO(args.model)

    # batch=0 -> dejar que YOLO decida automáticamente el tamaño del batch según la memoria GPU
    batch_value = None if args.batch == 0 else args.batch

    # Diccionario de argumentos para la función model.train()
    train_kwargs = {
        'data': str(data_yaml),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'project': args.project,
        'name': args.run_name,
    }
    if batch_value is not None:
        train_kwargs['batch'] = batch_value
    if args.device:
        train_kwargs['device'] = args.device
    if args.patience is not None:
        train_kwargs['patience'] = args.patience

    # Añadir argumentos extra (overrides)
    extra_overrides = parse_extra(args.extra)
    # Tip: convertir strings numéricos a int/float cuando sea posible para que YOLO los entienda
    for k, v in list(extra_overrides.items()):
        if v.isdigit():
            extra_overrides[k] = int(v)
        else:
            try:
                extra_overrides[k] = float(v)
            except ValueError:
                pass
    train_kwargs.update(extra_overrides)

    print("[INFO] Iniciando entrenamiento con parámetros (finales):")
    for k, v in train_kwargs.items():
        print(f"  - {k}: {v}")

    # Ejecutar entrenamiento
    model.train(**train_kwargs)
    print("[OK] Entrenamiento finalizado. Pesos en carpeta runs/detect/<nombre>/weights/")

def parse_args():
    """
    Define y parsea los argumentos de línea de comandos disponibles.
    """
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Pipeline unificado:
            1) Convierte COCO -> YOLO (si es necesario)
            2) Ejecuta entrenamiento con Ultralytics YOLO

            Ejemplo:
                python train_yolo.py \
                  --coco annotations.json \
                  --images-dir imagenes \
                  --out-dir yolo_dataset \
                  --model yolov8n.pt \
                  --epochs 50 \
                  --imgsz 640
            """
        )
    )
    # Nuevo: permitir dataset ya existente en formato YOLO
    ap.add_argument('--data', help='Ruta a un data.yaml YA existente (omite conversión)')
    ap.add_argument('--coco', help='Ruta al archivo COCO JSON exportado de Label Studio (si se requiere conversión)')
    ap.add_argument('--images-dir', help='Directorio con las imágenes originales (requerido si se usa --coco)')
    ap.add_argument('--out-dir', default='yolo_dataset', help='Directorio destino del dataset YOLO')
    ap.add_argument('--val-ratio', type=float, default=0.2, help='Proporción validación')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force-rebuild', action='store_true', help='Forzar reconversión aunque exista data.yaml')
    ap.add_argument('--model', default='yolov8n.pt', help='Checkpoint base (yolov8n.pt, yolov8s.pt, etc.)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16, help='Tamaño de batch (0 = auto)')
    ap.add_argument('--project', default='runs', help='Carpeta principal de resultados')
    ap.add_argument('--run-name', default='train', help='Nombre del subdirectorio de la corrida')
    ap.add_argument('--device', default='', help='Dispositivo ("0" para primera GPU, "cpu" para forzar CPU)')
    ap.add_argument('--patience', type=int, default=50, help='Épocas sin mejora antes de early stopping (Ultralytics)')
    ap.add_argument('--extra', action='append', help='Overrides adicionales key=val (puedes repetir). Ej: --extra lr0=0.005 --extra optimizer=AdamW')
    return ap.parse_args()

def main():
    """
    Función principal que orquesta la verificación de dependencias, preparación de datos y entrenamiento.
    """
    args = parse_args()
    ensure_ultralytics()
    data_yaml = resolve_dataset(args)
    train(args, data_yaml)

if __name__ == '__main__':
    main()